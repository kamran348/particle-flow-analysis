"""Orchestrate PIV/PTV measurements and exports."""

from __future__ import annotations

from pathlib import Path
from dataclasses import asdict, replace
import json

import cv2
import numpy as np
import pandas as pd

from .analysis import calculate_flow_properties, summarize_field
from .config import (
    AppConfig, CalibrationConfig, ConsistencyConfig, ExclusionConfig, PIVConfig,
    PreprocessingConfig, TrackingConfig, VideoConfig,
)
from .consistency import filter_consistent_tracks
from .exclusion_mask import build_exclusion_mask
from .piv import VelocityField, compute_piv_pair
from .plotting import save_plots
from .preprocessing import preprocess_frame
from .ptv import compute_track_pair
from .tracking import TRACK_COLUMNS, track_particles
from .tracking_video import save_tracking_video
from .video import VideoInfo, get_video_info, iter_frames


def _prepare_tracks(config: AppConfig, fps: float, output_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Link once, inspecting every source frame when consistency is enabled."""
    video = replace(config.video, step=1) if config.consistency.enabled else config.video
    source_frames = []

    def frames():
        for source_frame, image in iter_frames(video):
            source_frames.append(source_frame)
            yield source_frame, image

    print("Detecting and linking particle candidates...", flush=True)
    tracks = track_particles(
        frames(), fps, config.preprocessing, config.tracking,
        config.calibration.mm_per_pixel, exclusion=config.exclusion,
    )
    details = {
        **asdict(config.consistency),
        "candidate_particles": int(tracks["particle"].nunique()),
        "candidate_detections": len(tracks),
        "inspected_frames": len(source_frames),
        "detection_step": video.step,
    }
    if config.consistency.enabled:
        tracks, report = filter_consistent_tracks(tracks, source_frames, config.consistency)
        #report.to_csv(output_dir / "particle_consistency.csv", index=False)
        details.update({
            "accepted_particle_windows": int(report["accepted"].sum()),
            "accepted_particles": int(tracks["particle"].nunique()),
            "accepted_detections": len(tracks),
            "complete_windows": len(source_frames) // config.consistency.window_frames,
            "ignored_tail_frames": len(source_frames) % config.consistency.window_frames,
        })
        print(
            f"Consistency: kept {details['accepted_particles']} of "
            f"{details['candidate_particles']} IDs in "
            f"{details['accepted_particle_windows']} qualifying windows.", flush=True,
        )
    return tracks, details


def run_tracking(
    config: AppConfig, output_dir: str | Path, trail_length: int = 20,
    *, export_flow: bool = False,
) -> dict:
    """Export a PTV replay and optionally flow plots from the same detections."""
    config.validate()
    if trail_length < 1:
        raise ValueError("trail_length must be >= 1")
    info = get_video_info(config.video)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks, consistency = _prepare_tracks(config, info.fps, output_dir)
    save_tracking_video(
        iter_frames(config.video), tracks, output_dir / "tracking.avi",
        info.fps, config.tracking, config.calibration.mm_per_pixel,
        step=config.video.step, trail_length=trail_length,
    )
    #tracks.to_csv(output_dir / "trajectories.csv", index=False)
    calibrated = config.calibration.mm_per_pixel > 0
    summary = {
        "video": str(config.video.path),
        "fps": info.fps,
        "start_frame": config.video.start_frame,
        "stop_frame": config.video.stop_frame,
        "step": config.video.step,
        "playback_fps": info.fps / config.video.step,
        "tracked_particles": int(tracks["particle"].nunique()),
        "detections": len(tracks),
        "trajectory_position_unit": "mm" if calibrated else "px",
        "trajectory_velocity_unit": "mm/s" if calibrated else "px/s",
        "preprocessing": asdict(config.preprocessing),
        "tracking": asdict(config.tracking),
        "exclusion": asdict(config.exclusion),
        "mm_per_pixel": config.calibration.mm_per_pixel,
        "trail_length": trail_length,
        "consistency": consistency,
        "piv": asdict(config.piv),
    }
    if export_flow:
        print("Creating flow plots from the existing particle tracks...", flush=True)
        summary["flow_analysis"] = _export_flow(
            config, output_dir, info, tracks, consistency, from_tracks=True,
        )
    with (output_dir / "tracking_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def export_tracking_flow(tracking_dir: str | Path, output_dir: str | Path) -> dict:
    """Create flow exports from a saved tracking run without detecting again.

    Use the original run's metadata for units, frame selection, exclusion and
    FPS. The source video is read for geometry and per-frame exclusion masks.
    """
    tracking_dir = Path(tracking_dir)
    with (tracking_dir / "tracking_summary.json").open(encoding="utf-8") as handle:
        saved = json.load(handle)
    consistency = saved.get("consistency", {})
    config = AppConfig(
        video=VideoConfig(
            path=Path(saved["video"]), start_frame=saved["start_frame"],
            stop_frame=saved["stop_frame"], step=saved["step"],
            fps_override=saved["fps"],
        ),
        preprocessing=PreprocessingConfig(**saved["preprocessing"]),
        tracking=TrackingConfig(**saved["tracking"]),
        calibration=CalibrationConfig(mm_per_pixel=saved["mm_per_pixel"]),
        exclusion=ExclusionConfig(**saved["exclusion"]),
        piv=PIVConfig(**saved.get("piv", {})),
        consistency=ConsistencyConfig(**{
            key: value for key, value in consistency.items()
            if key in ConsistencyConfig.__dataclass_fields__
        }),
    )
    config.validate()
    tracks = pd.read_csv(tracking_dir / "trajectories.csv")
    missing = set(TRACK_COLUMNS) - set(tracks.columns)
    if missing:
        raise ValueError(f"Saved trajectories are missing columns: {sorted(missing)}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Creating flow plots from saved trajectories (no particle detection)...", flush=True)
    summary = _export_flow(
        config, output_dir, get_video_info(config.video), tracks, consistency,
        from_tracks=True,
    )
    summary["trajectory_source"] = str(tracking_dir / "trajectories.csv")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def run_pipeline(config: AppConfig, output_dir: str | Path) -> dict:
    """Export time-averaged PIV or consistency-filtered particle velocities.

    The time average is meaningful primarily for statistically steady flows.
    For unsteady fields, analyze shorter time windows or individual pairs.
    """
    config.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    info = get_video_info(config.video)
    tracks = None
    consistency = asdict(config.consistency)
    if config.consistency.enabled:
        tracks, consistency = _prepare_tracks(config, info.fps, output_dir)
        #tracks.to_csv(output_dir / "trajectories.csv", index=False)
    return _export_flow(
        config, output_dir, info, tracks, consistency,
        from_tracks=config.consistency.enabled,
    )


def _export_flow(
    config: AppConfig, output_dir: Path, info: VideoInfo,
    tracks: pd.DataFrame | None, consistency: dict, *, from_tracks: bool,
) -> dict:
    """Share grid measurements and exports across PIV, PTV and saved tracks."""
    if from_tracks and (tracks is None or tracks.empty):
        raise ValueError(
            "No particle tracks available for flow plots. Inspect particle_consistency.csv "
            "if enabled; adjust the tracking or consistency settings."
        )

    source = iter_frames(config.video)
    first = next(source, None)
    if first is None:
        raise ValueError("Video contains no frames in the selected range")
    prev_index, prev_raw = first
    prev_frame = preprocess_frame(prev_raw, config.preprocessing)
    #prev_mask = build_exclusion_mask(prev_raw, config.exclusion)
    #if config.exclusion.rectangles or config.exclusion.overlay_colors:
    #    # This preview represents the first selected frame; moving masks are
    #    # recomputed separately for every frame throughout the analysis.
    #    preview = prev_raw.copy()
    #    preview[prev_mask] = (
    #        0.4 * preview[prev_mask] + 0.6 * np.array([255, 0, 255])
    #    ).astype(np.uint8)
    #    cv2.imwrite(str(output_dir / "exclusion_preview.png"), preview)
    #    cv2.imwrite(str(output_dir / "exclusion_mask.png"), prev_mask.astype(np.uint8) * 255)

    sum_u = sum_v = counts = None
    x = y = None
    first_index = prev_index
    last_index = prev_index
    pair_count = 0

    for index, raw in source:
        current = preprocess_frame(raw, config.preprocessing)
        current_mask = build_exclusion_mask(raw, config.exclusion)
        dt = (index - prev_index) / info.fps
        if from_tracks:
            field = compute_track_pair(
                tracks, prev_index, index, current.shape, info.fps, config.piv,
                config.calibration.mm_per_pixel, 
                #exclusion_mask=prev_mask | current_mask,
            )
        else:
            field = compute_piv_pair(
                prev_frame, current, dt, config.piv, config.calibration.mm_per_pixel,
                #exclusion_mask=prev_mask | current_mask,
            )
        if sum_u is None:
            sum_u = np.zeros_like(field.u, dtype=float)
            sum_v = np.zeros_like(field.v, dtype=float)
            counts = np.zeros_like(field.u, dtype=np.int32)
            x, y = field.x, field.y

        valid = np.isfinite(field.u) & np.isfinite(field.v)
        sum_u += np.where(valid, field.u, 0.0)
        sum_v += np.where(valid, field.v, 0.0)
        counts += valid.astype(np.int32)
        pair_count += 1
        prev_index, prev_frame = index, current
        prev_mask = current_mask
        last_index = index

    if pair_count == 0:
        raise ValueError("Select at least two frames for velocity analysis")
    if not np.any(counts):
        if from_tracks:
            raise ValueError(
                "No accepted particle IDs were observed at both endpoints of an "
                "analysis pair in an unmasked grid window. Reduce video.step or "
                "adjust the consistency, tracking and exclusion settings."
            )
        raise ValueError(
            "No valid velocity vectors. Tune particle contrast, window sizes, "
            "or the PIV signal_to_noise_threshold, and check exclusion settings."
        )

    # A given cell only contributes its VALID temporal observations to the mean.
    u = np.divide(sum_u, counts, out=np.full_like(sum_u, np.nan), where=counts > 0)
    v = np.divide(sum_v, counts, out=np.full_like(sum_v, np.nan), where=counts > 0)
    scale_known = config.calibration.mm_per_pixel > 0
    mean_field = VelocityField(
        x=x, y=y, u=u, v=v, valid=counts > 0,
        position_unit="mm" if scale_known else "px",
        velocity_unit="mm/s" if scale_known else "px/s",
    )
    metrics = calculate_flow_properties(mean_field)
    save_plots(mean_field, metrics, output_dir)

    #np.savez_compressed(
    #    output_dir / "velocity_field.npz",
    #    x=x, y=y, u=u, v=v, valid=mean_field.valid,
    #    valid_observation_count=counts,
    #    speed=metrics.speed,
    #    vorticity=metrics.vorticity,
    #    divergence=metrics.divergence,
    #    strain_rate_magnitude=metrics.strain_rate_magnitude,
    #)
    #pd.DataFrame({
    #    "x": x.ravel(), "y": y.ravel(),
    #    "u": u.ravel(), "v": v.ravel(),
    #    "speed": metrics.speed.ravel(),
    #    "vorticity_1_per_s": metrics.vorticity.ravel(),
    #    "divergence_1_per_s": metrics.divergence.ravel(),
    #    "strain_rate_1_per_s": metrics.strain_rate_magnitude.ravel(),
    #    "valid_observations": counts.ravel(),
    #}).to_csv(output_dir / "velocity_field.csv", index=False)

    summary = summarize_field(mean_field, metrics)
    summary.update({
        "video": str(config.video.path),
        "fps": info.fps,
        "first_source_frame": first_index,
        "last_source_frame": last_index,
        "processed_frame_pairs": pair_count,
        "time_averaged": True,
        "exclusion": asdict(config.exclusion),
        "analysis_method": (
            "consistent_ptv" if config.consistency.enabled else "ptv"
        ) if from_tracks else "piv",
        "consistency": consistency,
        "tracking": asdict(config.tracking),
        "preprocessing": asdict(config.preprocessing),
        "mm_per_pixel": config.calibration.mm_per_pixel,
    })

    if tracks is not None or config.tracking.enabled:
        if tracks is None:
            tracks, _ = _prepare_tracks(config, info.fps, output_dir)
        #tracks.to_csv(output_dir / "trajectories.csv", index=False)
        summary["tracked_particles"] = int(tracks["particle"].nunique())
        summary["trajectory_position_unit"] = mean_field.position_unit
        summary["trajectory_velocity_unit"] = mean_field.velocity_unit

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
