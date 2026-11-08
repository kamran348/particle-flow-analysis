"""Orchestrate the full PIV -> measurements -> exports workflow."""

from __future__ import annotations

from pathlib import Path
from dataclasses import asdict
import json

import cv2
import numpy as np
import pandas as pd

from .analysis import calculate_flow_properties, summarize_field
from .config import AppConfig
from .exclusion import build_exclusion_mask
from .piv import VelocityField, compute_piv_pair
from .plotting import save_plots
from .preprocessing import preprocess_frame
from .tracking import track_particles
from .video import get_video_info, iter_frames


def run_pipeline(config: AppConfig, output_dir: str | Path) -> dict:
    """Process selected frame pairs and export time-averaged 2D PIV results.

    The time average is meaningful primarily for statistically steady flows.
    For unsteady fields, analyze shorter time windows or individual pairs.
    """
    config.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    info = get_video_info(config.video)

    source = iter_frames(config.video)
    first = next(source, None)
    if first is None:
        raise ValueError("Video contains no frames in the selected range")
    prev_index, prev_raw = first
    prev_frame = preprocess_frame(prev_raw, config.preprocessing)
    prev_mask = build_exclusion_mask(prev_raw, config.exclusion)
    if config.exclusion.rectangles or config.exclusion.overlay_colors:
        # This preview represents the first selected frame; moving masks are
        # recomputed separately for every frame throughout the analysis.
        preview = prev_raw.copy()
        preview[prev_mask] = (
            0.4 * preview[prev_mask] + 0.6 * np.array([255, 0, 255])
        ).astype(np.uint8)
        cv2.imwrite(str(output_dir / "exclusion_preview.png"), preview)
        cv2.imwrite(str(output_dir / "exclusion_mask.png"), prev_mask.astype(np.uint8) * 255)

    sum_u = sum_v = counts = None
    x = y = None
    first_index = prev_index
    last_index = prev_index
    pair_count = 0

    for index, raw in source:
        current = preprocess_frame(raw, config.preprocessing)
        current_mask = build_exclusion_mask(raw, config.exclusion)
        dt = (index - prev_index) / info.fps
        field = compute_piv_pair(
            prev_frame, current, dt, config.piv, config.calibration.mm_per_pixel,
            exclusion_mask=prev_mask | current_mask,
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
        raise ValueError("Select at least two frames for PIV analysis")
    if not np.any(counts):
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

    np.savez_compressed(
        output_dir / "velocity_field.npz",
        x=x, y=y, u=u, v=v, valid=mean_field.valid,
        valid_observation_count=counts,
        speed=metrics.speed,
        vorticity=metrics.vorticity,
        divergence=metrics.divergence,
        strain_rate_magnitude=metrics.strain_rate_magnitude,
    )
    pd.DataFrame({
        "x": x.ravel(), "y": y.ravel(),
        "u": u.ravel(), "v": v.ravel(),
        "speed": metrics.speed.ravel(),
        "vorticity_1_per_s": metrics.vorticity.ravel(),
        "divergence_1_per_s": metrics.divergence.ravel(),
        "strain_rate_1_per_s": metrics.strain_rate_magnitude.ravel(),
        "valid_observations": counts.ravel(),
    }).to_csv(output_dir / "velocity_field.csv", index=False)

    summary = summarize_field(mean_field, metrics)
    summary.update({
        "video": str(config.video.path),
        "fps": info.fps,
        "first_source_frame": first_index,
        "last_source_frame": last_index,
        "processed_frame_pairs": pair_count,
        "time_averaged": True,
        "exclusion": asdict(config.exclusion),
    })

    if config.tracking.enabled:
        tracks = track_particles(
            iter_frames(config.video), info.fps,
            config.preprocessing, config.tracking,
            config.calibration.mm_per_pixel,
            exclusion=config.exclusion,
        )
        tracks.to_csv(output_dir / "trajectories.csv", index=False)
        summary["tracked_particles"] = int(tracks["particle"].nunique())
        summary["trajectory_position_unit"] = mean_field.position_unit
        summary["trajectory_velocity_unit"] = mean_field.velocity_unit

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
