import cv2
import numpy as np
import pandas as pd
import pytest

from particleflow.config import (
    AppConfig, CalibrationConfig, ConsistencyConfig, PIVConfig, TrackingConfig,
    VideoConfig, load_config,
)
from particleflow.consistency import filter_consistent_tracks
from particleflow.pipeline import export_tracking_flow, run_pipeline, run_tracking
from particleflow.ptv import compute_track_pair
from particleflow.synthetic import create_demo_video


def make_tracks(observations):
    return pd.DataFrame([
        {"particle": particle, "source_frame": frame, "time_s": frame / 10,
         "x": frame + 30.0, "y": -40.0, "vx": 10.0, "vy": 0.0, "speed": 10.0}
        for particle, frames in observations.items() for frame in frames
    ])


def test_filter_requires_coverage_and_endpoints_without_requiring_stationary_particles():
    tracks = make_tracks({0: range(10, 20), 1: [10, 11, 19], 2: range(11, 20),
                          3: [10, 11, 12, 13, 14, 16, 17, 18, 19]})
    filtered, report = filter_consistent_tracks(
        tracks, range(10, 20), ConsistencyConfig(window_frames=10, min_fraction=0.9),
    )
    assert set(filtered.particle) == {0, 3}
    assert filtered.loc[filtered.particle == 0, "x"].nunique() == 10
    assert report.set_index("particle").loc[2, "fraction"] == 0.9
    assert not report.set_index("particle").loc[2, "seen_first"]
    assert len(filtered.loc[filtered.particle == 3]) == 9  # no interpolation
    np.testing.assert_allclose(filtered.loc[filtered.particle == 3, "vx"].dropna(), 10)
    # Strict mode really requires a detection in every frame.
    strict, _ = filter_consistent_tracks(
        tracks, range(10, 20), ConsistencyConfig(window_frames=10, min_fraction=1.0),
    )
    assert set(strict.particle) == {0}


def test_windows_reject_incomplete_tail_and_do_not_bridge_rejected_intervals():
    tracks = make_tracks({0: [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16]})
    filtered, report = filter_consistent_tracks(
        tracks, range(17), ConsistencyConfig(window_frames=5, min_fraction=1),
    )
    assert filtered.source_frame.tolist() == [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]
    assert np.isnan(filtered.loc[filtered.source_frame == 10, "vx"].iloc[0])
    assert filtered.consistency_segment.nunique() == 2
    assert not report.loc[report.window == 3, "accepted"].any()
    assert not report.loc[report.window == 3, "complete_window"].any()
    field = compute_track_pair(filtered, 4, 10, (96, 128), 10, PIVConfig())
    assert not field.valid.any()


def test_filter_counts_frames_without_any_detections_and_rejects_short_or_skipped_ranges():
    tracks = make_tracks({0: [0, 9]})
    filtered, report = filter_consistent_tracks(
        tracks, range(10), ConsistencyConfig(window_frames=10),
    )
    assert filtered.empty
    assert report.fraction.iloc[0] == 0.2
    with pytest.raises(ValueError, match="at least 100"):
        filter_consistent_tracks(tracks, range(10), ConsistencyConfig())
    with pytest.raises(ValueError, match="every source frame"):
        filter_consistent_tracks(tracks, range(0, 20, 2), ConsistencyConfig(window_frames=10))
    empty, report = filter_consistent_tracks(tracks.iloc[:0], range(10), ConsistencyConfig(window_frames=10))
    assert empty.empty and report.empty


def test_grid_uses_only_matched_selected_tracks_and_correct_physical_units():
    tracks = make_tracks({0: [0, 2], 1: [0]})
    tracks.loc[tracks.particle == 0, "x"] = [4.0, 4.4]
    tracks.loc[tracks.particle == 0, "y"] = [-4.0, -3.8]
    # A huge velocity on a track without a second endpoint cannot affect the field.
    tracks.loc[tracks.particle == 1, "vx"] = 9999
    field = compute_track_pair(tracks, 0, 2, (96, 128), 10, PIVConfig(), mm_per_pixel=0.1)
    assert field.valid.any()
    np.testing.assert_allclose(field.u[field.valid], 2)
    np.testing.assert_allclose(field.v[field.valid], 1)
    assert field.position_unit == "mm" and field.velocity_unit == "mm/s"
    assert field.x[0, 0] == pytest.approx(2.4)
    assert field.y[0, 0] == pytest.approx(7.2)
    mask = np.ones((96, 128), dtype=bool)
    excluded = compute_track_pair(tracks, 0, 2, (96, 128), 10, PIVConfig(), 0.1, mask)
    assert not excluded.valid.any()
    assert np.isnan(excluded.u).all()


def test_replay_and_analysis_use_identical_particle_selection(tmp_path, monkeypatch):
    pytest.importorskip("trackpy")
    video = tmp_path / "particles.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (128, 96))
    assert writer.isOpened()
    try:
        for frame in range(10):
            image = np.zeros((96, 128, 3), dtype=np.uint8)
            cv2.circle(image, (30 + frame, 40), 2, (240, 240, 240), -1)
            if frame < 2:
                cv2.circle(image, (90, 70), 2, (240, 240, 240), -1)
            writer.write(image)
    finally:
        writer.release()
    config = AppConfig(
        video=VideoConfig(path=video, step=3),
        piv=PIVConfig(window_size=16, search_area_size=32, overlap=16),
        tracking=TrackingConfig(diameter=9, min_mass=10, search_range=5),
        calibration=CalibrationConfig(mm_per_pixel=0.1),
        consistency=ConsistencyConfig(enabled=True, window_frames=5, min_fraction=1),
    )

    def reject_image_correlation(*args, **kwargs):
        pytest.fail("Consistent analysis must not correlate rejected particle pixels")

    monkeypatch.setattr("particleflow.pipeline.compute_piv_pair", reject_image_correlation)
    from particleflow.tracking import track_particles as original_tracking
    detection_calls = []

    def counted_tracking(*args, **kwargs):
        detection_calls.append(1)
        return original_tracking(*args, **kwargs)

    monkeypatch.setattr("particleflow.pipeline.track_particles", counted_tracking)
    replay = run_tracking(config, tmp_path / "replay", export_flow=True)
    assert len(detection_calls) == 1  # replay and flow share one detection pass
    assert replay["flow_analysis"]["analysis_method"] == "consistent_ptv"
    analysis = run_pipeline(config, tmp_path / "analysis")
    assert len(detection_calls) == 2
    assert replay["consistency"]["candidate_particles"] == 2
    assert replay["tracked_particles"] == analysis["tracked_particles"] == 1
    assert replay["consistency"]["detection_step"] == 1  # despite video.step=3
    a = pd.read_csv(tmp_path / "analysis" / "trajectories.csv")
    b = pd.read_csv(tmp_path / "replay" / "trajectories.csv")
    pd.testing.assert_frame_equal(a, b)
    assert set(a.source_frame) == set(range(10))
    assert analysis["analysis_method"] == "consistent_ptv"
    assert analysis["mean_u"] == pytest.approx(1, abs=0.1)
    assert analysis["mean_v"] == pytest.approx(0, abs=0.1)
    assert (tmp_path / "analysis" / "particle_consistency.csv").is_file()
    # Saved tracks preserve calibration, FPS, frame step and custom grid settings.
    reused = export_tracking_flow(tmp_path / "replay", tmp_path / "reused")
    assert len(detection_calls) == 2  # no detection when reusing saved trajectories
    assert reused["mean_u"] == pytest.approx(1, abs=0.1)
    assert reused["position_unit"] == "mm"
    assert reused["analysis_method"] == "consistent_ptv"
    for folder in ("replay", "analysis", "reused"):
        assert (tmp_path / folder / "velocity_vectors.png").is_file()
        assert (tmp_path / folder / "streamlines.png").is_file()
    original_field = pd.read_csv(tmp_path / "replay" / "velocity_field.csv")
    pd.testing.assert_frame_equal(original_field, pd.read_csv(tmp_path / "reused" / "velocity_field.csv"))
    pd.testing.assert_frame_equal(original_field, pd.read_csv(tmp_path / "analysis" / "velocity_field.csv"))


@pytest.mark.parametrize("settings", [
    ConsistencyConfig(window_frames=1), ConsistencyConfig(window_frames=2.5),
    ConsistencyConfig(min_fraction=0), ConsistencyConfig(min_fraction=1.1),
    ConsistencyConfig(min_fraction=float("nan")), ConsistencyConfig(enabled=1),
])
def test_rejects_invalid_consistency_configuration(settings):
    with pytest.raises(ValueError, match="consistency"):
        AppConfig(consistency=settings).validate()


def test_loads_consistency_settings(tmp_path):
    path = tmp_path / "consistent.toml"
    path.write_text('[consistency]\nenabled = true\nwindow_frames = 40\n'
                    'min_fraction = 0.8\nrequire_endpoints = false\n')
    config = load_config(path)
    assert config.consistency == ConsistencyConfig(True, 40, 0.8, False)


def test_standard_piv_remains_available_without_tracking(tmp_path, monkeypatch):
    video = create_demo_video(tmp_path / "piv.avi", width=128, height=96,
                              n_frames=3, n_particles=150)
    config = AppConfig(video=VideoConfig(path=video),
                       piv=PIVConfig(signal_to_noise_threshold=1.0))

    def reject_tracking(*args, **kwargs):
        pytest.fail("Standard PIV without tracking must not load particle tracking")

    monkeypatch.setattr("particleflow.pipeline.track_particles", reject_tracking)
    summary = run_pipeline(config, tmp_path / "piv-results")
    assert summary["analysis_method"] == "piv"
    assert summary["processed_frame_pairs"] == 2
    assert not (tmp_path / "piv-results" / "trajectories.csv").exists()
