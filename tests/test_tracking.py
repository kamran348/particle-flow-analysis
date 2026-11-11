import cv2
import numpy as np
import pandas as pd
import pytest

from particleflow.config import (
    AppConfig, PreprocessingConfig, TrackingConfig, VideoConfig,
)
from particleflow.pipeline import run_tracking
from particleflow.synthetic import create_demo_video
from particleflow.tracking import TRACK_COLUMNS, track_particles
from particleflow.tracking_video import iter_tracking_overlays, save_tracking_video


def test_tracks_preserve_identity_calibration_and_source_timing():
    pytest.importorskip("trackpy")
    frames = []
    # The middle frame has no detections; memory=1 should preserve the ID.
    for i, source_frame in enumerate((4, 6, 8)):
        image = np.zeros((96, 96, 3), dtype=np.uint8)
        if i != 1:
            cv2.circle(image, (30 + 2 * i, 50 - i), 2, (240, 240, 240), -1)
        frames.append((source_frame, image))
    tracks = track_particles(
        frames, 20, PreprocessingConfig(),
        TrackingConfig(diameter=9, min_mass=10, search_range=7, memory=1),
        mm_per_pixel=0.1,
    )
    assert tracks["source_frame"].tolist() == [4, 8]
    assert tracks["particle"].nunique() == 1
    np.testing.assert_allclose(tracks["time_s"], [0.2, 0.4])
    np.testing.assert_allclose(tracks["x"], [3, 3.4], atol=0.02)
    np.testing.assert_allclose(tracks["y"], [-5, -4.8], atol=0.02)
    assert tracks.iloc[1].vx == pytest.approx(2, abs=0.1)
    assert tracks.iloc[1].vy == pytest.approx(1, abs=0.1)


def _tracks(positions):
    return pd.DataFrame([
        {"particle": 0, "source_frame": frame, "x": x, "y": y}
        for frame, x, y in positions
    ])


def test_overlay_undoes_calibration_and_leaves_source_untouched():
    raw = np.zeros((96, 128, 3), dtype=np.uint8)
    tracks = _tracks([(7, 8, -12)])  # 0.2 mm/px -> image pixel (40, 60).
    [(source_frame, overlay)] = iter_tracking_overlays(
        [(7, raw)], tracks, 30, TrackingConfig(), mm_per_pixel=0.2,
    )
    assert source_frame == 7
    assert not raw.any()
    assert overlay[57:64, 37:44].any()
    assert not overlay[57:64, 77:84].any()


def test_trails_expire_on_empty_frames_and_do_not_bridge_gaps():
    raw = np.zeros((96, 128, 3), dtype=np.uint8)
    tracks = _tracks([(0, 20, -60), (1, 40, -60), (3, 80, -60)])
    overlays = list(iter_tracking_overlays(
        [(i, raw) for i in range(6)], tracks, 30, TrackingConfig(), trail_length=3,
    ))
    assert overlays[1][1][60, 30].any()  # connected consecutive observations
    assert not overlays[3][1][60, 60].any()  # missing frame breaks the trail
    assert not overlays[3][1][60, 30].any()  # old segment has expired
    assert not overlays[4][1][57:64, 77:84].any()  # no current marker when missing
    assert not overlays[5][1][40:].any()  # entire history has expired


def test_replay_includes_empty_frames_and_uses_selected_frame_rate(tmp_path):
    raw = np.zeros((96, 128, 3), dtype=np.uint8)
    path = save_tracking_video(
        [(i, raw) for i in (10, 12, 14)], pd.DataFrame(columns=TRACK_COLUMNS),
        tmp_path / "tracking.avi", 30, TrackingConfig(), step=2,
    )
    capture = cv2.VideoCapture(str(path))
    try:
        assert capture.isOpened()
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(15)
        assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 3
        success, image = capture.read()
        assert success and image.shape == raw.shape
    finally:
        capture.release()


def test_replay_rejects_empty_source(tmp_path):
    with pytest.raises(ValueError, match="no frames"):
        save_tracking_video(
            [], pd.DataFrame(columns=TRACK_COLUMNS), tmp_path / "empty.avi",
            30, TrackingConfig(),
        )


def test_overlay_does_not_connect_across_rejected_consistency_windows():
    raw = np.zeros((96, 128, 3), dtype=np.uint8)
    tracks = _tracks([(0, 20, -60), (200, 80, -60)])
    tracks["consistency_segment"] = [1, 2]
    overlays = list(iter_tracking_overlays(
        [(0, raw), (200, raw)], tracks, 30, TrackingConfig(),
    ))
    assert not overlays[1][1][60, 50].any()


def test_tracking_workflow_exports_video_and_calibrated_csv(tmp_path):
    pytest.importorskip("trackpy")
    video = create_demo_video(
        tmp_path / "particles.avi", width=128, height=96, n_frames=6, n_particles=15,
    )
    config = AppConfig(
        video=VideoConfig(path=video, step=2),
        tracking=TrackingConfig(enabled=True, min_mass=10),
    )
    output = tmp_path / "results"
    summary = run_tracking(config, output, export_flow=True)
    assert summary["tracked_particles"] > 0
    assert summary["playback_fps"] == 15
    tracks = pd.read_csv(output / "trajectories.csv")
    assert set(tracks["source_frame"]) == {0, 2, 4}
    assert len(tracks) == summary["detections"]
    assert (output / "tracking.avi").is_file()
    assert (output / "tracking_summary.json").is_file()
    assert summary["flow_analysis"]["analysis_method"] == "ptv"
    assert (output / "velocity_vectors.png").is_file()
    assert (output / "streamlines.png").is_file()
