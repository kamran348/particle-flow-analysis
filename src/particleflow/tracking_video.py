"""Replay linked particle positions over the original video frames."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import TrackingConfig


def iter_tracking_overlays(
    frames: Iterable[tuple[int, np.ndarray]],
    tracks: pd.DataFrame,
    fps: float,
    tracking: TrackingConfig,
    mm_per_pixel: float = 0.0,
    trail_length: int = 20,
) -> Iterator[tuple[int, np.ndarray]]:
    """Draw detections, stable IDs and trails on copies of selected BGR frames.

    Positions in the tracking CSV use +y up and may be calibrated. Convert
    back to original pixels before drawing. Trails span at most ``trail_length``
    selected frames; missing detections break the line (no predicted positions).
    """
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    if not np.isfinite(mm_per_pixel) or mm_per_pixel < 0:
        raise ValueError("mm_per_pixel must be nonnegative and finite")
    if trail_length < 1:
        raise ValueError("trail_length must be >= 1")
    scale = mm_per_pixel if mm_per_pixel > 0 else 1.0
    by_frame = {int(index): rows for index, rows in tracks.groupby("source_frame")}
    histories: dict[int, deque] = {}
    segments: dict[int, int | None] = {}
    radius = max(3, tracking.diameter // 2)

    for selected_index, (source_frame, raw) in enumerate(frames):
        image = raw.copy()
        # Expire histories even when the current frame has no detections.
        for particle in list(histories):
            history = histories[particle]
            while history and history[0][0] < selected_index - trail_length + 1:
                history.popleft()
            if not history:
                del histories[particle]
                segments.pop(particle, None)

        detections = by_frame.get(source_frame)
        if detections is not None:
            for row in detections.itertuples(index=False):
                particle = int(row.particle)
                segment = getattr(row, "consistency_segment", None)
                if particle in segments and segments[particle] != segment:
                    histories.pop(particle, None)
                segments[particle] = segment
                point = (int(round(row.x / scale)), int(round(-row.y / scale)))
                histories.setdefault(particle, deque()).append((selected_index, point))

        for particle, history in histories.items():
            # Golden-angle hues keep nearby IDs visually distinct.
            hsv = np.array([[[int(particle * 68.754) % 180, 210, 255]]], dtype=np.uint8)
            color = tuple(int(value) for value in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
            for previous, current in zip(history, list(history)[1:]):
                if current[0] == previous[0] + 1:
                    cv2.line(image, previous[1], current[1], color, 1, cv2.LINE_AA)
            if history[-1][0] == selected_index:
                point = history[-1][1]
                cv2.circle(image, point, radius, color, 1, cv2.LINE_AA)
                cv2.putText(image, str(particle), (point[0] + radius + 2, point[1] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        count = 0 if detections is None else len(detections)
        label = f"Frame {source_frame} | {source_frame / fps:.3f} s | {count} detections"
        # Outline the text to keep it legible on both bright and dark footage.
        for color, thickness in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.putText(image, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, thickness, cv2.LINE_AA)
        yield source_frame, image


def save_tracking_video(
    frames: Iterable[tuple[int, np.ndarray]],
    tracks: pd.DataFrame,
    output_path: str | Path,
    fps: float,
    tracking: TrackingConfig,
    mm_per_pixel: float = 0.0,
    step: int = 1,
    trail_length: int = 20,
) -> Path:
    """Save an MJPG AVI replay; playback accounts for the selected frame step."""
    if step < 1:
        raise ValueError("step must be >= 1")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlays = iter_tracking_overlays(
        frames, tracks, fps, tracking, mm_per_pixel, trail_length,
    )
    writer = None
    try:
        for _, image in overlays:
            if writer is None:
                height, width = image.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path), cv2.VideoWriter_fourcc(*"MJPG"),
                    fps / step, (width, height),
                )
                if not writer.isOpened():
                    raise OSError("Could not create tracking AVI; check OpenCV video codecs")
            writer.write(image)
        if writer is None:
            raise ValueError("Video contains no frames in the selected range")
    finally:
        if writer is not None:
            writer.release()
        overlays.close()
        # iter_frames owns a VideoCapture; release it on codec/render failures too.
        if hasattr(frames, "close"):
            frames.close()
    return output_path
