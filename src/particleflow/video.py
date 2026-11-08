"""Video metadata and streaming frame extraction via OpenCV."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator

import cv2
import numpy as np

from .config import VideoConfig


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int


def get_video_info(config: VideoConfig) -> VideoInfo:
    path = Path(config.path)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {path}")
    try:
        fps = config.fps_override or cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Could not determine FPS: set video.fps_override")
        return VideoInfo(
            fps=float(fps),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def iter_frames(config: VideoConfig) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (source frame number, BGR image) without loading whole video."""
    cap = cv2.VideoCapture(str(config.path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {config.path}")
    try:
        if config.start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, config.start_frame)
        index = config.start_frame
        while config.stop_frame == 0 or index < config.stop_frame:
            success, frame = cap.read()
            if not success:
                break
            if (index - config.start_frame) % config.step == 0:
                yield index, frame
            index += 1
    finally:
        cap.release()
