"""Generate a small reproducible artificial particle movie for first runs."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def create_demo_video(
    output_path: str | Path,
    width: int = 320,
    height: int = 240,
    n_frames: int = 24,
    fps: float = 30.0,
    dx_per_frame: float = 1.25,
    dy_per_frame: float = -0.65,
    n_particles: int = 800,
    seed: int = 7,
) -> Path:
    """Brightness images of particles undergoing uniform 2D translation."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"MJPG"),
        fps, (width, height), isColor=True,
    )
    if not writer.isOpened():
        raise OSError("Could not create AVI video; check OpenCV video codecs")

    rng = np.random.default_rng(seed)
    base_x = rng.uniform(0, width, n_particles)
    base_y = rng.uniform(0, height, n_particles)
    try:
        for t in range(n_frames):
            img = np.zeros((height, width), dtype=np.uint8)
            xs = np.mod(base_x + t * dx_per_frame, width)
            ys = np.mod(base_y + t * dy_per_frame, height)
            for x, y in zip(xs, ys):
                cv2.circle(img, (int(round(x)) % width, int(round(y)) % height),
                           radius=1, color=240, thickness=-1)
            img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.65)
            frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            writer.write(frame)
    finally:
        writer.release()
    return output_path
