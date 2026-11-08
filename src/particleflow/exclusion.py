"""Masks for static regions and moving colored overlays in grayscale footage."""

from __future__ import annotations

import cv2
import numpy as np

from .config import ExclusionConfig


def build_exclusion_mask(frame: np.ndarray, config: ExclusionConfig) -> np.ndarray:
    """Return True at excluded pixels, using original BGR colors before preprocessing.

    Color detection is intended for colored annotations on grayscale microscopy
    footage. Padding also covers nearby gray labels that cannot be color-detected.
    Rectangles use original image pixels with the origin at the top left.
    """
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    for x0, y0, x1, y1 in config.rectangles:
        if x1 > width or y1 > height:
            raise ValueError("Exclusion rectangle extends outside the video frame")
        mask[y0:y1, x0:x1] = True

    if config.overlay_colors and frame.ndim == 3:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        colored = np.zeros_like(mask)
        if "red" in config.overlay_colors:
            colored |= (hue <= 10) | (hue >= 170)
        if "green" in config.overlay_colors:
            colored |= (hue >= 35) & (hue <= 85)
        colored &= (saturation >= 100) & (value >= 100)
        if config.padding and colored.any():
            size = 2 * config.padding + 1
            colored = cv2.dilate(
                colored.astype(np.uint8), np.ones((size, size), dtype=np.uint8)
            ).astype(bool)
        mask |= colored
    return mask


def excluded_piv_windows(
    mask: np.ndarray, field_shape: tuple[int, int], search_area_size: int, overlap: int
) -> np.ndarray:
    """Reject any search window containing an excluded pixel, including its edges."""
    integral = cv2.integral(mask.astype(np.uint8))
    stride = search_area_size - overlap
    y0 = np.arange(field_shape[0])[:, None] * stride
    x0 = np.arange(field_shape[1])[None, :] * stride
    y1, x1 = y0 + search_area_size, x0 + search_area_size
    covered = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    return covered > 0
