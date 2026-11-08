"""Deterministic image enhancement, shared by PIV and PTV."""

from __future__ import annotations

import cv2
import numpy as np
from .config import PreprocessingConfig


def preprocess_frame(frame: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """Convert a BGR or grayscale frame to processed uint8 grayscale."""
    if frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 2:
        gray = frame.copy()
    else:
        raise ValueError("Expected BGR image (H,W,3) or grayscale (H,W)")
    if gray.dtype != np.uint8:
        raise ValueError("Expected uint8 video frame")
    if config.gaussian_sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), config.gaussian_sigma)
    if config.clahe_clip_limit > 0:
        clahe = cv2.createCLAHE(clipLimit=config.clahe_clip_limit, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    #cv2.imshow("Grayscale frame", gray)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return gray
