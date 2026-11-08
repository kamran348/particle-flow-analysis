"""Single-pass 2D PIV using OpenPIV cross-correlation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PIVConfig
from .exclusion import excluded_piv_windows


@dataclass
class VelocityField:
    """Cartesian coordinates: +x right, +y up; raw frames have +y down."""
    x: np.ndarray            # 2D grid, px or mm
    y: np.ndarray            # 2D grid, px or mm
    u: np.ndarray            # horizontal speed, px/s or mm/s
    v: np.ndarray            # vertical speed (+up), px/s or mm/s
    valid: np.ndarray        # True where at least one valid measurement exists
    position_unit: str
    velocity_unit: str


def compute_piv_pair(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    dt_seconds: float,
    config: PIVConfig,
    mm_per_pixel: float = 0.0,
    exclusion_mask: np.ndarray | None = None,
) -> VelocityField:
    """Estimate the velocity field between two processed grayscale frames.

    The first and second images must be consecutive *selected* frames,
    with dt_seconds computed from their original indices and video FPS.
    """
    from openpiv import pyprocess, validation

    if frame_a.shape != frame_b.shape or frame_a.ndim != 2:
        raise ValueError("PIV expects two grayscale images of identical shape")
    if min(frame_a.shape) < config.search_area_size:
        raise ValueError("PIV search area exceeds at least one frame dimension")
    if dt_seconds <= 0:
        raise ValueError("dt_seconds must be positive")
    if exclusion_mask is not None and exclusion_mask.shape != frame_a.shape:
        raise ValueError("Exclusion mask must match the PIV image shape")

    u_img, v_img, s2n = pyprocess.extended_search_area_piv(
        frame_a.astype(np.int32),
        frame_b.astype(np.int32),
        window_size=config.window_size,
        overlap=config.overlap,
        search_area_size=config.search_area_size,
        dt=dt_seconds,
        sig2noise_method="peak2peak",
    )
    x_img, y_img = pyprocess.get_coordinates(
        image_size=frame_a.shape,
        search_area_size=config.search_area_size,
        overlap=config.overlap,
        center_on_field=False,
    )

    bad = validation.sig2noise_val(
        s2n, threshold=config.signal_to_noise_threshold
    )
    bad = np.asarray(bad, dtype=bool) | ~np.isfinite(u_img) | ~np.isfinite(v_img)
    if exclusion_mask is not None and exclusion_mask.any():
        bad |= excluded_piv_windows(
            exclusion_mask, u_img.shape, config.search_area_size, config.overlap
        )

    # Apply an actual measurement mask. Missing data stay NaN; we do not
    # silently fabricate vectors via full-field interpolation.
    u_img = np.where(bad, np.nan, u_img)
    v_img = np.where(bad, np.nan, v_img)

    scale = mm_per_pixel if mm_per_pixel > 0 else 1.0
    position_unit = "mm" if mm_per_pixel > 0 else "px"
    velocity_unit = "mm/s" if mm_per_pixel > 0 else "px/s"

    # OpenPIV returns image coordinates with downward-positive y, v.
    # Convert to Cartesian coordinates (+y up) while preserving row order.
    return VelocityField(
        x=np.asarray(x_img) * scale,
        y=(frame_a.shape[0] - np.asarray(y_img)) * scale,
        u=np.asarray(u_img, dtype=float) * scale,
        v=-np.asarray(v_img, dtype=float) * scale,
        valid=~bad,
        position_unit=position_unit,
        velocity_unit=velocity_unit,
    )
