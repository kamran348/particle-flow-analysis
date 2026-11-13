"""Grid velocities measured exclusively from selected particle trajectories."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PIVConfig
from .exclusion_mask import excluded_piv_windows
from .piv import VelocityField


def compute_track_pair(
    tracks: pd.DataFrame,
    first_frame: int,
    last_frame: int,
    image_shape: tuple[int, int],
    fps: float,
    config: PIVConfig,
    mm_per_pixel: float = 0.0,
    exclusion_mask: np.ndarray | None = None,
) -> VelocityField:
    """Average matched-particle velocities within overlapping grid windows.

    Only IDs observed at BOTH selected endpoints contribute. Coordinates in
    tracks already carry the calibration and +y-up conversion. Window selection
    uses displacement midpoints. No image correlation or spatial interpolation
    is performed; unsupported grid cells remain NaN.
    """
    height, width = image_shape
    size = config.search_area_size
    if min(image_shape) < size:
        raise ValueError("Analysis grid window exceeds at least one frame dimension")
    if last_frame <= first_frame or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Track pair needs increasing source frames and positive finite FPS")
    if exclusion_mask is not None and exclusion_mask.shape != image_shape:
        raise ValueError("Exclusion mask must match the analysis image shape")
    scale = mm_per_pixel if mm_per_pixel > 0 else 1.0
    stride = size - config.overlap
    x0, y0 = np.meshgrid(np.arange(0, width - size + 1, stride),
                         np.arange(0, height - size + 1, stride))
    sum_u = np.zeros_like(x0, dtype=float)
    sum_v = np.zeros_like(y0, dtype=float)
    counts = np.zeros_like(x0, dtype=int)
    endpoints = tracks.loc[tracks["source_frame"].isin([first_frame, last_frame])]
    first = endpoints.loc[endpoints["source_frame"] == first_frame]
    last = endpoints.loc[endpoints["source_frame"] == last_frame]
    pairs = first.merge(last, on="particle", suffixes=("_a", "_b"), validate="one_to_one")
    if "consistency_segment_a" in pairs:
        pairs = pairs.loc[pairs["consistency_segment_a"] == pairs["consistency_segment_b"]]
    dt = (last_frame - first_frame) / fps
    for row in pairs.itertuples(index=False):
        px = (row.x_a + row.x_b) / (2 * scale)
        py = -(row.y_a + row.y_b) / (2 * scale)
        vx = (row.x_b - row.x_a) / dt
        vy = (row.y_b - row.y_a) / dt
        if not np.isfinite([px, py, vx, vy]).all():
            continue
        inside = (px >= x0) & (px < x0 + size) & (py >= y0) & (py < y0 + size)
        sum_u[inside] += vx
        sum_v[inside] += vy
        counts[inside] += 1
    valid = counts > 0
    if exclusion_mask is not None and exclusion_mask.any():
        valid &= ~excluded_piv_windows(exclusion_mask, x0.shape, size, config.overlap)
    u = np.divide(sum_u, counts, out=np.full_like(sum_u, np.nan), where=valid)
    v = np.divide(sum_v, counts, out=np.full_like(sum_v, np.nan), where=valid)
    return VelocityField(
        x=(x0 + size / 2) * scale, y=(height - y0 - size / 2) * scale,
        u=u, v=v, valid=valid,
        position_unit="mm" if mm_per_pixel > 0 else "px",
        velocity_unit="mm/s" if mm_per_pixel > 0 else "px/s",
    )
