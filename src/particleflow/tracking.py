"""Optional individual particle trajectories using Trackpy (PTV)."""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np
import pandas as pd

from .config import ExclusionConfig, PreprocessingConfig, TrackingConfig
from .exclusion_mask import build_exclusion_mask
from .preprocessing import preprocess_frame


TRACK_COLUMNS = [
    "particle", "source_frame", "time_s", "x", "y", "vx", "vy", "speed"
]


def track_particles(
    frames: Iterable[tuple[int, np.ndarray]],
    fps: float,
    preprocessing: PreprocessingConfig,
    tracking: TrackingConfig,
    mm_per_pixel: float = 0.0,
    exclusion: ExclusionConfig | None = None,
) -> pd.DataFrame:
    """Detect and link particle centroids; output tracks in px or mm.

    `frame` in Trackpy counts PROCESSED frames; `source_frame` retains
    actual video positions so time differences remain physically correct.
    """
    try:
        import trackpy as tp
    except ImportError as exc:
        raise ImportError(
            "Tracking requires trackpy: pip install -e '.[tracking]'"
        ) from exc

    feature_chunks = []
    for selected_index, (source_frame, frame) in enumerate(frames):
        grayscale = preprocess_frame(frame, preprocessing)
        features = tp.locate(
            grayscale,
            diameter=tracking.diameter,
            minmass=tracking.min_mass,
        )
        if exclusion is not None and not features.empty:
            mask = build_exclusion_mask(frame, exclusion)
            if mask.any():
                # Also reject centroids whose detection neighborhood touches
                # the mask; filtering precedes linking into trajectories.
                mask = cv2.dilate(
                    mask.astype(np.uint8),
                    np.ones((tracking.diameter, tracking.diameter), dtype=np.uint8),
                ).astype(bool)
                x = np.clip(np.rint(features["x"]).astype(int), 0, mask.shape[1] - 1)
                y = np.clip(np.rint(features["y"]).astype(int), 0, mask.shape[0] - 1)
                features = features.loc[~mask[y, x]]
        if not features.empty:
            features = features.copy()
            features["frame"] = selected_index
            features["source_frame"] = source_frame
            feature_chunks.append(features)

    if not feature_chunks:
        return pd.DataFrame(columns=TRACK_COLUMNS)

    combined = pd.concat(feature_chunks, ignore_index=True)
    linked = tp.link_df(
        combined, search_range=tracking.search_range, memory=tracking.memory
    )
    linked = linked.sort_values(["particle", "source_frame"]).copy()
    scale = mm_per_pixel if mm_per_pixel > 0 else 1.0
    linked["time_s"] = linked["source_frame"] / fps
    linked["x"] = linked["x"] * scale
    # Physical origin is bottom left, unlike image coordinates.
    # Absolute y origin doesn't affect velocities. Negate instead of using H.
    linked["y"] = -linked["y"] * scale

    grouped = linked.groupby("particle", sort=False)
    dt = grouped["time_s"].diff()
    linked["vx"] = grouped["x"].diff() / dt
    linked["vy"] = grouped["y"].diff() / dt
    linked["speed"] = np.hypot(linked["vx"], linked["vy"])
    return linked[TRACK_COLUMNS].reset_index(drop=True)
