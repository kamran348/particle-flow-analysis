"""Select persistent trajectories using full windows of decoded source frames."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .config import ConsistencyConfig


REPORT_COLUMNS = [
    "particle", "window", "first_source_frame", "last_source_frame",
    "expected_frames", "observed_frames", "fraction", "seen_first", "seen_last",
    "complete_window", "accepted",
]


def filter_consistent_tracks(
    tracks: pd.DataFrame,
    source_frames: Sequence[int],
    config: ConsistencyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep observations only in qualifying, nonoverlapping temporal windows.

    Link IDs must already follow motion. Coverage counts actual observations,
    including empty decoded frames in the denominator. Partial final windows
    never qualify. No observations are interpolated or invented.
    """
    if len(source_frames) < config.window_frames:
        raise ValueError(
            f"Consistency needs at least {config.window_frames} source frames; "
            f"decoded {len(source_frames)}. Extend the range or reduce "
            "consistency.window_frames."
        )
    frames = np.asarray(source_frames, dtype=int)
    if not np.all(np.diff(frames) == 1):
        raise ValueError("Consistency must inspect every source frame (step=1)")
    if tracks.duplicated(["particle", "source_frame"]).any():
        raise ValueError("A particle must have at most one detection per source frame")
    if not tracks["source_frame"].isin(frames).all():
        raise ValueError("Tracks contain frames outside the consistency range")

    selected = tracks.copy()
    selected["_window"] = ((selected["source_frame"] - frames[0]) // config.window_frames).astype(int)
    keep = np.zeros(len(selected), dtype=bool)
    records = []
    # Positional indices keep this correct for arbitrary incoming DataFrame indices.
    for (particle, window), positions in selected.groupby(["particle", "_window"]).indices.items():
        start = int(window) * config.window_frames
        window_frames = frames[start:start + config.window_frames]
        observed = selected.iloc[positions]["source_frame"]
        seen_first = bool(observed.eq(window_frames[0]).any())
        seen_last = bool(observed.eq(window_frames[-1]).any())
        complete = len(window_frames) == config.window_frames
        accepted = (
            complete
            and len(observed) / config.window_frames >= config.min_fraction
            and (not config.require_endpoints or (seen_first and seen_last))
        )
        keep[positions] = accepted
        records.append({
            "particle": int(particle), "window": int(window),
            "first_source_frame": int(window_frames[0]),
            "last_source_frame": int(window_frames[-1]),
            "expected_frames": len(window_frames), "observed_frames": len(observed),
            "fraction": len(observed) / len(window_frames),
            "seen_first": seen_first, "seen_last": seen_last,
            "complete_window": complete, "accepted": bool(accepted),
        })
    filtered = selected.loc[keep].drop(columns="_window").reset_index(drop=True)
    filtered["consistency_segment"] = pd.Series(index=filtered.index, dtype="int64")
    # Recompute derivatives after filtering: never retain velocities connecting
    # a rejected interval to an accepted interval.
    if not filtered.empty:
        filtered = filtered.sort_values(["particle", "source_frame"]).reset_index(drop=True)
        grouped = filtered.groupby("particle", sort=False)
        dt = grouped["time_s"].diff()
        previous_frame = grouped["source_frame"].shift()
        previous_window = ((previous_frame - frames[0]) // config.window_frames)
        current_window = ((filtered["source_frame"] - frames[0]) // config.window_frames)
        # Consecutive accepted windows may connect; skipped rejected windows cannot.
        connected = (current_window - previous_window) <= 1
        filtered["consistency_segment"] = (~connected).groupby(filtered["particle"]).cumsum().astype(int)
        filtered["vx"] = (grouped["x"].diff() / dt).where(connected)
        filtered["vy"] = (grouped["y"].diff() / dt).where(connected)
        filtered["speed"] = np.hypot(filtered["vx"], filtered["vy"])
    report = pd.DataFrame(records, columns=REPORT_COLUMNS)
    if not report.empty:
        report = report.sort_values(["window", "particle"]).reset_index(drop=True)
    return filtered, report
