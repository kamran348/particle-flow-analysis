import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from particleflow.config import ExclusionConfig, PreprocessingConfig, TrackingConfig
from particleflow.exclusion import build_exclusion_mask, excluded_piv_windows
from particleflow.tracking import track_particles


def test_color_mask_follows_overlay_and_covers_gray_label():
    frame = np.full((100, 100, 3), 120, dtype=np.uint8)
    frame[20, 20] = (0, 0, 255)
    frame[21:24, 21:24] = 50  # nearby gray tooltip
    frame[70, 70] = (0, 255, 0)
    config = ExclusionConfig(overlay_colors=("red",), padding=5)
    before = build_exclusion_mask(frame, config)
    assert before[21:24, 21:24].all()
    assert not before[70, 70]
    assert not before[0, 0]
    frame[20, 20] = 120
    frame[50, 50] = (0, 0, 255)
    after = build_exclusion_mask(frame, config)
    assert after[50, 50]
    assert not after[20, 20]
    both_colors = build_exclusion_mask(
        frame, ExclusionConfig(overlay_colors=("red", "green"), padding=0)
    )
    assert both_colors[50, 50] and both_colors[70, 70]


def test_fixed_rectangles_and_disabled_mask():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    assert not build_exclusion_mask(frame, ExclusionConfig()).any()
    mask = build_exclusion_mask(frame, ExclusionConfig(rectangles=((2, 3, 5, 7),)))
    assert mask.sum() == 12
    assert mask[3:7, 2:5].all()
    assert not mask[7, 5]
    with pytest.raises(ValueError, match="outside"):
        build_exclusion_mask(frame, ExclusionConfig(rectangles=((2, 3, 25, 7),)))


def test_search_window_edges_are_excluded_even_when_center_is_clear():
    mask = np.zeros((101, 101), dtype=bool)
    mask[47, 47] = True
    # Windows start at 0, 24, 48, regardless of the unused image border.
    bad = excluded_piv_windows(mask, (3, 3), 48, 24)
    expected = np.array([[True, True, False],
                         [True, True, False],
                         [False, False, False]])
    np.testing.assert_array_equal(bad, expected)
    assert not mask[24, 24]  # testing only the vector center would miss this


def test_tracking_excludes_nearby_detections_before_linking(monkeypatch):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[20, 20] = (0, 0, 255)

    def locate(*args, **kwargs):
        return pd.DataFrame({"x": [22.0, 80.0], "y": [20.0, 80.0]})

    def link(features, **kwargs):
        assert features["x"].tolist() == [80.0]
        return features.assign(particle=0)

    monkeypatch.setitem(sys.modules, "trackpy", SimpleNamespace(locate=locate, link_df=link))
    result = track_particles(
        [(0, frame)], 15, PreprocessingConfig(), TrackingConfig(),
        exclusion=ExclusionConfig(overlay_colors=("red",), padding=0),
    )
    assert result["x"].tolist() == [80.0]
