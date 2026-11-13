import numpy as np
import pytest

from particleflow.config import ExclusionConfig, PIVConfig, PreprocessingConfig
from particleflow.exclusion_mask import build_exclusion_mask, excluded_piv_windows
from particleflow.piv import compute_piv_pair
from particleflow.preprocessing import preprocess_frame


def test_openpiv_detects_rightward_motion():
    pytest.importorskip("openpiv")
    # Reproducible spots shifted by 2 pixels in x.
    import cv2
    rng = np.random.default_rng(2)
    a = np.zeros((128, 128), dtype=np.uint8)
    for x, y in rng.integers(5, 123, (350, 2)):
        cv2.circle(a, (int(x), int(y)), 1, 230, -1)
    b = np.roll(a, shift=2, axis=1)
    result = compute_piv_pair(a, b, dt_seconds=0.1,
                              config=PIVConfig(signal_to_noise_threshold=1.0))
    assert result.valid.any()
    assert np.nanmedian(result.u) > 4.0  # px/s, expected ~20 px/s
    assert np.nanmedian(np.abs(result.v)) < 10.0


def test_moving_overlay_is_excluded_in_both_frames():
    pytest.importorskip("openpiv")
    rng = np.random.default_rng(12)
    a = rng.integers(0, 180, (128, 128, 3), dtype=np.uint8)
    b = np.roll(a, 2, axis=1)
    a[3, 3] = (0, 0, 255)
    b[85, 85] = (0, 0, 255)
    exclusion = ExclusionConfig(overlay_colors=("red",), padding=0)
    # Only the artificial annotation has color; keep the background grayscale.
    a[:] = a[:, :, :1]
    b[:] = b[:, :, :1]
    a[3, 3] = (0, 0, 255)
    b[85, 85] = (0, 0, 255)
    mask = build_exclusion_mask(a, exclusion) | build_exclusion_mask(b, exclusion)
    preprocessing = PreprocessingConfig(gaussian_sigma=0)
    a, b = preprocess_frame(a, preprocessing), preprocess_frame(b, preprocessing)
    config = PIVConfig(signal_to_noise_threshold=1.0)
    baseline = compute_piv_pair(a, b, 0.1, config)
    result = compute_piv_pair(a, b, 0.1, config, exclusion_mask=mask)
    bad = excluded_piv_windows(mask, result.u.shape, 48, 24)
    assert bad[0, 0] and bad[-1, -1]
    assert not result.valid[bad].any()
    assert np.isnan(result.u[bad]).all()
    assert np.isnan(result.v[bad]).all()
    assert result.valid[~bad].any()
    np.testing.assert_allclose(result.u[~bad], baseline.u[~bad], equal_nan=True)
    # Grid centers agree with the actual windows, without shifting to center
    # the unused pixels at the right/bottom image edges.
    assert result.x[0, 0] == 24
    assert result.y[0, 0] == 128 - 24
