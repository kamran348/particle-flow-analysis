import numpy as np
import pytest

from particleflow.config import PreprocessingConfig
from particleflow.preprocessing import preprocess_frame


def test_preprocess_returns_uint8_grayscale():
    rgb = np.zeros((10, 12, 3), dtype=np.uint8)
    rgb[3:6, 4:7] = [255, 255, 255]
    gray = preprocess_frame(rgb, PreprocessingConfig(gaussian_sigma=0))
    assert gray.shape == (10, 12)
    assert gray.dtype == np.uint8
    assert gray.max() == 255


def test_invalid_image_raises():
    with pytest.raises(ValueError):
        preprocess_frame(np.zeros((10, 10, 4), dtype=np.uint8), PreprocessingConfig())
