from pathlib import Path

import pytest

from particleflow.config import AppConfig, ExclusionConfig, PIVConfig, load_config


def test_default_configuration_loads():
    config = load_config(Path("configs/default.toml"))
    assert config.piv.window_size == 32
    assert config.video.step >= 1
    assert config.calibration.mm_per_pixel == 0


def test_rejects_bad_overlap():
    with pytest.raises(ValueError, match="overlap"):
        AppConfig(piv=PIVConfig(overlap=48)).validate()


@pytest.mark.parametrize("settings", [
    ExclusionConfig(padding=-1),
    ExclusionConfig(padding=1.5),
    ExclusionConfig(overlay_colors=("purple",)),
    ExclusionConfig(rectangles=((20, 0, 10, 10),)),
    ExclusionConfig(rectangles=((0, 0, 10.5, 10),)),
])
def test_rejects_invalid_exclusion(settings):
    with pytest.raises(ValueError, match="[Ee]xclusion"):
        AppConfig(exclusion=settings).validate()


def test_loads_exclusion_from_toml(tmp_path):
    path = tmp_path / "mask.toml"
    path.write_text('[exclusion]\noverlay_colors = ["red", "green"]\n'
                    'padding = 12\nrectangles = [[1, 2, 5, 6]]\n')
    config = load_config(path)
    assert config.exclusion.overlay_colors == ["red", "green"]
    assert config.exclusion.rectangles == [[1, 2, 5, 6]]
    assert config.exclusion.padding == 12
