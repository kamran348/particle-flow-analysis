import numpy as np

from particleflow.piv import VelocityField
from particleflow.analysis import calculate_flow_properties
from particleflow.plotting import save_plots


def test_plots_created(tmp_path):
    x, y = np.meshgrid(np.arange(4.0), np.arange(3.0, -1.0, -1.0))
    u = np.full_like(x, 2.0)
    v = np.zeros_like(x)
    field = VelocityField(x, y, u, v, np.ones_like(u, bool), "mm", "mm/s")
    save_plots(field, calculate_flow_properties(field), tmp_path)
    for name in ["streamlines.png", "speed_map.png", "velocity_vectors.png"]:
        assert (tmp_path / name).stat().st_size > 0
