import numpy as np

from particleflow.analysis import calculate_flow_properties, summarize_field
from particleflow.piv import VelocityField


def test_solid_body_rotation_curl_and_divergence():
    # u = -y, v = x gives positive 2/s vorticity and zero divergence.
    x, y = np.meshgrid(np.arange(4.0), np.arange(3.0, -1.0, -1.0))
    u, v = -y, x
    field = VelocityField(x, y, u, v, np.ones_like(u, bool), "mm", "mm/s")
    result = calculate_flow_properties(field)
    np.testing.assert_allclose(result.vorticity, 2.0, atol=1e-10)
    np.testing.assert_allclose(result.divergence, 0.0, atol=1e-10)
    summary = summarize_field(field, result)
    assert summary["valid_vector_fraction"] == 1
