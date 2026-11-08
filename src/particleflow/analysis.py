"""Physical quantities that can be inferred from 2D velocity fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .piv import VelocityField


@dataclass
class FlowMetrics:
    speed: np.ndarray
    vorticity: np.ndarray           # d(v)/dx - d(u)/dy, 1/s
    divergence: np.ndarray          # d(u)/dx + d(v)/dy, 1/s
    strain_rate_magnitude: np.ndarray  # sqrt(2 S:S), 1/s


def calculate_flow_properties(field: VelocityField) -> FlowMetrics:
    """Compute 2D speed, vorticity, divergence, strain-rate magnitude.

    Derivatives near missing vectors become NaN. This is intentional;
    validation/interpolation should be a conscious scientific decision.
    """
    if min(field.u.shape) < 2:
        raise ValueError("Need >= 2 grid points in each direction for derivatives")
    x_values = field.x[0, :]
    y_values = field.y[:, 0]  # can be descending; numpy.gradient supports that

    du_dy, du_dx = np.gradient(field.u, y_values, x_values)
    dv_dy, dv_dx = np.gradient(field.v, y_values, x_values)

    speed = np.hypot(field.u, field.v)
    vorticity = dv_dx - du_dy
    divergence = du_dx + dv_dy
    s_xy = 0.5 * (du_dy + dv_dx)
    strain_rate = np.sqrt(2.0 * (du_dx**2 + dv_dy**2 + 2.0 * s_xy**2))
    return FlowMetrics(speed, vorticity, divergence, strain_rate)


def summarize_field(field: VelocityField, metrics: FlowMetrics) -> dict:
    speeds = metrics.speed[np.isfinite(metrics.speed)]
    if speeds.size == 0:
        raise ValueError("No valid PIV vectors were found. Adjust illumination or PIV settings.")
    return {
        "velocity_unit": field.velocity_unit,
        "position_unit": field.position_unit,
        "valid_vector_fraction": float(field.valid.mean()),
        "mean_speed": float(speeds.mean()),
        "max_speed": float(speeds.max()),
        "median_speed": float(np.median(speeds)),
        "mean_u": float(np.nanmean(field.u)),
        "mean_v": float(np.nanmean(field.v)),
        "note": (
            "Velocity is 2D and represents visible tracer motion. "
            "Net force, pressure, and volumetric flow rate require further "
            "physical assumptions and measurements."
        ),
    }
