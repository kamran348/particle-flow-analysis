"""Write publication-friendly vector, speed and streamline plots to disk."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save images without requiring a display or Jupyter
import matplotlib.pyplot as plt
import numpy as np

from .piv import VelocityField
from .analysis import FlowMetrics


def _axes(name: str, field: VelocityField):
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    ax.set_title(name)
    ax.set_xlabel(f"x [{field.position_unit}]")
    ax.set_ylabel(f"y [{field.position_unit}]")
    ax.set_aspect("equal", adjustable="box")
    return fig, ax


def save_plots(field: VelocityField, metrics: FlowMetrics, output_dir: str | Path) -> None:
    """Plot time-averaged field; streamlines are NOT individual trajectories."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # OpenPIV row order is top-to-bottom; streamline y grid must ascend.
    x = field.x[::-1, :]
    y = field.y[::-1, :]
    u = field.u[::-1, :]
    v = field.v[::-1, :]
    speed = metrics.speed[::-1, :]
    good = np.isfinite(u) & np.isfinite(v)

    # Velocity vectors
    fig, ax = _axes("Mean velocity vectors", field)
    ax.quiver(x[good], y[good], u[good], v[good],
              color="tab:blue", angles="xy", scale_units="xy", scale=None)
    fig.savefig(output_dir / "velocity_vectors.png", dpi=160)
    plt.close(fig)

    # Velocity magnitude
    fig, ax = _axes("Mean velocity magnitude", field)
    img = ax.pcolormesh(x, y, np.ma.masked_invalid(speed), shading="auto", cmap="viridis")
    fig.colorbar(img, ax=ax, label=f"Speed [{field.velocity_unit}]")
    fig.savefig(output_dir / "speed_map.png", dpi=160)
    plt.close(fig)

    # Streamplot accepts masked arrays: lines stop at invalid data regions.
    fig, ax = _axes("Streamlines of mean flow", field)
    if min(u.shape) >= 2 and good.any():
        try:
            lines = ax.streamplot(
                x[0, :], y[:, 0],
                np.ma.masked_invalid(u), np.ma.masked_invalid(v),
                color=np.ma.masked_invalid(speed),
                cmap="viridis", density=1.2, linewidth=1.2,
            )
            fig.colorbar(lines.lines, ax=ax, label=f"Speed [{field.velocity_unit}]")
        except ValueError as exc:
            # Sparse valid vectors can prevent streamplot integration.
            ax.text(0.5, 0.5, f"Insufficient valid vectors for streamlines: {exc}",
                    transform=ax.transAxes, ha="center", va="center", wrap=True)
    fig.savefig(output_dir / "streamlines.png", dpi=160)
    plt.close(fig)
