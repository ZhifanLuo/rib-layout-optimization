"""Consistent transverse pressure loads for a structured rectangular Q4 mesh."""

from __future__ import annotations

import numpy as np


def uniform_q4_pressure(model, pressure: dict) -> np.ndarray:
    """Integrate constant global-z traction, including supported nodes.

    Config: ``{"type": "uniform", "value": -0.01}`` in N/mm^2.
    For an affine rectangular Q4, integral(N_i dA) = dx*dy/4 exactly.
    Element assembly preserves force and first moments under refinement;
    no nearest-node point-load approximation or renormalization is used.
    """
    if pressure.get("type") != "uniform":
        raise ValueError("pressure type must be uniform")
    value = float(pressure["value"])
    if not np.isfinite(value):
        raise ValueError("pressure value must be finite")
    if model.nx <= 0 or model.ny <= 0 or model.dx <= 0 or model.dy <= 0:
        raise ValueError("pressure requires a positive rectangular mesh")
    force = np.zeros(model.ndof)
    quarter_force = value*model.dx*model.dy/4.0
    for iy in range(model.ny):
        for ix in range(model.nx):
            for x, y in ((ix, iy), (ix+1, iy), (ix+1, iy+1), (ix, iy+1)):
                force[6*model.node(x, y)+2] += quarter_force
    return force
