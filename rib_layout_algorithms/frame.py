"""Internal 3-D Euler--Bernoulli frame-element utilities.

The paper used commercial shell elements for the ribs.  The independent
implementation uses an energetically equivalent eccentric line stiffener:
the rib's rectangular section is placed at h/2 above the wall midsurface and
rigidly tied to the wall degrees of freedom.  This preserves axial, bending,
torsional and eccentric effects while keeping the implementation inspectable.
"""

from __future__ import annotations

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def local_axes(p0: np.ndarray, p1: np.ndarray) -> tuple[np.ndarray, float]:
    """Return rotation whose rows are local axes and the element length."""
    d = np.asarray(p1, float) - np.asarray(p0, float)
    length = float(np.linalg.norm(d))
    if length <= 1.0e-12:
        raise ValueError("zero-length frame element")
    ex = d / length
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(ex @ reference)) > 0.95:
        reference = np.array([0.0, 1.0, 0.0])
    ey = np.cross(reference, ex)
    ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    return np.vstack((ex, ey, ez)), length


def frame_local_stiffness(
    length: float, E: float, G: float, A: float, Iy: float, Iz: float, J: float
) -> np.ndarray:
    """12 x 12 local stiffness for a two-node 3-D frame."""
    L = float(length)
    k = np.zeros((12, 12), dtype=float)

    def add(indices: tuple[int, ...], block: np.ndarray) -> None:
        k[np.ix_(indices, indices)] += block

    add((0, 6), E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]]))
    add((3, 9), G * J / L * np.array([[1.0, -1.0], [-1.0, 1.0]]))

    # Transverse local-y displacement, bending about local z.
    cz = E * Iz / L**3
    add(
        (1, 5, 7, 11),
        cz
        * np.array(
            [
                [12, 6 * L, -12, 6 * L],
                [6 * L, 4 * L**2, -6 * L, 2 * L**2],
                [-12, -6 * L, 12, -6 * L],
                [6 * L, 2 * L**2, -6 * L, 4 * L**2],
            ],
            float,
        ),
    )

    # Transverse local-z displacement, bending about local y.
    cy = E * Iy / L**3
    add(
        (2, 4, 8, 10),
        cy
        * np.array(
            [
                [12, -6 * L, -12, -6 * L],
                [-6 * L, 4 * L**2, 6 * L, 2 * L**2],
                [-12, 6 * L, 12, 6 * L],
                [-6 * L, 2 * L**2, 6 * L, 4 * L**2],
            ],
            float,
        ),
    )
    return k


def frame_global_stiffness(
    p0: np.ndarray,
    p1: np.ndarray,
    E: float,
    nu: float,
    A: float,
    Iy: float,
    Iz: float,
    J: float,
) -> np.ndarray:
    R, length = local_axes(p0, p1)
    G = E / (2.0 * (1.0 + nu))
    local = frame_local_stiffness(length, E, G, A, Iy, Iz, J)
    transform = np.zeros((12, 12), float)
    for offset in (0, 3, 6, 9):
        transform[offset : offset + 3, offset : offset + 3] = R
    return transform.T @ local @ transform


def rigid_offset_transform(offset: np.ndarray) -> np.ndarray:
    """Map [translation, rotation] at a base point to an offset point."""
    transform = np.eye(6)
    transform[:3, 3:] = -skew(np.asarray(offset, float))
    return transform
