"""Internal four-node MITC4 Reissner--Mindlin flat-shell finite element.

The element retains six global degrees of freedom per node. Membrane and
bending terms use 2x2 Gauss integration. Transverse shear uses the MITC4
assumed-natural-strain interpolation, avoiding both thin-shell shear locking
and the hourglass modes of one-point shear integration. The drilling rotation
is weakly tied to the continuum in-plane rotation, so only the physical rigid
rotation remains unstiffened.
"""

from __future__ import annotations

import numpy as np


def shape_q4(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """Return shape functions and natural derivatives [d/dxi, d/deta]."""
    n = 0.25 * np.array(
        [(1-xi)*(1-eta), (1+xi)*(1-eta), (1+xi)*(1+eta), (1-xi)*(1+eta)],
        float,
    )
    dn = 0.25 * np.array(
        [
            [-(1-eta), -(1-xi)],
            [ +(1-eta), -(1+xi)],
            [ +(1+eta), +(1+xi)],
            [-(1+eta), +(1-xi)],
        ],
        float,
    )
    return n, dn


def _local_frame(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    complex_step = np.iscomplexobj(xyz)
    origin = xyz[0]
    ex = xyz[1] - origin
    ex_norm = np.sqrt(ex @ ex)
    if abs(ex_norm) <= 1.0e-14:
        raise ValueError("degenerate shell element")
    ex /= ex_norm
    trial = xyz[3] - origin
    normal = np.cross(ex, trial)
    norm = np.sqrt(normal @ normal)
    if abs(norm) <= 1.0e-14:
        raise ValueError("degenerate shell element")
    normal /= norm
    ey = np.cross(normal, ex)
    rotation = np.vstack((ex, ey, normal))
    local_xyz = (xyz - origin) @ rotation.T
    if (
        not complex_step
        and np.max(np.abs(local_xyz[:, 2]))
        > 1.0e-8 * max(np.ptp(local_xyz[:, 0]), np.ptp(local_xyz[:, 1]), 1.0)
    ):
        raise ValueError("flat-shell element nodes are not coplanar")
    return rotation, local_xyz[:, :2]


def shell_q4_stiffness(
    xyz: np.ndarray,
    thickness: float,
    E: float,
    nu: float,
    shear_correction: float = 5.0 / 6.0,
    drilling_factor: float = 1.0e-4,
) -> np.ndarray:
    """Return the 24x24 global stiffness of a planar Q4 shell."""
    linear, cubic = shell_q4_stiffness_components(
        xyz,
        E,
        nu,
        shear_correction=shear_correction,
        drilling_factor=drilling_factor,
    )
    t = float(thickness)
    if t <= 0:
        raise ValueError("shell thickness must be positive")
    return t*linear+t**3*cubic


def shell_q4_stiffness_components(
    xyz: np.ndarray,
    E: float,
    nu: float,
    shear_correction: float = 5.0 / 6.0,
    drilling_factor: float = 1.0e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return thickness-linear and thickness-cubic Q4 shell matrices."""
    xyz = np.asarray(xyz)
    if xyz.shape != (4, 3):
        raise ValueError("xyz must have shape (4, 3)")
    dtype = np.result_type(xyz.dtype, float)
    complex_step = np.iscomplexobj(xyz)
    rotation, xy = _local_frame(xyz)
    plane = E / (1.0 - nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]], float)
    Dm = plane
    Db = plane / 12.0
    G = E / (2.0 * (1.0 + nu))
    Ds = shear_correction * G * np.eye(2)
    local_linear = np.zeros((24, 24), dtype=dtype)
    local_cubic = np.zeros((24, 24), dtype=dtype)

    def covariant_shear(xi: float, eta: float) -> np.ndarray:
        """Return covariant shear rows [gamma_xi, gamma_eta]."""
        n, dn_nat = shape_q4(xi, eta)
        jac = dn_nat.T @ xy
        determinant = np.linalg.det(jac)
        if float(np.real(determinant)) <= 0:
            raise ValueError("inverted or zero-area shell element")
        out = np.zeros((2, 24), dtype=dtype)
        for i, (dxi, deta) in enumerate(dn_nat):
            j = 6 * i
            # gamma_xi = w,xi + x,xi*theta_y - y,xi*theta_x
            out[0, j+2] = dxi
            out[0, j+3] = -jac[0, 1] * n[i]
            out[0, j+4] = +jac[0, 0] * n[i]
            # gamma_eta = w,eta + x,eta*theta_y - y,eta*theta_x
            out[1, j+2] = deta
            out[1, j+3] = -jac[1, 1] * n[i]
            out[1, j+4] = +jac[1, 0] * n[i]
        return out

    # MITC4 tying-point shear rows. The xi component is sampled at the
    # midpoints of eta=-1,+1 edges; the eta component at xi=+1,-1 edges.
    shear_xi_bottom = covariant_shear(0.0, -1.0)[0]
    shear_xi_top = covariant_shear(0.0, +1.0)[0]
    shear_eta_right = covariant_shear(+1.0, 0.0)[1]
    shear_eta_left = covariant_shear(-1.0, 0.0)[1]
    drilling_modulus = drilling_factor * E

    gauss = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))
    for xi in gauss:
        for eta in gauss:
            n, dn_nat = shape_q4(xi, eta)
            jac = dn_nat.T @ xy
            det = np.linalg.det(jac)
            if float(np.real(det)) <= 0:
                raise ValueError("inverted or zero-area shell element")
            dn = dn_nat @ np.linalg.inv(jac)
            Bm = np.zeros((3, 24), dtype=dtype)
            Bb = np.zeros((3, 24), dtype=dtype)
            Bd = np.zeros((1, 24), dtype=dtype)
            for i, (dx, dy) in enumerate(dn):
                j = 6 * i
                Bm[0, j] = dx
                Bm[1, j+1] = dy
                Bm[2, j] = dy; Bm[2, j+1] = dx
                # theta_x = dw/dy, theta_y = -dw/dx
                Bb[0, j+4] = -dx
                Bb[1, j+3] = dy
                Bb[2, j+3] = dx; Bb[2, j+4] = -dy
                # theta_z equals the continuum rotation
                # omega_z=0.5*(v,x-u,y), including for rigid-body rotation.
                Bd[0, j] = +0.5 * dy
                Bd[0, j+1] = -0.5 * dx
                Bd[0, j+5] = n[i]

            assumed_covariant = np.vstack((
                0.5 * ((1.0-eta)*shear_xi_bottom + (1.0+eta)*shear_xi_top),
                0.5 * ((1.0+xi)*shear_eta_right + (1.0-xi)*shear_eta_left),
            ))
            Bs = np.linalg.solve(jac, assumed_covariant)
            local_linear += (
                Bm.T @ Dm @ Bm
                + Bs.T @ Ds @ Bs
                + drilling_modulus * (Bd.T @ Bd)
            ) * det
            local_cubic += (Bb.T @ Db @ Bb) * det

    transform = np.zeros((24, 24), dtype=dtype)
    for i in range(4):
        transform[6*i:6*i+3, 6*i:6*i+3] = rotation
        transform[6*i+3:6*i+6, 6*i+3:6*i+6] = rotation
    global_linear = transform.T @ local_linear @ transform
    global_cubic = transform.T @ local_cubic @ transform
    return (
        0.5*(global_linear+global_linear.T),
        0.5*(global_cubic+global_cubic.T),
    )
