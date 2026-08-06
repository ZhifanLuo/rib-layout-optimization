"""Eight-node serendipity shell element for fixed-layout verification."""

from __future__ import annotations

import numpy as np

from .shell import _local_frame


def shape_q8(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """Return Q8 shape functions and natural derivatives."""
    n = np.zeros(8)
    dn = np.zeros((8, 2))
    corners = (
        (-1, -1, 1+xi+eta), (1, -1, 1-xi+eta),
        (1, 1, 1-xi-eta), (-1, 1, 1+xi-eta),
    )
    for i, (sx, sy, c) in enumerate(corners):
        ax, ay = 1+sx*xi, 1+sy*eta
        n[i] = -0.25*ax*ay*c
        dn[i, 0] = -0.25*(sx*ay*c- sx*ax*ay)
        dn[i, 1] = -0.25*(ax*sy*c- sy*ax*ay)
    n[4] = 0.5*(1-xi**2)*(1-eta)
    n[5] = 0.5*(1+xi)*(1-eta**2)
    n[6] = 0.5*(1-xi**2)*(1+eta)
    n[7] = 0.5*(1-xi)*(1-eta**2)
    dn[4] = [-xi*(1-eta), -0.5*(1-xi**2)]
    dn[5] = [0.5*(1-eta**2), -(1+xi)*eta]
    dn[6] = [-xi*(1+eta), 0.5*(1-xi**2)]
    dn[7] = [-0.5*(1-eta**2), -(1-xi)*eta]
    return n, dn


def shell_q8_stiffness_components(
    xyz: np.ndarray,
    E: float,
    nu: float,
    shear_correction: float = 5.0/6.0,
    drilling_factor: float = 1.0e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return thickness-linear and cubic Q8 shell stiffness components."""
    xyz = np.asarray(xyz)
    if xyz.shape != (8, 3):
        raise ValueError("xyz must have shape (8, 3)")
    dtype = np.result_type(xyz.dtype, float)
    rotation, xy = _local_frame(xyz)
    plane = E/(1.0-nu**2)*np.array(
        [[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]], float,
    )
    Dm = plane
    Db = plane/12.0
    G = E/(2.0*(1.0+nu))
    Ds = shear_correction*G*np.eye(2)
    linear = np.zeros((48, 48), dtype=dtype)
    cubic = np.zeros((48, 48), dtype=dtype)
    points, weights = np.polynomial.legendre.leggauss(3)
    def raw_shear(xi: float, eta: float) -> tuple[np.ndarray, float]:
        n, dn_nat = shape_q8(xi, eta)
        jac = dn_nat.T@xy
        det = np.linalg.det(jac)
        if float(np.real(det)) <= 0.0:
            raise ValueError("inverted or zero-area shell element")
        dn = dn_nat@np.linalg.inv(jac)
        out = np.zeros((2, 48), dtype=dtype)
        for i, (dx, dy) in enumerate(dn):
            j = 6*i
            out[0, j+2] = dx
            out[0, j+4] = n[i]
            out[1, j+2] = dy
            out[1, j+3] = -n[i]
        return out, det

    # Project the Q8 shear field to a bilinear assumed-strain space.  This is
    # a compact MITC/B-bar style treatment: it removes the higher-order shear
    # components responsible for thin-shell locking without using one-point
    # integration (and therefore avoids the associated hourglass modes).
    projection_mass = np.zeros((4, 4), dtype=dtype)
    projection_rhs = np.zeros((2, 4, 48), dtype=dtype)
    for xi, wx in zip(points, weights):
        for eta, wy in zip(points, weights):
            raw, det = raw_shear(xi, eta)
            basis = np.array([1.0, xi, eta, xi*eta], dtype=dtype)
            weight = det*wx*wy
            projection_mass += np.outer(basis, basis)*weight
            projection_rhs += basis[None, :, None]*raw[:, None, :]*weight
    projection_coefficients = np.array([
        np.linalg.solve(projection_mass, projection_rhs[component])
        for component in range(2)
    ])
    for xi, wx in zip(points, weights):
        for eta, wy in zip(points, weights):
            n, dn_nat = shape_q8(xi, eta)
            jac = dn_nat.T@xy
            det = np.linalg.det(jac)
            if float(np.real(det)) <= 0.0:
                raise ValueError("inverted or zero-area shell element")
            dn = dn_nat@np.linalg.inv(jac)
            bm = np.zeros((3, 48), dtype=dtype)
            bb = np.zeros((3, 48), dtype=dtype)
            bd = np.zeros((1, 48), dtype=dtype)
            for i, (dx, dy) in enumerate(dn):
                j = 6*i
                bm[0, j] = dx
                bm[1, j+1] = dy
                bm[2, j] = dy
                bm[2, j+1] = dx
                bb[0, j+4] = -dx
                bb[1, j+3] = dy
                bb[2, j+3] = dx
                bb[2, j+4] = -dy
                bd[0, j] = 0.5*dy
                bd[0, j+1] = -0.5*dx
                bd[0, j+5] = n[i]
            weight = det*wx*wy
            linear += (bm.T@Dm@bm
                       +drilling_factor*E*(bd.T@bd))*weight
            cubic += (bb.T@Db@bb)*weight
            raw, det = raw_shear(xi, eta)
            basis = np.array([1.0, xi, eta, xi*eta], dtype=dtype)
            bs = np.array([
                basis@projection_coefficients[0],
                basis@projection_coefficients[1],
            ])
            linear += bs.T@Ds@bs*(det*wx*wy)
    transform = np.zeros((48, 48), dtype=dtype)
    for i in range(8):
        transform[6*i:6*i+3, 6*i:6*i+3] = rotation
        transform[6*i+3:6*i+6, 6*i+3:6*i+6] = rotation
    return (
        0.5*(transform.T@linear@transform
             +transform.T@linear.T@transform),
        0.5*(transform.T@cubic@transform
             +transform.T@cubic.T@transform),
    )


def shell_q8_stiffness(
    xyz: np.ndarray, thickness: float, E: float, nu: float,
) -> np.ndarray:
    """Return the 48x48 Q8 shell stiffness matrix."""
    if float(thickness) <= 0.0:
        raise ValueError("shell thickness must be positive")
    linear, cubic = shell_q8_stiffness_components(xyz, E, nu)
    t = float(thickness)
    return t*linear+t**3*cubic
