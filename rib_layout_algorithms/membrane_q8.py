"""Eight-node serendipity Q8 membrane verification model."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix

from .membrane import MembraneQ4Model


class MembraneQ8Model(MembraneQ4Model):
    """Higher-order membrane baseline with the same boundary semantics as Q4."""

    def __init__(self, width, height, nx, ny, wall_thickness, E, nu, loads, supports):
        self.width, self.height = float(width), float(height)
        self.nx, self.ny = int(nx), int(ny)
        self.wall_thickness = float(wall_thickness)
        self.E, self.nu = float(E), float(nu)
        self.dx, self.dy = self.width/self.nx, self.height/self.ny
        self._node_ids = {}
        for iy in range(2*self.ny+1):
            for ix in range(2*self.nx+1):
                # Q8 has no element-center node; omit odd/odd grid points.
                if ix % 2 == 1 and iy % 2 == 1:
                    continue
                self._node_ids[(ix, iy)] = len(self._node_ids)
        self.nnode = len(self._node_ids)
        self.ndof = 2*self.nnode
        self.base_stiffness = self._build_ground_membrane()
        self.fixed_dofs = self._build_supports(supports)
        self.base_stiffness = (
            self.base_stiffness+self._build_support_springs(supports)
        ).tocsc()
        self.free_dofs = np.setdiff1d(np.arange(self.ndof), self.fixed_dofs)
        self.load_vectors, self.load_weights = self._build_loads(loads)

    def node(self, ix: int, iy: int) -> int:
        return self._node_ids[(ix, iy)]

    @staticmethod
    def _shape_q8(xi: float, eta: float):
        n = np.zeros(8)
        dn = np.zeros((8, 2))
        corners = (
            (-1, -1, 1+xi+eta), (1, -1, 1-xi+eta),
            (1, 1, 1-xi-eta), (-1, 1, 1+xi-eta),
        )
        for i, (sx, sy, c) in enumerate(corners):
            ax, ay = 1+sx*xi, 1+sy*eta
            n[i] = -0.25*ax*ay*c
            dc_x = -sx
            dc_y = -sy
            dn[i, 0] = -0.25*(sx*ay*c+ax*ay*dc_x)
            dn[i, 1] = -0.25*(ax*sy*c+ax*ay*dc_y)
        n[4] = 0.5*(1-xi**2)*(1-eta)
        n[5] = 0.5*(1+xi)*(1-eta**2)
        n[6] = 0.5*(1-xi**2)*(1+eta)
        n[7] = 0.5*(1-xi)*(1-eta**2)
        dn[4] = [-xi*(1-eta), -0.5*(1-xi**2)]
        dn[5] = [0.5*(1-eta**2), -(1+xi)*eta]
        dn[6] = [-xi*(1+eta), 0.5*(1-xi**2)]
        dn[7] = [-0.5*(1-eta**2), -(1-xi)*eta]
        return n, dn

    def _element_stiffness(self, x: float, y: float) -> np.ndarray:
        jac = np.diag([x/2, y/2])
        det = float(np.linalg.det(jac))
        plane = self.E/(1-self.nu**2)*np.array([
            [1, self.nu, 0], [self.nu, 1, 0],
            [0, 0, (1-self.nu)/2],
        ])
        out = np.zeros((16, 16))
        points, weights = np.polynomial.legendre.leggauss(3)
        for xi, wx in zip(points, weights):
            for eta, wy in zip(points, weights):
                _, dn_nat = self._shape_q8(xi, eta)
                dn = dn_nat@np.linalg.inv(jac)
                B = np.zeros((3, 16))
                for i, (dx, dy) in enumerate(dn):
                    j = 2*i
                    B[0, j] = dx
                    B[1, j+1] = dy
                    B[2, j] = dy
                    B[2, j+1] = dx
                out += self.wall_thickness*(B.T@plane@B)*det*wx*wy
        return out

    def _build_ground_membrane(self) -> csc_matrix:
        rows: list[int] = []; cols: list[int] = []; data: list[float] = []
        for iy in range(self.ny):
            for ix in range(self.nx):
                nodes = [
                    self.node(2*ix, 2*iy), self.node(2*ix+2, 2*iy),
                    self.node(2*ix+2, 2*iy+2), self.node(2*ix, 2*iy+2),
                    self.node(2*ix+1, 2*iy), self.node(2*ix+2, 2*iy+1),
                    self.node(2*ix+1, 2*iy+2), self.node(2*ix, 2*iy+1),
                ]
                dofs = np.array([2*n+c for n in nodes for c in (0, 1)])
                rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
                values = self._element_stiffness(self.dx, self.dy)
                rows.extend(rr.ravel().tolist()); cols.extend(cc.ravel().tolist())
                data.extend(values.ravel().tolist())
        return coo_matrix((data, (rows, cols)), shape=(self.ndof, self.ndof)).tocsc()

    def interpolation(self, point):
        x = float(np.clip(point[0], 0, self.width))
        y = float(np.clip(point[1], 0, self.height))
        ix = min(int(x/self.dx), self.nx-1)
        iy = min(int(y/self.dy), self.ny-1)
        xi = 2*(x-ix*self.dx)/self.dx-1
        eta = 2*(y-iy*self.dy)/self.dy-1
        nodes = np.array([
            self.node(2*ix, 2*iy), self.node(2*ix+2, 2*iy),
            self.node(2*ix+2, 2*iy+2), self.node(2*ix, 2*iy+2),
            self.node(2*ix+1, 2*iy), self.node(2*ix+2, 2*iy+1),
            self.node(2*ix+1, 2*iy+2), self.node(2*ix, 2*iy+1),
        ])
        weights, _ = self._shape_q8(xi, eta)
        return nodes, weights
