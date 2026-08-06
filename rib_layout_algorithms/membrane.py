"""Two-dimensional Q4 membrane baseline for shell-model verification."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.sparse import csc_matrix, coo_matrix
from scipy.sparse.linalg import splu

from .model import AnalysisResult


class MembraneQ4Model:
    """Plane-stress Q4 model used only for membrane convergence diagnostics.

    The model deliberately has no ribs or rotational degrees of freedom.  It
    shares the physical load/support patch semantics with the shell model so
    that Case-II boundary discretization can be separated from shell bending,
    transverse shear, and drilling effects.
    """

    def __init__(
        self,
        width: float,
        height: float,
        nx: int,
        ny: int,
        wall_thickness: float,
        E: float,
        nu: float,
        loads: Sequence[dict],
        supports: dict,
    ) -> None:
        self.width, self.height = float(width), float(height)
        self.nx, self.ny = int(nx), int(ny)
        self.wall_thickness = float(wall_thickness)
        self.E, self.nu = float(E), float(nu)
        self.dx, self.dy = self.width/self.nx, self.height/self.ny
        self.nnode = (self.nx+1)*(self.ny+1)
        self.ndof = 2*self.nnode
        self.base_stiffness = self._build_ground_membrane()
        self.fixed_dofs = self._build_supports(supports)
        self.base_stiffness = (
            self.base_stiffness+self._build_support_springs(supports)
        ).tocsc()
        self.free_dofs = np.setdiff1d(np.arange(self.ndof), self.fixed_dofs)
        self.load_vectors, self.load_weights = self._build_loads(loads)

    def node(self, ix: int, iy: int) -> int:
        return iy*(self.nx+1)+ix

    def _shape(self, xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
        n = 0.25*np.array([
            (1-xi)*(1-eta), (1+xi)*(1-eta),
            (1+xi)*(1+eta), (1-xi)*(1+eta),
        ])
        dn = 0.25*np.array([
            [-(1-eta), -(1-xi)], [(1-eta), -(1+xi)],
            [(1+eta), (1+xi)], [-(1+eta), (1-xi)],
        ])
        return n, dn

    def _element_stiffness(self, x: float, y: float) -> np.ndarray:
        jac = np.diag([x/2.0, y/2.0])
        det = float(np.linalg.det(jac))
        plane = self.E/(1-self.nu**2)*np.array([
            [1, self.nu, 0],
            [self.nu, 1, 0],
            [0, 0, (1-self.nu)/2],
        ])
        out = np.zeros((8, 8))
        for xi in (-1/np.sqrt(3), 1/np.sqrt(3)):
            for eta in (-1/np.sqrt(3), 1/np.sqrt(3)):
                _, dn_nat = self._shape(xi, eta)
                dn = dn_nat@np.linalg.inv(jac)
                B = np.zeros((3, 8))
                for i, (dx, dy) in enumerate(dn):
                    j = 2*i
                    B[0, j] = dx
                    B[1, j+1] = dy
                    B[2, j] = dy
                    B[2, j+1] = dx
                out += self.wall_thickness*(B.T@plane@B)*det
        return out

    def _build_ground_membrane(self) -> csc_matrix:
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for iy in range(self.ny):
            for ix in range(self.nx):
                nodes = [
                    self.node(ix, iy), self.node(ix+1, iy),
                    self.node(ix+1, iy+1), self.node(ix, iy+1),
                ]
                dofs = np.array([2*n+c for n in nodes for c in (0, 1)])
                rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
                values = self._element_stiffness(self.dx, self.dy)
                rows.extend(rr.ravel().tolist())
                cols.extend(cc.ravel().tolist())
                data.extend(values.ravel().tolist())
        return coo_matrix((data, (rows, cols)), shape=(self.ndof, self.ndof)).tocsc()

    def interpolation(self, point: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        x = float(np.clip(point[0], 0, self.width))
        y = float(np.clip(point[1], 0, self.height))
        ix = min(int(x/self.dx), self.nx-1)
        iy = min(int(y/self.dy), self.ny-1)
        xi = (x-ix*self.dx)/self.dx
        eta = (y-iy*self.dy)/self.dy
        nodes = np.array([
            self.node(ix, iy), self.node(ix+1, iy),
            self.node(ix+1, iy+1), self.node(ix, iy+1),
        ])
        weights = np.array([
            (1-xi)*(1-eta), xi*(1-eta), xi*eta, (1-xi)*eta,
        ])
        return nodes, weights

    def _patch_bounds(self, center: Sequence[float], size: Sequence[float]):
        cx, cy = (float(value) for value in center)
        sx, sy = (float(value) for value in size)
        if sx <= 0 or sy <= 0:
            raise ValueError("patch sizes must be positive")
        x0, x1 = max(0., cx-sx/2), min(self.width, cx+sx/2)
        y0, y1 = max(0., cy-sy/2), min(self.height, cy+sy/2)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("patch does not intersect the domain")
        return x0, x1, y0, y1

    @staticmethod
    def _profile(x: float, y: float, bounds, name: str) -> float:
        if name == "uniform":
            return 1.0
        if name != "cosine":
            raise ValueError("profile must be uniform or cosine")
        x0, x1, y0, y1 = bounds
        rx = 2*(x-0.5*(x0+x1))/(x1-x0)
        ry = 2*(y-0.5*(y0+y1))/(y1-y0)
        return (np.pi/2)**2*np.cos(np.pi*rx/2)*np.cos(np.pi*ry/2)

    @staticmethod
    def _gauss(profile: str):
        if profile == "uniform":
            return (-1/np.sqrt(3), 1/np.sqrt(3))
        if profile == "cosine":
            return np.polynomial.legendre.leggauss(4)[0]
        raise ValueError("profile must be uniform or cosine")

    def _nodes_from_selector(self, selector: dict) -> list[int]:
        kind = selector.get("type", "points")
        if kind == "points":
            return sorted({self.interpolation(p)[0][0] for p in selector["points"]})
        if kind == "edge":
            edge = selector["edge"]
            if edge == "bottom": return [self.node(ix, 0) for ix in range(self.nx+1)]
            if edge == "top": return [self.node(ix, self.ny) for ix in range(self.nx+1)]
            if edge == "left": return [self.node(0, iy) for iy in range(self.ny+1)]
            if edge == "right": return [self.node(self.nx, iy) for iy in range(self.ny+1)]
        raise ValueError(f"unknown membrane support selector: {selector}")

    def _build_supports(self, supports: dict) -> np.ndarray:
        if supports.get("type") == "patch_springs":
            return np.array([], dtype=int)
        components = [c for c in supports.get("components", [0, 1]) if c < 2]
        nodes = self._nodes_from_selector(supports)
        return np.array(sorted(2*n+c for n in nodes for c in components), dtype=int)

    def _build_support_springs(self, supports: dict) -> csc_matrix:
        if supports.get("type") != "patch_springs":
            return csc_matrix((self.ndof, self.ndof))
        stiffness = float(supports["stiffness"])
        components = [c for c in supports.get("components", [0, 1]) if c < 2]
        profile = supports.get("profile", "uniform")
        gauss = self._gauss(profile)
        rows: list[int] = []; cols: list[int] = []; data: list[float] = []
        for patch in supports["patches"]:
            bounds = self._patch_bounds(patch.get("center", patch.get("point")), patch["size"])
            x0, x1, y0, y1 = bounds
            xm, ym, hx, hy = (0.5*(x0+x1), 0.5*(y0+y1), 0.5*(x1-x0), 0.5*(y1-y0))
            for xi in gauss:
                for eta in gauss:
                    point = (xm+hx*xi, ym+hy*eta)
                    nodes, shape = self.interpolation(point)
                    factor = stiffness*hx*hy*self._profile(*point, bounds, profile)
                    for c in components:
                        for i, ni in enumerate(nodes):
                            for j, nj in enumerate(nodes):
                                rows.append(2*int(ni)+c); cols.append(2*int(nj)+c)
                                data.append(factor*shape[i]*shape[j])
        return coo_matrix((data, (rows, cols)), shape=(self.ndof, self.ndof)).tocsc()

    def _build_loads(self, cases: Sequence[dict]):
        vectors: list[np.ndarray] = []; weights: list[float] = []
        for case in cases:
            f = np.zeros(self.ndof)
            for item in case["forces"]:
                value = np.asarray(item["value"], float)[:2]
                if "patch_size" not in item:
                    node = self.interpolation(item["point"])[0][0]
                    f[2*node:2*node+2] += value
                    continue
                bounds = self._patch_bounds(item["point"], item["patch_size"])
                x0, x1, y0, y1 = bounds
                xm, ym, hx, hy = 0.5*(x0+x1), 0.5*(y0+y1), 0.5*(x1-x0), 0.5*(y1-y0)
                profile = item.get("profile", "uniform")
                nodal: dict[int, float] = {}
                for xi in self._gauss(profile):
                    for eta in self._gauss(profile):
                        point = (xm+hx*xi, ym+hy*eta)
                        nodes, shape = self.interpolation(point)
                        factor = self._profile(*point, bounds, profile)
                        for node, weight in zip(nodes, shape):
                            nodal[int(node)] = nodal.get(int(node), 0.) + factor*weight
                total = sum(nodal.values())
                for node, weight in nodal.items():
                    f[2*node:2*node+2] += weight/total*value
            vectors.append(f); weights.append(float(case.get("weight", 1.)))
        w = np.asarray(weights); w /= w.sum()
        return vectors, w

    def stiffness(self, ribs=(), thicknesses=()):
        if ribs:
            raise ValueError("MembraneQ4Model does not support ribs")
        return self.base_stiffness

    def analyze(self, ribs=(), thicknesses=()):
        Kff = self.base_stiffness[self.free_dofs][:, self.free_dofs].tocsc()
        solver = splu(Kff)
        displacements = []
        compliances = []
        for load in self.load_vectors:
            u = np.zeros(self.ndof)
            u[self.free_dofs] = solver.solve(load[self.free_dofs])
            displacements.append(u)
            compliances.append(float(load@u))
        return AnalysisResult(
            compliance=float(self.load_weights@np.asarray(compliances)),
            displacements=displacements,
            load_compliances=compliances,
        )
