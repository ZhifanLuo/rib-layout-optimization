"""Locally refined structured Q4 membrane verification model."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix

from .membrane import MembraneQ4Model


class LocallyRefinedMembraneQ4Model(MembraneQ4Model):
    """Q4 membrane model with smaller cells near load/support patches.

    The mesh remains Cartesian but has nonuniform x/y coordinates. One coarse
    cell of transition padding is refined around every physical patch. This
    class is a verification model; the production shell/rib model remains
    unchanged until the local-refinement behavior is validated.
    """

    def __init__(
        self, width, height, nx, ny, wall_thickness, E, nu, loads, supports,
        refinement_factor: int = 4,
    ):
        self.width, self.height = float(width), float(height)
        base_nx, base_ny = int(nx), int(ny)
        self.wall_thickness = float(wall_thickness)
        self.E, self.nu = float(E), float(nu)
        self.x_coords = self._build_axis(
            self.width, base_nx, self._patch_intervals(loads, supports, 0),
            refinement_factor,
        )
        self.y_coords = self._build_axis(
            self.height, base_ny, self._patch_intervals(loads, supports, 1),
            refinement_factor,
        )
        self.nx, self.ny = len(self.x_coords)-1, len(self.y_coords)-1
        self.dx = self.width/base_nx
        self.dy = self.height/base_ny
        self.nnode = (self.nx+1)*(self.ny+1)
        self.ndof = 2*self.nnode
        self.base_stiffness = self._build_ground_membrane()
        self.fixed_dofs = self._build_supports(supports)
        self.base_stiffness = (
            self.base_stiffness+self._build_support_springs(supports)
        ).tocsc()
        self.free_dofs = np.setdiff1d(np.arange(self.ndof), self.fixed_dofs)
        self.load_vectors, self.load_weights = self._build_loads(loads)

    @staticmethod
    def _patch_intervals(loads, supports, axis: int):
        intervals = []
        for case in loads:
            for item in case["forces"]:
                if "patch_size" in item:
                    cx, cy = item["point"]
                    sx, sy = item["patch_size"]
                    center, size = (cx, cy), (sx, sy)
                    intervals.append((center[axis]-size[axis]/2,
                                      center[axis]+size[axis]/2))
        if supports.get("type") == "patch_springs":
            for patch in supports.get("patches", []):
                center = patch.get("center", patch.get("point"))
                size = patch["size"]
                intervals.append((center[axis]-size[axis]/2,
                                  center[axis]+size[axis]/2))
        return intervals

    @staticmethod
    def _build_axis(length, base_count, intervals, refinement_factor):
        if refinement_factor < 1:
            raise ValueError("refinement_factor must be positive")
        base_step = length/base_count
        target = base_step/refinement_factor
        coordinates = list(np.linspace(0.0, length, base_count+1))
        for left, right in intervals:
            left = max(0.0, left-base_step)
            right = min(length, right+base_step)
            count = max(1, int(np.ceil((right-left)/target)))
            coordinates.extend(np.linspace(left, right, count+1).tolist())
        return np.asarray(sorted(set(round(value, 12) for value in coordinates)))

    def node(self, ix: int, iy: int) -> int:
        return iy*(self.nx+1)+ix

    def _element_stiffness(self, x: float, y: float) -> np.ndarray:
        return super()._element_stiffness(x, y)

    def _build_ground_membrane(self) -> csc_matrix:
        rows: list[int] = []; cols: list[int] = []; data: list[float] = []
        for iy in range(self.ny):
            for ix in range(self.nx):
                nodes = [
                    self.node(ix, iy), self.node(ix+1, iy),
                    self.node(ix+1, iy+1), self.node(ix, iy+1),
                ]
                dofs = np.array([2*n+c for n in nodes for c in (0, 1)])
                rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
                values = self._element_stiffness(
                    self.x_coords[ix+1]-self.x_coords[ix],
                    self.y_coords[iy+1]-self.y_coords[iy],
                )
                rows.extend(rr.ravel().tolist()); cols.extend(cc.ravel().tolist())
                data.extend(values.ravel().tolist())
        return coo_matrix((data, (rows, cols)), shape=(self.ndof, self.ndof)).tocsc()

    def interpolation(self, point):
        x = float(np.clip(point[0], 0, self.width))
        y = float(np.clip(point[1], 0, self.height))
        ix = min(max(int(np.searchsorted(self.x_coords, x, side="right")-1), 0), self.nx-1)
        iy = min(max(int(np.searchsorted(self.y_coords, y, side="right")-1), 0), self.ny-1)
        dx = self.x_coords[ix+1]-self.x_coords[ix]
        dy = self.y_coords[iy+1]-self.y_coords[iy]
        xi = (x-self.x_coords[ix])/dx
        eta = (y-self.y_coords[iy])/dy
        nodes = np.array([
            self.node(ix, iy), self.node(ix+1, iy),
            self.node(ix+1, iy+1), self.node(ix, iy+1),
        ])
        weights = np.array([
            (1-xi)*(1-eta), xi*(1-eta), xi*eta, (1-xi)*eta,
        ])
        return nodes, weights

