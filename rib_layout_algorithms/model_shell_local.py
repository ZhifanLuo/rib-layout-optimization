"""Locally refined shell/rib verification model."""

from __future__ import annotations

import numpy as np

from .model import Rib
from .model_shell import ShellStiffenedPlateModel


class LocallyRefinedShellStiffenedPlateModel(ShellStiffenedPlateModel):
    """Nonuniform Cartesian shell mesh refined near loads, supports and ribs.

    This class is intended for fixed-layout response studies. Optimization
    geometry sensitivities are deliberately not enabled because the current
    production sensitivity derivation assumes a uniform ground grid.
    """

    def __init__(
        self, width, height, nx, ny, wall_thickness, E, nu, loads, supports,
        refinement_ribs=(), refinement_factor: int = 2,
        interface_subdivisions_per_cell: int = 4,
        refinement_regions=(), refinement_margin=None,
        **kwargs,
    ):
        self._base_dx = float(width)/int(nx)
        self._base_dy = float(height)/int(ny)
        self._refinement_margin = refinement_margin
        self.x_coords = self._build_axis(
            float(width), int(nx), self._refinement_intervals(
                loads, supports, refinement_ribs, 0, refinement_regions,
            ), refinement_factor, refinement_margin,
        )
        self.y_coords = self._build_axis(
            float(height), int(ny), self._refinement_intervals(
                loads, supports, refinement_ribs, 1, refinement_regions,
            ), refinement_factor, refinement_margin,
        )
        effective_nx = len(self.x_coords)-1
        effective_ny = len(self.y_coords)-1
        super().__init__(
            width, height, effective_nx, effective_ny, wall_thickness,
            E, nu, loads, supports,
            interface_subdivisions_per_cell=interface_subdivisions_per_cell,
            **kwargs,
        )
        self._base_dx = float(width)/int(nx)
        self._base_dy = float(height)/int(ny)

    @staticmethod
    def _refinement_intervals(
        loads, supports, ribs, axis: int, refinement_regions=(),
    ):
        intervals = []
        for case in loads:
            for item in case["forces"]:
                if "patch_size" in item:
                    center, size = item["point"], item["patch_size"]
                    intervals.append((center[axis]-size[axis]/2,
                                      center[axis]+size[axis]/2))
        if supports.get("type") == "patch_springs":
            for patch in supports.get("patches", []):
                center, size = patch.get("center", patch.get("point")), patch["size"]
                intervals.append((center[axis]-size[axis]/2,
                                  center[axis]+size[axis]/2))
        for rib in ribs:
            p0, p1 = np.asarray(rib.p0, float), np.asarray(rib.p1, float)
            # A tensor-product Cartesian mesh cannot follow a diagonal strip
            # without creating a coordinate at every sample. Refine the rib's
            # bounding interval instead; this keeps the diagnostic mesh local
            # to the rib band rather than degenerating into a global mesh.
            intervals.append((float(min(p0[axis], p1[axis])),
                              float(max(p0[axis], p1[axis]))))
        for region in refinement_regions:
            if isinstance(region, dict):
                if "bounds" in region:
                    bounds = region["bounds"]
                else:
                    center, size = region["center"], region["size"]
                    bounds = (
                        center[0]-size[0]/2, center[0]+size[0]/2,
                        center[1]-size[1]/2, center[1]+size[1]/2,
                    )
            else:
                bounds = region
            if len(bounds) != 4:
                raise ValueError(
                    "refinement regions must be [x0, x1, y0, y1]"
                )
            intervals.append((float(bounds[2*axis]), float(bounds[2*axis+1])))
        return intervals

    @staticmethod
    def _build_axis(
        length, base_count, intervals, refinement_factor,
        refinement_margin=None,
    ):
        if refinement_factor < 1:
            raise ValueError("refinement_factor must be positive")
        base_step = length/base_count
        target = base_step/refinement_factor
        # Refine complete base cells that intersect a projected physical
        # interval.  Building an independent linspace for every overlapping
        # interval creates duplicate, nearly coincident coordinates and can
        # accidentally make a nominally local tensor mesh much denser than
        # requested (especially when several ribs overlap in projection).
        projected = []
        for left, right in intervals:
            margin = base_step if refinement_margin is None else float(refinement_margin)
            if margin < 0.0:
                raise ValueError("refinement_margin must be nonnegative")
            left = max(0.0, float(left)-margin)
            right = min(float(length), float(right)+margin)
            if right > left:
                projected.append((left, right))
        base_edges = np.linspace(0.0, float(length), int(base_count)+1)
        coordinates = [float(base_edges[0])]
        for left, right in zip(base_edges[:-1], base_edges[1:]):
            refine = any(
                right > interval_left and left < interval_right
                for interval_left, interval_right in projected
            )
            count = max(1, int(np.ceil((right-left)/target))) if refine else 1
            coordinates.extend(
                np.linspace(left, right, count+1)[1:].tolist()
            )
        return np.asarray(coordinates, float)

    def node_xy(self, node: int) -> np.ndarray:
        iy, ix = divmod(int(node), self.nx+1)
        return np.array([self.x_coords[ix], self.y_coords[iy]], float)

    def nearest_node(self, point):
        ix = int(np.clip(np.searchsorted(self.x_coords, point[0]), 1, self.nx))-1
        iy = int(np.clip(np.searchsorted(self.y_coords, point[1]), 1, self.ny))-1
        choices = []
        for j in (max(0, iy), min(self.ny, iy+1)):
            for i in (max(0, ix), min(self.nx, ix+1)):
                choices.append(self.node(i, j))
        return min(choices, key=lambda n: float(np.linalg.norm(self.node_xy(n)-point)))

    def interpolation(self, point):
        x = float(np.clip(point[0], 0., self.width))
        y = float(np.clip(point[1], 0., self.height))
        ix = min(max(int(np.searchsorted(self.x_coords, x, side="right")-1), 0), self.nx-1)
        iy = min(max(int(np.searchsorted(self.y_coords, y, side="right")-1), 0), self.ny-1)
        dx = self.x_coords[ix+1]-self.x_coords[ix]
        dy = self.y_coords[iy+1]-self.y_coords[iy]
        xi, eta = (x-self.x_coords[ix])/dx, (y-self.y_coords[iy])/dy
        nodes = np.array([
            self.node(ix, iy), self.node(ix+1, iy),
            self.node(ix+1, iy+1), self.node(ix, iy+1),
        ])
        weights = np.array([(1-xi)*(1-eta), xi*(1-eta), xi*eta, (1-xi)*eta])
        return nodes, weights

    def _nodes_in_patch(self, center, size):
        x0, x1, y0, y1 = self._patch_bounds(center, size)
        nodes = []
        for iy, y in enumerate(self.y_coords):
            if y0-1e-12 <= y <= y1+1e-12:
                for ix, x in enumerate(self.x_coords):
                    if x0-1e-12 <= x <= x1+1e-12:
                        nodes.append(self.node(ix, iy))
        if not nodes:
            raise ValueError("physical patch contains no refined mesh nodes")
        return nodes

    def rib_bottom_points(self, rib: Rib) -> np.ndarray:
        p0, p1 = np.asarray(rib.p0, float), np.asarray(rib.p1, float)
        delta = p1-p0
        length = max(float(np.linalg.norm(delta)), 1e-14)
        h = min(self._base_dx, self._base_dy)
        merge_tolerance = max(1e-5*h/length, self.interface_subdivisions_per_cell*1e-3*h/length)
        grid_parameters = []
        if abs(delta[0]) > 1e-14:
            grid_parameters.extend(
                (float(value-p0[0])/delta[0])
                for value in self.x_coords[1:-1]
                if 1e-12 < (value-p0[0])/delta[0] < 1-1e-12
            )
        if abs(delta[1]) > 1e-14:
            grid_parameters.extend(
                (float(value-p0[1])/delta[1])
                for value in self.y_coords[1:-1]
                if 1e-12 < (value-p0[1])/delta[1] < 1-1e-12
            )
        grid_parameters = [v for v in grid_parameters if v > merge_tolerance and 1-v > merge_tolerance]
        values = [0., 1., *grid_parameters]
        values.extend(
            v for v in np.linspace(0., 1., int(rib.segments)+1)[1:-1]
            if all(abs(v-g) > merge_tolerance for g in grid_parameters)
        )
        ordered = sorted(values)
        merged = [ordered[0]]
        for value in ordered[1:]:
            if value-merged[-1] > merge_tolerance:
                merged.append(value)
        if merged[-1] < 1.: merged.append(1.)
        subdivisions = self.interface_subdivisions_per_cell
        refined = [merged[0]]
        for left, right in zip(merged[:-1], merged[1:]):
            refined.extend(left+(right-left)*j/subdivisions for j in range(1, subdivisions+1))
        s = np.asarray(refined)
        return (1-s[:, None])*p0+s[:, None]*p1

    def geometry_gradient(self, *args, **kwargs):
        raise NotImplementedError("local-refinement model is response-only")
