"""Independent structured 3-D solid reference for fixed rib layouts.

This module is deliberately response-only.  It is not a replacement for the
shell optimizer: a regular Cartesian Q1 hexahedral mesh is used for the plate
and ribs, while the physical rib footprint is integrated as a fixed area
fraction inside each planform cell.  Thus the base plate and all ribs share
the same continuum displacement field, including at rib intersections.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.sparse import csc_matrix, coo_matrix
from scipy.sparse.linalg import splu

from .model import AnalysisResult, Rib


def _elasticity_matrix(E: float, nu: float) -> np.ndarray:
    """Return the 3-D isotropic elasticity matrix in engineering notation."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    matrix = np.zeros((6, 6), float)
    matrix[:3, :3] = lam
    matrix[0, 0] += 2.0 * mu
    matrix[1, 1] += 2.0 * mu
    matrix[2, 2] += 2.0 * mu
    matrix[3:, 3:] = np.eye(3) * mu
    return matrix


def _hex8_stiffness(dx: float, dy: float, dz: float, E: float, nu: float) -> np.ndarray:
    """Full-integration stiffness of an axis-aligned trilinear brick."""
    gauss = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))
    signs = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], float)
    constitutive = _elasticity_matrix(float(E), float(nu))
    result = np.zeros((24, 24), float)
    jacobian = np.diag([dx / 2.0, dy / 2.0, dz / 2.0])
    inverse_jacobian = np.diag([2.0 / dx, 2.0 / dy, 2.0 / dz])
    determinant = dx * dy * dz / 8.0
    for xi in gauss:
        for eta in gauss:
            for zeta in gauss:
                derivatives = np.empty((8, 3), float)
                for node, (sx, sy, sz) in enumerate(signs):
                    derivatives[node] = 0.125 * np.array([
                        sx * (1.0 + sy * eta) * (1.0 + sz * zeta),
                        sy * (1.0 + sx * xi) * (1.0 + sz * zeta),
                        sz * (1.0 + sx * xi) * (1.0 + sy * eta),
                    ])
                derivatives = derivatives @ inverse_jacobian
                B = np.zeros((6, 24), float)
                for node, (dxn, dyn, dzn) in enumerate(derivatives):
                    start = 3 * node
                    B[0, start] = dxn
                    B[1, start + 1] = dyn
                    B[2, start + 2] = dzn
                    B[3, start] = dyn
                    B[3, start + 1] = dxn
                    B[4, start + 1] = dzn
                    B[4, start + 2] = dyn
                    B[5, start] = dzn
                    B[5, start + 2] = dxn
                result += B.T @ constitutive @ B * determinant
    return result


class StructuredQ1SolidReferenceModel:
    """Fixed-layout 3-D continuum reference with conforming shared nodes.

    The mesh spans the plate thickness and the maximum rib height.  The
    physical rib strips are represented by deterministic planform area
    fractions in the upper brick layers.  This is a verification model only:
    it has no geometry or thickness sensitivities and is not used by the
    production optimizer.
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
        ribs: Sequence[Rib],
        rib_thicknesses: Sequence[float],
        footprint_samples: int = 8,
    ) -> None:
        if len(ribs) != len(rib_thicknesses):
            raise ValueError("ribs and rib_thicknesses must have equal length")
        if nx < 1 or ny < 1:
            raise ValueError("nx and ny must be positive")
        if footprint_samples < 2:
            raise ValueError("footprint_samples must be at least two")
        self.width, self.height = float(width), float(height)
        self.nx, self.ny = int(nx), int(ny)
        self.wall_thickness = float(wall_thickness)
        self.E, self.nu = float(E), float(nu)
        self.dx, self.dy = self.width / self.nx, self.height / self.ny
        self.ribs = tuple(ribs)
        self.rib_thicknesses = tuple(float(value) for value in rib_thicknesses)
        self.footprint_samples = int(footprint_samples)
        max_height = max((float(rib.height) for rib in self.ribs), default=0.0)
        if self.wall_thickness <= 0.0 or max_height < 0.0:
            raise ValueError("wall thickness and rib height must be valid")
        if max_height > 0.0:
            self.z_coords = np.array([
                -self.wall_thickness / 2.0,
                0.0,
                self.wall_thickness / 2.0,
                self.wall_thickness / 2.0 + max_height / 2.0,
                self.wall_thickness / 2.0 + max_height,
            ])
        else:
            # Do not create zero-thickness rib layers for a no-rib reference.
            self.z_coords = np.array([
                -self.wall_thickness / 2.0, 0.0, self.wall_thickness / 2.0,
            ])
        self._grid_nx = self.nx + 1
        self._grid_ny = self.ny + 1
        self._grid_nz = len(self.z_coords)
        self._node_count = self._grid_nx * self._grid_ny * self._grid_nz
        self._element_stiffness = {
            int(layer): _hex8_stiffness(
                self.dx, self.dy,
                float(self.z_coords[layer + 1] - self.z_coords[layer]),
                self.E, self.nu,
            )
            for layer in range(self._grid_nz - 1)
        }
        self._footprints = self._build_footprints()
        self._build_system(loads, supports)

    def grid_node(self, ix: int, iy: int, iz: int) -> int:
        return (iz * self._grid_ny + iy) * self._grid_nx + ix

    def _line_distance(self, points: np.ndarray, rib: Rib) -> np.ndarray:
        p0 = np.asarray(rib.p0, float)
        delta = np.asarray(rib.p1, float) - p0
        denominator = float(delta @ delta)
        if denominator <= 1.0e-30:
            return np.linalg.norm(points - p0, axis=1)
        parameter = np.clip((points - p0) @ delta / denominator, 0.0, 1.0)
        return np.linalg.norm(points - (p0 + parameter[:, None] * delta), axis=1)

    def _build_footprints(self) -> dict[int, np.ndarray]:
        sample_count = self.footprint_samples
        offsets = (np.arange(sample_count, dtype=float) + 0.5) / sample_count
        footprints: dict[int, np.ndarray] = {}
        for layer in range(2, self._grid_nz - 1):
            z_mid = 0.5 * (self.z_coords[layer] + self.z_coords[layer + 1])
            fractions = np.zeros((self.ny, self.nx), float)
            for iy in range(self.ny):
                for ix in range(self.nx):
                    x = (ix + offsets[:, None]) * self.dx
                    y = (iy + offsets[None, :]) * self.dy
                    points = np.column_stack((np.broadcast_to(x, (sample_count, sample_count)).ravel(),
                                               np.broadcast_to(y, (sample_count, sample_count)).ravel()))
                    occupied = np.zeros(len(points), dtype=bool)
                    for rib, thickness in zip(self.ribs, self.rib_thicknesses):
                        if z_mid > self.wall_thickness / 2.0 + float(rib.height):
                            continue
                        occupied |= self._line_distance(points, rib) <= thickness / 2.0
                    fractions[iy, ix] = float(np.mean(occupied))
            footprints[layer] = fractions
        return footprints

    def _element_nodes(self, ix: int, iy: int, layer: int) -> np.ndarray:
        return np.array([
            self.grid_node(ix, iy, layer),
            self.grid_node(ix + 1, iy, layer),
            self.grid_node(ix + 1, iy + 1, layer),
            self.grid_node(ix, iy + 1, layer),
            self.grid_node(ix, iy, layer + 1),
            self.grid_node(ix + 1, iy, layer + 1),
            self.grid_node(ix + 1, iy + 1, layer + 1),
            self.grid_node(ix, iy + 1, layer + 1),
        ], dtype=int)

    def _build_system(self, loads: Sequence[dict], supports: dict) -> None:
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        active_nodes: set[int] = set()
        for layer in range(self._grid_nz - 1):
            for iy in range(self.ny):
                for ix in range(self.nx):
                    fraction = 1.0 if layer < 2 else self._footprints[layer][iy, ix]
                    if fraction <= 1.0e-12:
                        continue
                    nodes = self._element_nodes(ix, iy, layer)
                    active_nodes.update(int(node) for node in nodes)
                    dofs = np.concatenate([3 * node + np.arange(3) for node in nodes])
                    stiffness = fraction * self._element_stiffness[layer]
                    rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
                    rows.extend(rr.ravel().tolist())
                    cols.extend(cc.ravel().tolist())
                    data.extend(stiffness.ravel().tolist())
        self._active_grid_nodes = np.array(sorted(active_nodes), dtype=int)
        self._node_map = {int(node): index for index, node in enumerate(self._active_grid_nodes)}
        remapped_rows = np.empty(len(rows), dtype=int)
        remapped_cols = np.empty(len(cols), dtype=int)
        # The temporary assembly uses global grid-node IDs for readability;
        # remap each triplet to the compact active-node system before solving.
        for index, (row, col) in enumerate(zip(rows, cols)):
            remapped_rows[index] = 3 * self._node_map[row // 3] + row % 3
            remapped_cols[index] = 3 * self._node_map[col // 3] + col % 3
        self.ndof = 3 * len(self._active_grid_nodes)
        self.stiffness_matrix = coo_matrix(
            (data, (remapped_rows, remapped_cols)), shape=(self.ndof, self.ndof)
        ).tocsc()
        fixed = self._support_dofs(supports)
        self.fixed_dofs = np.array(sorted(fixed), dtype=int)
        self.free_dofs = np.setdiff1d(np.arange(self.ndof), self.fixed_dofs)
        self.load_vectors, self.load_weights = self._build_loads(loads)

    def _nearest_active_node(self, point: Sequence[float]) -> int:
        candidates = []
        for node in self._active_grid_nodes:
            iz, remainder = divmod(int(node), self._grid_nx * self._grid_ny)
            iy, ix = divmod(remainder, self._grid_nx)
            xyz = np.array([ix * self.dx, iy * self.dy, self.z_coords[iz]])
            candidates.append((float(np.linalg.norm(xyz - point)), int(node)))
        return min(candidates)[1]

    def _compact_dofs(self, grid_node: int, components: Sequence[int]) -> list[int]:
        compact = self._node_map.get(int(grid_node))
        if compact is None:
            return []
        return [3 * compact + int(component) for component in components]

    def _support_dofs(self, supports: dict) -> set[int]:
        fixed: set[int] = set()
        components = supports.get("components", [0, 1, 2])
        if supports.get("type") == "patch_springs":
            for patch in supports.get("patches", []):
                center = patch.get("center", patch.get("point"))
                size = patch["size"]
                x0 = max(0.0, float(center[0]) - float(size[0]) / 2.0)
                x1 = min(self.width, float(center[0]) + float(size[0]) / 2.0)
                y0 = max(0.0, float(center[1]) - float(size[1]) / 2.0)
                y1 = min(self.height, float(center[1]) + float(size[1]) / 2.0)
                for node in self._active_grid_nodes:
                    iz, remainder = divmod(int(node), self._grid_nx * self._grid_ny)
                    iy, ix = divmod(remainder, self._grid_nx)
                    if (
                        iz == 1 and x0 - 1.0e-12 <= ix * self.dx <= x1 + 1.0e-12
                        and y0 - 1.0e-12 <= iy * self.dy <= y1 + 1.0e-12
                    ):
                        fixed.update(self._compact_dofs(int(node), components))
            if not fixed:
                raise ValueError("physical support patches contain no active mid-plane nodes")
            first = self._nearest_active_node([0.0, 0.0, -self.wall_thickness / 2.0])
            fixed.update(self._compact_dofs(first, [1]))
            return fixed
        if supports.get("type", "points") == "points":
            points = supports.get("points", [])
            for point in points:
                node = self._nearest_active_node([point[0], point[1], 0.0])
                fixed.update(self._compact_dofs(node, components))
            # The two point supports lie on the mid-plane.  Fixing one
            # additional in-plane dof at the lower surface removes the
            # otherwise unrestrained rigid rotation about the loading axis.
            if points:
                node = self._nearest_active_node([points[0][0], points[0][1], -self.wall_thickness / 2.0])
                fixed.update(self._compact_dofs(node, [1]))
            return fixed
        if supports.get("type") == "edge" and supports.get("edge") == "right":
            for node in self._active_grid_nodes:
                iz, remainder = divmod(int(node), self._grid_nx * self._grid_ny)
                iy, ix = divmod(remainder, self._grid_nx)
                if ix == self.nx and abs(self.z_coords[iz]) <= self.wall_thickness:
                    fixed.update(self._compact_dofs(int(node), components))
            return fixed
        raise ValueError("solid reference supports currently require points or right edge")

    def _build_loads(self, loads: Sequence[dict]) -> tuple[list[np.ndarray], np.ndarray]:
        vectors: list[np.ndarray] = []
        weights: list[float] = []
        for case in loads:
            vector = np.zeros(self.ndof, float)
            for force in case["forces"]:
                value = np.asarray(force["value"], float)
                if "patch_size" not in force:
                    node = self._nearest_active_node([
                        force["point"][0], force["point"][1], 0.0,
                    ])
                    compact = self._node_map[int(node)]
                    vector[3 * compact:3 * compact + 3] += value
                    continue
                center = np.asarray(force["point"], float)
                size = np.asarray(force["patch_size"], float)
                x0 = max(0.0, center[0] - size[0] / 2.0)
                x1 = min(self.width, center[0] + size[0] / 2.0)
                y0 = max(0.0, center[1] - size[1] / 2.0)
                y1 = min(self.height, center[1] + size[1] / 2.0)
                if x1 <= x0 or y1 <= y0:
                    raise ValueError("load patch does not intersect the solid domain")
                gauss = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))
                for xi in gauss:
                    for eta in gauss:
                        x = 0.5 * (x0 + x1) + 0.5 * (x1 - x0) * xi
                        y = 0.5 * (y0 + y1) + 0.5 * (y1 - y0) * eta
                        ix = min(max(int(np.floor(x / self.dx)), 0), self.nx - 1)
                        iy = min(max(int(np.floor(y / self.dy)), 0), self.ny - 1)
                        local_x = (x - ix * self.dx) / self.dx
                        local_y = (y - iy * self.dy) / self.dy
                        plan_nodes = [
                            self.grid_node(ix, iy, 1),
                            self.grid_node(ix + 1, iy, 1),
                            self.grid_node(ix + 1, iy + 1, 1),
                            self.grid_node(ix, iy + 1, 1),
                        ]
                        shape = [
                            (1.0 - local_x) * (1.0 - local_y),
                            local_x * (1.0 - local_y),
                            local_x * local_y,
                            (1.0 - local_x) * local_y,
                        ]
                        for plan_node, weight in zip(plan_nodes, shape):
                            compact = self._node_map.get(int(plan_node))
                            if compact is not None:
                                vector[3 * compact:3 * compact + 3] += value * float(weight) / 4.0
            vectors.append(vector)
            weights.append(float(case.get("weight", 1.0)))
        return vectors, np.asarray(weights, float)

    def analyze(self) -> AnalysisResult:
        matrix = self.stiffness_matrix[self.free_dofs][:, self.free_dofs].tocsc()
        rhs = np.column_stack([load[self.free_dofs] for load in self.load_vectors])
        solution = np.asarray(splu(matrix).solve(rhs), float)
        if solution.ndim == 1:
            solution = solution[:, None]
        displacements: list[np.ndarray] = []
        compliances: list[float] = []
        for column, load in enumerate(self.load_vectors):
            displacement = np.zeros(self.ndof, float)
            displacement[self.free_dofs] = solution[:, column]
            displacements.append(displacement)
            compliances.append(float(load @ displacement))
        return AnalysisResult(float(self.load_weights @ compliances), displacements, compliances)
