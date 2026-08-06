"""Q8 shell reference model with consistent embedded line-beam ribs."""

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import numpy as np
from scipy.sparse import csc_matrix, coo_matrix

from rib_layout_env import (
    DETERMINISTIC_LINEAR_SOLVER_THREADS,
    DETERMINISTIC_SENSITIVITY_WORKERS,
    configure_deterministic_runtime,
)

from .model_shell import ShellStiffenedPlateModel
from .model_shell_beam import EmbeddedLineBeamShellReferenceModel
from .shell_q8 import shape_q8, shell_q8_stiffness


class Q8EmbeddedLineBeamShellReferenceModel(
    EmbeddedLineBeamShellReferenceModel
):
    """Response-only Q8 plate plus embedded line-beam rib reference.

    The base panel is discretized with eight-node serendipity shell elements.
    Rib strains are evaluated directly from the Q8 shell displacement and
    rotation field, so the reference has no mesh-dependent rib-root tie nodes.
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
        interface_subdivisions_per_cell: int = 2,
        rib_cache_max_entries: int = 128,
        candidate_operator_cache_max_entries: int = 512,
        rib_basis_cache_max_entries: int = 128,
        sensitivity_workers: int = 1,
        linear_solver: str = "auto",
        linear_solver_threads: int = 1,
        **_: object,
    ) -> None:
        self.width, self.height = float(width), float(height)
        self.nx, self.ny = int(nx), int(ny)
        if self.nx < 1 or self.ny < 1:
            raise ValueError("nx and ny must be positive")
        self.wall_thickness = float(wall_thickness)
        self.E, self.nu = float(E), float(nu)
        self.dx, self.dy = self.width / self.nx, self.height / self.ny
        self.interface_subdivisions_per_cell = max(
            1, int(interface_subdivisions_per_cell)
        )
        self.rib_cache_max_entries = max(1, int(rib_cache_max_entries))
        self.candidate_operator_cache_max_entries = max(
            1, int(candidate_operator_cache_max_entries)
        )
        self.rib_basis_cache_max_entries = max(1, int(rib_basis_cache_max_entries))
        if int(sensitivity_workers) != DETERMINISTIC_SENSITIVITY_WORKERS:
            raise ValueError("deterministic formal runs require sensitivity_workers=1")
        if int(linear_solver_threads) != DETERMINISTIC_LINEAR_SOLVER_THREADS:
            raise ValueError("deterministic formal runs require linear_solver_threads=1")
        self.sensitivity_workers = DETERMINISTIC_SENSITIVITY_WORKERS
        self.linear_solver = str(linear_solver).lower()
        self.linear_solver_threads = DETERMINISTIC_LINEAR_SOLVER_THREADS
        if self.linear_solver not in {"auto", "pardiso", "superlu"}:
            raise ValueError("linear_solver must be auto, pardiso, or superlu")
        self._pardiso_spsolve = None
        self._pardiso_solver = None
        if self.linear_solver in {"auto", "pardiso"}:
            configure_deterministic_runtime()
            try:
                from pypardiso import spsolve as pardiso_spsolve
                from pypardiso.scipy_aliases import pypardiso_solver
                self._pardiso_spsolve = pardiso_spsolve
                self._pardiso_solver = pypardiso_solver
            except (ImportError, OSError):
                if self.linear_solver == "pardiso":
                    raise

        # Q8 node order is a doubled integer lattice with cell centers absent:
        # corners and midside nodes are retained, interior Q4 grid centers are
        # not physical nodes.
        self._node_ids: dict[tuple[int, int], int] = {}
        self._node_xy: list[np.ndarray] = []
        for iy in range(2*self.ny + 1):
            for ix in range(2*self.nx + 1):
                if ix % 2 == 1 and iy % 2 == 1:
                    continue
                self._node_ids[(ix, iy)] = len(self._node_xy)
                self._node_xy.append(
                    np.array([0.5*ix*self.dx, 0.5*iy*self.dy], float)
                )
        self.nnode = len(self._node_xy)
        self.ndof = 6*self.nnode
        self._rib_cache: OrderedDict[tuple, csc_matrix] = OrderedDict()
        self._embedded_line_cache: OrderedDict[tuple, csc_matrix] = OrderedDict()
        self._candidate_operator_cache: OrderedDict[tuple, tuple] = OrderedDict()
        self._rib_basis_cache: OrderedDict[tuple, tuple] = OrderedDict()
        self.base_stiffness = self._build_ground_shell_q8()
        self.fixed_dofs = self._build_supports(supports)
        self.base_stiffness = (
            self.base_stiffness + self._build_support_springs(supports)
        ).tocsc()
        self.free_dofs = np.setdiff1d(np.arange(self.ndof), self.fixed_dofs)
        self.load_vectors, self.load_weights = self._build_loads(loads)

    def node(self, ix: int, iy: int) -> int:
        return self._node_ids[(int(ix), int(iy))]

    def node_xy(self, node: int) -> np.ndarray:
        return self._node_xy[int(node)].copy()

    def nearest_node(self, point: Sequence[float]) -> int:
        target = np.asarray(point, float)[:2]
        return min(
            range(self.nnode),
            key=lambda node: float(np.sum((self._node_xy[node]-target)**2)),
        )

    def _nodes_in_patch(
        self, center: Sequence[float], size: Sequence[float],
    ) -> list[int]:
        x0, x1, y0, y1 = self._patch_bounds(center, size)
        tolerance = 1.0e-12*max(self.width, self.height, 1.0)
        nodes = [
            node for node, xy in enumerate(self._node_xy)
            if x0-tolerance <= xy[0] <= x1+tolerance
            and y0-tolerance <= xy[1] <= y1+tolerance
        ]
        if not nodes:
            raise ValueError("physical patch contains no mesh nodes")
        return nodes

    def _nodes_from_selector(self, selector: dict) -> list[int]:
        selector_type = selector.get("type", "points")
        if selector_type != "edge":
            return super()._nodes_from_selector(selector)
        edge = selector["edge"]
        if edge == "right":
            return [self.node(2*self.nx, iy) for iy in range(2*self.ny+1)]
        if edge == "left":
            return [self.node(0, iy) for iy in range(2*self.ny+1)]
        if edge == "top":
            return [self.node(ix, 2*self.ny) for ix in range(2*self.nx+1)]
        if edge == "bottom":
            return [self.node(ix, 0) for ix in range(2*self.nx+1)]
        raise ValueError(f"unknown edge {edge}")

    def interpolation(self, point: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        x = float(np.clip(point[0], 0.0, self.width))
        y = float(np.clip(point[1], 0.0, self.height))
        ix = min(int(x/self.dx), self.nx-1)
        iy = min(int(y/self.dy), self.ny-1)
        xi = 2.0*(x-ix*self.dx)/self.dx-1.0
        eta = 2.0*(y-iy*self.dy)/self.dy-1.0
        nodes = np.array([
            self.node(2*ix, 2*iy), self.node(2*ix+2, 2*iy),
            self.node(2*ix+2, 2*iy+2), self.node(2*ix, 2*iy+2),
            self.node(2*ix+1, 2*iy), self.node(2*ix+2, 2*iy+1),
            self.node(2*ix+1, 2*iy+2), self.node(2*ix, 2*iy+1),
        ], int)
        weights, _ = shape_q8(xi, eta)
        return nodes, weights

    def _interpolation_with_line_derivative(
        self, point: np.ndarray, tangent: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = float(np.clip(point[0], 0.0, self.width))
        y = float(np.clip(point[1], 0.0, self.height))
        ix = min(int(x/self.dx), self.nx-1)
        iy = min(int(y/self.dy), self.ny-1)
        xi = 2.0*(x-ix*self.dx)/self.dx-1.0
        eta = 2.0*(y-iy*self.dy)/self.dy-1.0
        nodes, shape = self.interpolation((x, y))
        _, dn_nat = shape_q8(xi, eta)
        dshape_xy = dn_nat @ np.diag([2.0/self.dx, 2.0/self.dy])
        return nodes, shape, dshape_xy @ np.asarray(tangent, float)

    def _build_ground_shell_q8(self) -> csc_matrix:
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for iy in range(self.ny):
            for ix in range(self.nx):
                nodes = np.array([
                    self.node(2*ix, 2*iy), self.node(2*ix+2, 2*iy),
                    self.node(2*ix+2, 2*iy+2), self.node(2*ix, 2*iy+2),
                    self.node(2*ix+1, 2*iy), self.node(2*ix+2, 2*iy+1),
                    self.node(2*ix+1, 2*iy+2), self.node(2*ix, 2*iy+1),
                ], int)
                xyz = np.array([[*self.node_xy(node), 0.0] for node in nodes])
                stiffness = shell_q8_stiffness(
                    xyz, self.wall_thickness, self.E, self.nu,
                )
                dofs = np.concatenate([
                    np.arange(6*int(node), 6*int(node)+6) for node in nodes
                ])
                self._append_dense(rows, cols, data, dofs, stiffness)
        return coo_matrix(
            (data, (rows, cols)), shape=(self.ndof, self.ndof),
        ).tocsc()
