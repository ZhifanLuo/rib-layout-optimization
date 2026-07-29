"""Internal ground-shell plus vertical rib-shell structural model."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Sequence
import warnings

import numpy as np
from scipy.linalg import LinAlgWarning, solve
from scipy.sparse import csc_matrix, coo_matrix
from scipy.sparse.linalg import splu

from .model import AnalysisResult, Rib
from rib_layout_env import (
    DETERMINISTIC_LINEAR_SOLVER_THREADS,
    DETERMINISTIC_SENSITIVITY_WORKERS,
    configure_deterministic_runtime,
)
from .shell import shell_q4_stiffness, shell_q4_stiffness_components


class ShellStiffenedPlateModel:
    """Six-DOF Q4 shell wall with vertical Q4 shell-strip ribs.

    Every rib strip has a free top edge.  Its top degrees of freedom are
    statically condensed, and its bottom edge is bilinearly mapped to the
    ground-shell grid.  Different ribs are intentionally not tied to each
    other at crossings, as requested.
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
        self.wall_thickness = float(wall_thickness)
        self.E, self.nu = float(E), float(nu)
        self.dx, self.dy = self.width / self.nx, self.height / self.ny
        self.interface_subdivisions_per_cell = max(1, int(interface_subdivisions_per_cell))
        self.rib_cache_max_entries = max(1, int(rib_cache_max_entries))
        self.candidate_operator_cache_max_entries = max(
            1, int(candidate_operator_cache_max_entries)
        )
        self.rib_basis_cache_max_entries = max(1, int(rib_basis_cache_max_entries))
        requested_sensitivity_workers = int(sensitivity_workers)
        requested_solver_threads = int(linear_solver_threads)
        if requested_sensitivity_workers != DETERMINISTIC_SENSITIVITY_WORKERS:
            raise ValueError(
                "deterministic formal runs require sensitivity_workers=1"
            )
        if requested_solver_threads != DETERMINISTIC_LINEAR_SOLVER_THREADS:
            raise ValueError(
                "deterministic formal runs require linear_solver_threads=1"
            )
        self.sensitivity_workers = DETERMINISTIC_SENSITIVITY_WORKERS
        self.linear_solver = str(linear_solver).lower()
        self.linear_solver_threads = DETERMINISTIC_LINEAR_SOLVER_THREADS
        if self.linear_solver not in {"auto", "pardiso", "superlu"}:
            raise ValueError("linear_solver must be auto, pardiso, or superlu")
        self._pardiso_spsolve = None
        self._pardiso_solver = None
        if self.linear_solver in {"auto", "pardiso"}:
            # Repeated optimization paths are highly sensitive to tiny FEA
            # differences. Pin both OpenMP and MKL before importing Pardiso;
            # one configured thread provides deterministic reduction order.
            configure_deterministic_runtime()
            try:
                from pypardiso import spsolve as pardiso_spsolve
                from pypardiso.scipy_aliases import pypardiso_solver
                self._pardiso_spsolve = pardiso_spsolve
                self._pardiso_solver = pypardiso_solver
            except (ImportError, OSError):
                if self.linear_solver == "pardiso":
                    raise
        self.nnode = (self.nx + 1) * (self.ny + 1)
        self.ndof = 6 * self.nnode
        # Geometry and rationalization evaluate thousands of perturbed ribs.
        # Keeping every condensed matrix made the formal 80x40 examples grow
        # beyond 10 GB and eventually starve SuperLU.  An LRU retains the
        # matrices reused within the current sensitivity/verification cycle
        # while placing a strict bound on the long-running FEA memory footprint.
        self._rib_cache: OrderedDict[tuple, csc_matrix] = OrderedDict()
        self._candidate_operator_cache: OrderedDict[tuple, tuple] = OrderedDict()
        self._candidate_operator_cache_lock = Lock()
        self._rib_basis_cache: OrderedDict[tuple, tuple] = OrderedDict()
        self.base_stiffness = self._build_ground_shell()
        self.fixed_dofs = self._build_supports(supports)
        self.free_dofs = np.setdiff1d(np.arange(self.ndof), self.fixed_dofs)
        self.load_vectors, self.load_weights = self._build_loads(loads)

    @staticmethod
    def _rib_numeric_geometry_key(rib: Rib) -> tuple:
        """Return an exact, orientation-preserving key for numeric caches."""
        return (
            tuple(float(value) for value in rib.p0),
            tuple(float(value) for value in rib.p1),
            float(rib.height),
            int(rib.segments),
        )

    @classmethod
    def _rib_numeric_state_key(cls, rib: Rib, thickness: float) -> tuple:
        """Return the exact geometry-and-thickness numeric cache key."""
        return cls._rib_numeric_geometry_key(rib), float(thickness)

    def node(self, ix: int, iy: int) -> int:
        return iy * (self.nx + 1) + ix

    def node_xy(self, node: int) -> np.ndarray:
        iy, ix = divmod(int(node), self.nx + 1)
        return np.array([ix * self.dx, iy * self.dy], float)

    def nearest_node(self, point: Sequence[float]) -> int:
        x, y = point
        ix = int(np.clip(round(x / self.dx), 0, self.nx))
        iy = int(np.clip(round(y / self.dy), 0, self.ny))
        return self.node(ix, iy)

    @staticmethod
    def _append_dense(rows: list[int], cols: list[int], data: list[float], dofs: np.ndarray, k: np.ndarray) -> None:
        rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
        mask = np.abs(k) > 1.0e-18
        rows.extend(rr[mask].ravel().tolist()); cols.extend(cc[mask].ravel().tolist()); data.extend(k[mask].ravel().tolist())

    @staticmethod
    def _factorize_spd(matrix: csc_matrix):
        """Use SuperLU's symmetric path, with a robust general-LU fallback."""
        try:
            return splu(
                matrix,
                diag_pivot_thresh=0.0,
                options={"SymmetricMode": True},
            )
        except RuntimeError:
            return splu(matrix)

    def _solve_global(self, matrix: csc_matrix, rhs: np.ndarray) -> np.ndarray:
        """Solve the global FEA system and always release Pardiso afterward."""
        try:
            if self._pardiso_spsolve is not None:
                try:
                    solution = np.asarray(
                        self._pardiso_spsolve(matrix.tocsr(), rhs), float
                    )
                    if rhs.ndim == 2 and solution.ndim == 1:
                        solution = solution[:, None]
                    elif rhs.ndim == 1 and solution.ndim == 2:
                        solution = solution[:, 0]
                    residual = matrix@solution-rhs
                    relative_residual = float(
                        np.linalg.norm(residual)/max(np.linalg.norm(rhs), 1.0e-30)
                    )
                    if np.all(np.isfinite(solution)) and relative_residual <= 1.0e-8:
                        return solution
                except (RuntimeError, ValueError, np.linalg.LinAlgError):
                    if self.linear_solver == "pardiso":
                        raise
            solver = self._factorize_spd(matrix)
            return np.asarray(solver.solve(rhs), float)
        finally:
            # pypardiso's module-level spsolve retains the most recent MKL
            # factorization. Global FEA matrices change between evaluations,
            # so release it at the single solver boundary on every exit path:
            # Pardiso success, SuperLU fallback, or an exception from either.
            if self._pardiso_solver is not None:
                try:
                    self._pardiso_solver.free_memory()
                except (RuntimeError, ValueError):
                    if self.linear_solver == "pardiso":
                        raise

    def _build_ground_shell(self) -> csc_matrix:
        rows: list[int] = []; cols: list[int] = []; data: list[float] = []
        for iy in range(self.ny):
            for ix in range(self.nx):
                nodes = np.array([self.node(ix, iy), self.node(ix+1, iy), self.node(ix+1, iy+1), self.node(ix, iy+1)], int)
                xyz = np.array([[*self.node_xy(n), 0.0] for n in nodes])
                k = shell_q4_stiffness(xyz, self.wall_thickness, self.E, self.nu)
                dofs = np.concatenate([np.arange(6*n, 6*n+6) for n in nodes])
                self._append_dense(rows, cols, data, dofs, k)
        return coo_matrix((data, (rows, cols)), shape=(self.ndof, self.ndof)).tocsc()

    def _nodes_from_selector(self, selector: dict) -> list[int]:
        if selector.get("type", "points") == "points":
            return sorted({self.nearest_node(p) for p in selector["points"]})
        edge = selector["edge"]
        if edge == "right": return [self.node(self.nx, iy) for iy in range(self.ny+1)]
        if edge == "left": return [self.node(0, iy) for iy in range(self.ny+1)]
        if edge == "top": return [self.node(ix, self.ny) for ix in range(self.nx+1)]
        if edge == "bottom": return [self.node(ix, 0) for ix in range(self.nx+1)]
        raise ValueError(f"unknown edge {edge}")

    def _build_supports(self, supports: dict) -> np.ndarray:
        nodes = self._nodes_from_selector(supports)
        components = supports.get("components", [0,1,2,3,4,5])
        return np.array(sorted(6*n+int(c) for n in nodes for c in components), int)

    def _build_loads(self, cases: Sequence[dict]) -> tuple[list[np.ndarray], np.ndarray]:
        vectors: list[np.ndarray] = []; weights: list[float] = []
        for case in cases:
            f = np.zeros(self.ndof)
            for item in case["forces"]:
                n = self.nearest_node(item["point"])
                f[6*n:6*n+3] += np.asarray(item["value"], float)
            vectors.append(f); weights.append(float(case.get("weight", 1.0)))
        w = np.asarray(weights, float); w /= w.sum()
        return vectors, w

    def interpolation(self, point: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        x = float(np.clip(point[0], 0, self.width)); y = float(np.clip(point[1], 0, self.height))
        ix = min(int(x / self.dx), self.nx-1); iy = min(int(y / self.dy), self.ny-1)
        xi = (x-ix*self.dx)/self.dx; eta = (y-iy*self.dy)/self.dy
        nodes = np.array([self.node(ix,iy),self.node(ix+1,iy),self.node(ix+1,iy+1),self.node(ix,iy+1)],int)
        weights = np.array([(1-xi)*(1-eta),xi*(1-eta),xi*eta,(1-xi)*eta])
        return nodes, weights

    def rib_bottom_points(self, rib: Rib) -> np.ndarray:
        """Return a ground-grid-conforming discretization of a rib root.

        The configured ``rib.segments`` remains a minimum discretization. All
        intersections with ground-shell grid lines are inserted, then every
        resulting interval is subdivided further. Consequently no rib element
        crosses a ground-element boundary without an interface node, and the
        tied slave trace closely follows the bilinear ground-shell trace.
        """
        p0, p1 = np.asarray(rib.p0, float), np.asarray(rib.p1, float)
        delta = p1 - p0
        length = max(float(np.linalg.norm(delta)), 1.0e-14)
        minimum_refined_length = 1.0e-3 * min(self.dx, self.dy)
        merge_tolerance = max(
            1.0e-5 * min(self.dx, self.dy) / length,
            self.interface_subdivisions_per_cell * minimum_refined_length / length,
        )
        base_parameters = list(np.linspace(0.0, 1.0, int(rib.segments) + 1))
        grid_parameters: list[float] = []
        if abs(delta[0]) > 1.0e-14:
            for ix in range(1, self.nx):
                value = (ix * self.dx - p0[0]) / delta[0]
                if 1.0e-12 < value < 1.0 - 1.0e-12:
                    grid_parameters.append(float(value))
        if abs(delta[1]) > 1.0e-14:
            for iy in range(1, self.ny):
                value = (iy * self.dy - p0[1]) / delta[1]
                if 1.0e-12 < value < 1.0 - 1.0e-12:
                    grid_parameters.append(float(value))
        # Grid intersections take priority over nearby nominal segment points.
        # Intersections within a tiny physical distance of a rib endpoint are
        # merged into that endpoint to avoid nearly zero-length shell slivers.
        grid_parameters = [
            value for value in grid_parameters
            if value > merge_tolerance and 1.0-value > merge_tolerance
        ]
        parameters = [0.0, 1.0, *grid_parameters]
        parameters.extend(
            value for value in base_parameters[1:-1]
            if all(abs(value-grid) > merge_tolerance for grid in grid_parameters)
        )
        ordered = sorted(parameters)
        merged = [ordered[0]]
        for value in ordered[1:]:
            if value-merged[-1] > merge_tolerance:
                merged.append(value)
        if 1.0-merged[-1] <= merge_tolerance:
            merged[-1] = 1.0
        elif merged[-1] < 1.0:
            merged.append(1.0)
        breaks = np.asarray(merged, float)
        refined = [float(breaks[0])]
        subdivisions = self.interface_subdivisions_per_cell
        for left, right in zip(breaks[:-1], breaks[1:]):
            refined.extend(float(left + (right-left)*j/subdivisions) for j in range(1, subdivisions + 1))
        s = np.asarray(refined, float)
        return (1.0-s[:, None])*p0 + s[:, None]*p1

    def _rib_bottom_points_and_shape_derivatives(
        self,
        rib: Rib,
        side: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the root trace and piecewise-analytic endpoint derivatives.

        ``side`` selects the positive/negative directional trace at coincident
        x/y-grid intersections.  Zero returns their symmetric generalized
        derivative.
        """
        p0 = np.asarray(rib.p0, float)
        p1 = np.asarray(rib.p1, float)
        delta = p1-p0
        length = max(float(np.linalg.norm(delta)), 1.0e-14)
        minimum_refined_length = 1.0e-3*min(self.dx, self.dy)
        merge_tolerance = max(
            1.0e-5*min(self.dx, self.dy)/length,
            self.interface_subdivisions_per_cell*minimum_refined_length/length,
        )
        zero = np.zeros(4, float)
        grid_parameters: list[tuple[float, np.ndarray]] = []
        if abs(delta[0]) > 1.0e-14:
            for ix in range(1, self.nx):
                value = (ix*self.dx-p0[0])/delta[0]
                if 1.0e-12 < value < 1.0-1.0e-12:
                    derivative = np.zeros(4, float)
                    derivative[0] = -(1.0-value)/delta[0]
                    derivative[2] = -value/delta[0]
                    grid_parameters.append((float(value), derivative))
        if abs(delta[1]) > 1.0e-14:
            for iy in range(1, self.ny):
                value = (iy*self.dy-p0[1])/delta[1]
                if 1.0e-12 < value < 1.0-1.0e-12:
                    derivative = np.zeros(4, float)
                    derivative[1] = -(1.0-value)/delta[1]
                    derivative[3] = -value/delta[1]
                    grid_parameters.append((float(value), derivative))
        grid_parameters = [
            item for item in grid_parameters
            if item[0] > merge_tolerance and 1.0-item[0] > merge_tolerance
        ]
        entries: list[tuple[float, np.ndarray]] = [
            (0.0, zero.copy()),
            (1.0, zero.copy()),
            *grid_parameters,
        ]
        for value in np.linspace(0.0, 1.0, int(rib.segments)+1)[1:-1]:
            if all(abs(float(value)-grid) > merge_tolerance for grid, _ in grid_parameters):
                entries.append((float(value), zero.copy()))
        ordered = sorted(entries, key=lambda item: item[0])
        merged = [ordered[0]]
        for value, derivative in ordered[1:]:
            if value-merged[-1][0] > merge_tolerance:
                merged.append((value, derivative))
            elif abs(value-merged[-1][0]) <= 1.0e-12:
                # At a ground-grid node the x- and y-grid intersections have
                # the same parameter but exchange order under opposite
                # perturbations.  Their mean is the symmetric generalized
                # derivative of this mesh-induced kink and avoids biasing the
                # optimizer toward either side of a grid line.
                previous = merged[-1][1]
                if side > 0:
                    selected = np.minimum(previous, derivative)
                elif side < 0:
                    selected = np.maximum(previous, derivative)
                else:
                    selected = 0.5*(previous+derivative)
                merged[-1] = (merged[-1][0], selected)
        if 1.0-merged[-1][0] <= merge_tolerance:
            merged[-1] = (1.0, zero.copy())
        elif merged[-1][0] < 1.0:
            merged.append((1.0, zero.copy()))

        refined: list[tuple[float, np.ndarray]] = [merged[0]]
        subdivisions = self.interface_subdivisions_per_cell
        for (left, dleft), (right, dright) in zip(merged[:-1], merged[1:]):
            for subdivision in range(1, subdivisions+1):
                fraction = subdivision/subdivisions
                refined.append((
                    float(left+(right-left)*fraction),
                    dleft+(dright-dleft)*fraction,
                ))
        parameters = np.asarray([item[0] for item in refined], float)
        parameter_derivatives = np.asarray([item[1] for item in refined], float)
        bottom = (1.0-parameters[:,None])*p0+parameters[:,None]*p1
        derivatives = np.zeros((len(bottom), 2, 4), float)
        for coordinate in range(4):
            endpoint = coordinate//2
            axis = coordinate%2
            derivatives[:, :, coordinate] = parameter_derivatives[:, coordinate, None]*delta
            derivatives[:, axis, coordinate] += (
                1.0-parameters if endpoint == 0 else parameters
            )
        return bottom, derivatives

    def _rib_condensed(self, rib: Rib, thickness: float) -> tuple[np.ndarray, np.ndarray]:
        bottom = self.rib_bottom_points(rib)
        nseg = len(bottom) - 1
        xyz = np.vstack((np.c_[bottom,np.zeros(nseg+1)], np.c_[bottom,np.full(nseg+1,rib.height)]))
        nlocal = 2*(nseg+1); K = np.zeros((6*nlocal,6*nlocal))
        for s in range(nseg):
            nodes = np.array([s,s+1,nseg+1+s+1,nseg+1+s])
            ke = shell_q4_stiffness(xyz[nodes], thickness, self.E, self.nu)
            dofs = np.concatenate([np.arange(6*n,6*n+6) for n in nodes])
            K[np.ix_(dofs,dofs)] += ke
        retained = np.arange(6*(nseg+1))
        internal = np.arange(6*(nseg+1),6*nlocal)
        Krr = K[np.ix_(retained,retained)]; Kri = K[np.ix_(retained,internal)]
        Kir = Kri.T; Kii = K[np.ix_(internal,internal)]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", LinAlgWarning)
                recovery = solve(Kii, Kir, assume_a="pos")
        except (LinAlgWarning, np.linalg.LinAlgError) as exc:
            raise np.linalg.LinAlgError("ill-conditioned condensed rib shell") from exc
        condensed = Krr - Kri @ recovery
        return bottom, 0.5*(condensed+condensed.T)

    def _rib_sparse_basis(
        self,
        rib: Rib,
    ) -> tuple[np.ndarray, csc_matrix, csc_matrix]:
        """Assemble thickness-linear/cubic sparse matrices for one rib."""
        key = self._rib_numeric_geometry_key(rib)
        if key in self._rib_basis_cache:
            value = self._rib_basis_cache.pop(key)
            self._rib_basis_cache[key] = value
            return value
        bottom = self.rib_bottom_points(rib)
        nseg = len(bottom)-1
        xyz = np.vstack((
            np.c_[bottom, np.zeros(nseg+1)],
            np.c_[bottom, np.full(nseg+1, rib.height)],
        ))
        nlocal = 2*(nseg+1)
        linear_rows: list[int] = []
        linear_cols: list[int] = []
        linear_data: list[float] = []
        cubic_rows: list[int] = []
        cubic_cols: list[int] = []
        cubic_data: list[float] = []
        element_components: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}
        for s in range(nseg):
            nodes = np.array([s, s+1, nseg+1+s+1, nseg+1+s])
            delta = bottom[s+1]-bottom[s]
            element_key = (round(float(delta[0]), 12), round(float(delta[1]), 12))
            components = element_components.get(element_key)
            if components is None:
                components = shell_q4_stiffness_components(
                    xyz[nodes], self.E, self.nu
                )
                element_components[element_key] = components
            linear, cubic = components
            dofs = np.concatenate([np.arange(6*n, 6*n+6) for n in nodes])
            self._append_dense(
                linear_rows, linear_cols, linear_data, dofs, linear
            )
            self._append_dense(
                cubic_rows, cubic_cols, cubic_data, dofs, cubic
            )
        shape = (6*nlocal, 6*nlocal)
        linear_matrix = coo_matrix(
            (linear_data, (linear_rows, linear_cols)), shape=shape
        ).tocsc()
        cubic_matrix = coo_matrix(
            (cubic_data, (cubic_rows, cubic_cols)), shape=shape
        ).tocsc()
        value = (bottom, linear_matrix, cubic_matrix)
        self._rib_basis_cache[key] = value
        while len(self._rib_basis_cache) > self.rib_basis_cache_max_entries:
            self._rib_basis_cache.popitem(last=False)
        return value

    def _rib_sparse(
        self,
        rib: Rib,
        thickness: float,
        cache_basis: bool = False,
    ) -> tuple[np.ndarray, csc_matrix]:
        """Assemble one rib strip without forming its dense Schur complement."""
        if cache_basis:
            bottom, linear, cubic = self._rib_sparse_basis(rib)
            t = float(thickness)
            return bottom, (t*linear+t**3*cubic).tocsc()
        bottom = self.rib_bottom_points(rib)
        nseg = len(bottom)-1
        xyz = np.vstack((
            np.c_[bottom, np.zeros(nseg+1)],
            np.c_[bottom, np.full(nseg+1, rib.height)],
        ))
        nlocal = 2*(nseg+1)
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        element_matrices: dict[tuple[float, float], np.ndarray] = {}
        for s in range(nseg):
            nodes = np.array([s, s+1, nseg+1+s+1, nseg+1+s])
            delta = bottom[s+1]-bottom[s]
            element_key = (round(float(delta[0]), 12), round(float(delta[1]), 12))
            ke = element_matrices.get(element_key)
            if ke is None:
                ke = shell_q4_stiffness(
                    xyz[nodes], thickness, self.E, self.nu
                )
                element_matrices[element_key] = ke
            dofs = np.concatenate([np.arange(6*n, 6*n+6) for n in nodes])
            self._append_dense(rows, cols, data, dofs, ke)
        matrix = coo_matrix(
            (data, (rows, cols)), shape=(6*nlocal, 6*nlocal)
        ).tocsc()
        return bottom, matrix

    def _rib_energy_direct(
        self,
        rib: Rib,
        thickness: float,
        result: AnalysisResult,
        cache_operator: bool = False,
    ) -> float:
        """Evaluate condensed rib energy with a sparse one-RHS local solve.

        Sensitivities only require ``u.T @ K_condensed @ u``. Forming the full
        dense Schur complement, mapping it to all ground DOFs, and then taking
        that scalar is needlessly expensive for long ribs. This routine gives
        the identical scalar by equilibrating the free top-edge DOFs directly.
        """
        key = self._rib_numeric_state_key(rib, thickness)
        if cache_operator:
            with self._candidate_operator_cache_lock:
                cached = self._candidate_operator_cache.get(key)
                if cached is not None:
                    self._candidate_operator_cache.move_to_end(key)
        else:
            cached = None
        if cached is None:
            bottom, matrix = self._rib_sparse(rib, float(thickness))
            retained_count = 6*len(bottom)
            internal = np.arange(retained_count, matrix.shape[0])
            retained = np.arange(retained_count)
            kii = matrix[internal][:, internal].tocsc()
            kir = matrix[internal][:, retained].tocsc()
            solver = self._factorize_spd(kii)
            if cache_operator:
                with self._candidate_operator_cache_lock:
                    self._candidate_operator_cache[key] = (
                        bottom, matrix, kir, solver
                    )
                    while (
                        len(self._candidate_operator_cache)
                        > self.candidate_operator_cache_max_entries
                    ):
                        self._candidate_operator_cache.popitem(last=False)
        else:
            bottom, matrix, kir, solver = cached
        retained_displacements = np.column_stack([
            np.concatenate([self.response_at(u, point) for point in bottom])
            for u in result.displacements
        ])
        internal_displacements = -solver.solve(kir @ retained_displacements)
        local_displacements = np.vstack((
            retained_displacements,
            internal_displacements,
        ))
        energies = np.sum(local_displacements*(matrix @ local_displacements), axis=0)
        return float(self.load_weights @ energies)

    def _equilibrated_local_displacements(
        self,
        bottom: np.ndarray,
        matrix: csc_matrix,
        result: AnalysisResult,
    ) -> np.ndarray:
        """Return prescribed bottom plus equilibrated free-top rib DOFs."""
        retained_count = 6*len(bottom)
        internal = np.arange(retained_count, matrix.shape[0])
        retained = np.arange(retained_count)
        kii = matrix[internal][:,internal].tocsc()
        kir = matrix[internal][:,retained].tocsc()
        retained_displacements = np.column_stack([
            np.concatenate([self.response_at(u,point) for point in bottom])
            for u in result.displacements
        ])
        internal_displacements = -self._factorize_spd(kii).solve(
            kir@retained_displacements
        )
        return np.vstack((retained_displacements,internal_displacements))

    @staticmethod
    def _rib_parameters(rib: Rib, points: np.ndarray) -> np.ndarray:
        p0=np.asarray(rib.p0,float); delta=np.asarray(rib.p1,float)-p0
        denominator=max(float(delta@delta),1.0e-30)
        return np.clip((np.asarray(points,float)-p0)@delta/denominator,0.0,1.0)

    def _energy_with_interpolated_internal_field(
        self,
        rib: Rib,
        thickness: float,
        result: AnalysisResult,
        source_parameters: np.ndarray,
        source_internal: np.ndarray,
    ) -> float:
        """Evaluate rib energy with a fixed, parametrically interpolated top field."""
        bottom,matrix=self._rib_sparse(rib,float(thickness))
        target_parameters=self._rib_parameters(rib,bottom)
        nload=source_internal.shape[1]
        source_nodes=source_internal.reshape(len(source_parameters),6,nload)
        target_internal=np.empty((len(target_parameters),6,nload))
        for component in range(6):
            for load in range(nload):
                target_internal[:,component,load]=np.interp(
                    target_parameters,
                    source_parameters,
                    source_nodes[:,component,load],
                )
        retained=np.column_stack([
            np.concatenate([self.response_at(u,point) for point in bottom])
            for u in result.displacements
        ])
        local=np.vstack((retained,target_internal.reshape(-1,nload)))
        energies=np.sum(local*(matrix@local),axis=0)
        return float(self.load_weights@energies)

    def _expanded_stiffness(
        self,
        ribs: Sequence[Rib],
        thicknesses: Sequence[float],
    ) -> tuple[csc_matrix, np.ndarray]:
        """Assemble a sparse tied model while retaining rib top-edge DOFs.

        Static condensation is mathematically convenient but turns every long
        rib into a dense coupling between all ground nodes along its root. The
        expanded system is exactly equivalent: bottom traces are tied through
        the same bilinear interpolation, while each rib top edge keeps its own
        internal DOFs. Its element-band sparsity makes the formal examples much
        faster and avoids the large SuperLU memory peak.
        """
        local_data = [
            self._rib_sparse(rib, float(thickness), cache_basis=True)
            for rib, thickness in zip(ribs, thicknesses)
        ]
        top_dof_count = sum(6*len(bottom) for bottom, _ in local_data)
        total_dofs = self.ndof+top_dof_count

        base = self.base_stiffness.tocoo()
        row_chunks = [base.row]
        col_chunks = [base.col]
        data_chunks = [base.data]
        top_offset = self.ndof
        for (bottom, local), rib in zip(local_data, ribs):
            retained_count = 6*len(bottom)
            transform_rows: list[int] = []
            transform_cols: list[int] = []
            transform_data: list[float] = []
            for point_index, point in enumerate(bottom):
                nodes, weights = self.interpolation(point)
                for node, weight in zip(nodes, weights):
                    for component in range(6):
                        transform_rows.append(6*point_index+component)
                        transform_cols.append(6*int(node)+component)
                        transform_data.append(float(weight))
            for point_index in range(len(bottom)):
                for component in range(6):
                    transform_rows.append(retained_count+6*point_index+component)
                    transform_cols.append(top_offset+6*point_index+component)
                    transform_data.append(1.0)
            transform = coo_matrix(
                (transform_data, (transform_rows, transform_cols)),
                shape=(local.shape[0], total_dofs),
            ).tocsc()
            contribution = (transform.T @ local @ transform).tocoo()
            row_chunks.append(contribution.row)
            col_chunks.append(contribution.col)
            data_chunks.append(contribution.data)
            top_offset += retained_count

        matrix = coo_matrix(
            (
                np.concatenate(data_chunks),
                (np.concatenate(row_chunks), np.concatenate(col_chunks)),
            ),
            shape=(total_dofs, total_dofs),
        ).tocsc()
        free_dofs = np.concatenate((
            self.free_dofs,
            np.arange(self.ndof, total_dofs, dtype=int),
        ))
        return matrix, free_dofs

    def rib_stiffness(self, rib: Rib, thickness: float) -> csc_matrix:
        key = self._rib_numeric_state_key(rib, thickness)
        if key in self._rib_cache:
            matrix = self._rib_cache.pop(key)
            self._rib_cache[key] = matrix
            return matrix
        bottom, condensed = self._rib_condensed(rib, float(thickness))
        interp = [self.interpolation(p) for p in bottom]
        ground_nodes = np.unique(np.concatenate([x[0] for x in interp]))
        lookup = {int(n):i for i,n in enumerate(ground_nodes)}
        B = np.zeros((6*len(bottom),6*len(ground_nodes)))
        for i,(nodes,weights) in enumerate(interp):
            for node,w in zip(nodes,weights):
                j=6*lookup[int(node)]; B[6*i:6*i+6,j:j+6] += float(w)*np.eye(6)
        kg = B.T @ condensed @ B
        dofs = np.concatenate([np.arange(6*n,6*n+6) for n in ground_nodes])
        rr,cc=np.meshgrid(dofs,dofs,indexing="ij"); mask=np.abs(kg)>1e-18
        matrix=coo_matrix((kg[mask],(rr[mask],cc[mask])),shape=(self.ndof,self.ndof)).tocsc()
        self._rib_cache[key]=matrix
        while len(self._rib_cache) > self.rib_cache_max_entries:
            self._rib_cache.popitem(last=False)
        return matrix

    def stiffness(self, ribs: Sequence[Rib], thicknesses: Sequence[float]) -> csc_matrix:
        K=self.base_stiffness.copy()
        for rib,t in zip(ribs,thicknesses): K=K+self.rib_stiffness(rib,float(t))
        return K

    def analyze(self, ribs: Sequence[Rib], thicknesses: Sequence[float]) -> AnalysisResult:
        K,free_dofs=self._expanded_stiffness(ribs,thicknesses)
        Kff=K[free_dofs][:,free_dofs].tocsc()
        right_hand_sides=np.zeros((len(free_dofs),len(self.load_vectors)))
        expanded_loads=[]
        for column,f in enumerate(self.load_vectors):
            expanded_f=np.zeros(K.shape[0]); expanded_f[:self.ndof]=f
            expanded_loads.append(expanded_f)
            right_hand_sides[:,column]=expanded_f[free_dofs]
        free_solutions=self._solve_global(Kff,right_hand_sides)
        if free_solutions.ndim == 1:
            free_solutions=free_solutions[:,None]
        us=[]; cs=[]
        for column,(f,expanded_f) in enumerate(zip(self.load_vectors,expanded_loads)):
            expanded_u=np.zeros(K.shape[0]); expanded_u[free_dofs]=free_solutions[:,column]
            u=expanded_u[:self.ndof]; us.append(u); cs.append(float(f@u))
        return AnalysisResult(float(self.load_weights@cs),us,cs)

    def _energy(self, matrix: csc_matrix, result: AnalysisResult) -> float:
        return float(sum(w*(u@(matrix@u)) for w,u in zip(self.load_weights,result.displacements)))

    def candidate_efficiency(self, rib: Rib, thickness: float, result: AnalysisResult) -> float:
        """Return frozen-field stiffness contribution per added rib volume."""
        volume = rib.length * rib.height * float(thickness)
        if volume <= 0.0:
            return -np.inf
        return float(self.candidate_stiffness_energy(rib, thickness, result)/volume)

    def candidate_stiffness_energy(
        self, rib: Rib, thickness: float, result: AnalysisResult
    ) -> float:
        """Return the candidate's weighted frozen-field stiffness energy."""
        return self._rib_energy_direct(
            rib, float(thickness), result, cache_operator=True
        )

    def compliance_gradient(self, ribs: Sequence[Rib], thicknesses: Sequence[float], result: AnalysisResult) -> np.ndarray:
        def component(item: tuple[int, Rib, float]) -> tuple[int, float]:
            i, rib, t = item
            bottom,linear,cubic=self._rib_sparse_basis(rib)
            matrix=(t*linear+t**3*cubic).tocsc()
            local=self._equilibrated_local_displacements(bottom,matrix,result)
            derivative=linear+3.0*t**2*cubic
            energies=np.sum(local*(derivative@local),axis=0)
            return i,-float(self.load_weights@energies)

        items = [
            (i, rib, float(t))
            for i, (rib, t) in enumerate(zip(ribs, thicknesses))
        ]
        if self.sensitivity_workers > 1 and len(items) > 1:
            with ThreadPoolExecutor(max_workers=self.sensitivity_workers) as pool:
                values = list(pool.map(component, items))
        else:
            values = [component(item) for item in items]
        gradient=np.zeros(len(ribs))
        for i, value in values:
            gradient[i] = value
        return gradient

    def _rib_element_energies_from_bottom_points(
        self,
        bottom: np.ndarray,
        height: float,
        thickness: float,
        local_displacements: np.ndarray,
    ) -> np.ndarray:
        """Evaluate load-case rib energies without assembling a sparse matrix."""
        bottom = np.asarray(bottom)
        nseg = len(bottom)-1
        xyz = np.vstack((
            np.c_[bottom, np.zeros(nseg+1, dtype=bottom.dtype)],
            np.c_[bottom, np.full(nseg+1, height, dtype=bottom.dtype)],
        ))
        energies = np.zeros(local_displacements.shape[1], dtype=bottom.dtype)
        element_matrices: dict[tuple[float, float, float, float], np.ndarray] = {}
        for segment in range(nseg):
            nodes = np.array([
                segment,
                segment+1,
                nseg+1+segment+1,
                nseg+1+segment,
            ])
            element_dofs = np.concatenate([
                np.arange(6*node, 6*node+6) for node in nodes
            ])
            element_displacements = local_displacements[element_dofs, :]
            delta = bottom[segment+1]-bottom[segment]
            element_key = (
                round(float(np.real(delta[0])), 12),
                round(float(np.imag(delta[0])), 18),
                round(float(np.real(delta[1])), 12),
                round(float(np.imag(delta[1])), 18),
            )
            stiffness = element_matrices.get(element_key)
            if stiffness is None:
                stiffness = shell_q4_stiffness(
                    xyz[nodes], float(thickness), self.E, self.nu
                )
                element_matrices[element_key] = stiffness
            energies += np.sum(
                element_displacements*(stiffness@element_displacements), axis=0
            )
        return energies

    def _response_at_complex_point(
        self,
        displacement: np.ndarray,
        reference_point: Sequence[float],
        positive_complex_point: Sequence[complex],
        positive_derivative: Sequence[float],
        negative_complex_point: Sequence[complex],
        negative_derivative: Sequence[float],
    ) -> np.ndarray:
        """Analytically continue the active Q4 master trace for complex step."""
        x = float(np.clip(reference_point[0], 0.0, self.width))
        y = float(np.clip(reference_point[1], 0.0, self.height))

        def active_cell(
            value: float,
            spacing: float,
            count: int,
            direction: float,
            side: float,
        ) -> int:
            """Choose the element entered on one side of a perturbation."""
            if value <= 1.0e-12*max(count*spacing, 1.0):
                return 0
            if value >= count*spacing-1.0e-12*max(count*spacing, 1.0):
                return count-1
            grid_index = int(round(value/spacing))
            on_grid_line = abs(value-grid_index*spacing) <= 1.0e-10*spacing
            if on_grid_line:
                signed_direction = side*direction
                if signed_direction < -1.0e-14:
                    return grid_index-1
                if signed_direction > 1.0e-14:
                    return grid_index
            return min(int(value/spacing), count-1)

        def response_in_cell(
            side: float,
            complex_point: Sequence[complex],
            derivative: Sequence[float],
        ) -> np.ndarray:
            derivative = np.asarray(derivative, float)
            ix = active_cell(x, self.dx, self.nx, derivative[0], side)
            iy = active_cell(y, self.dy, self.ny, derivative[1], side)
            xc, yc = complex_point
            xi = (xc-ix*self.dx)/self.dx
            eta = (yc-iy*self.dy)/self.dy
            nodes = np.array([
                self.node(ix,iy),
                self.node(ix+1,iy),
                self.node(ix+1,iy+1),
                self.node(ix,iy+1),
            ], int)
            weights = np.array([
                (1.0-xi)*(1.0-eta),
                xi*(1.0-eta),
                xi*eta,
                (1.0-xi)*eta,
            ], dtype=complex)
            return sum(
                weight*displacement[6*node:6*node+6]
                for node, weight in zip(nodes, weights)
            )

        # Q4 traces are C0 but their normal gradients jump at element edges.
        # Averaging the two analytic continuations supplies the symmetric
        # generalized derivative previously approximated by coordinate
        # centered differences, without reassembling two perturbed ribs.
        return 0.5*(
            response_in_cell(
                +1.0, positive_complex_point, positive_derivative
            )
            + response_in_cell(
                -1.0, negative_complex_point, negative_derivative
            )
        )

    def geometry_gradient(
        self,
        ribs: Sequence[Rib],
        thicknesses: Sequence[float],
        result: AnalysisResult,
        step: float,
    ) -> np.ndarray:
        """Return semi-analytic endpoint shape sensitivities by complex step.

        The current grid-conforming rib topology is held fixed.  Complex-step
        differentiation supplies the local shell-stiffness and Q4 master-trace
        derivatives, while the envelope theorem removes derivatives of the
        equilibrated free top-edge degrees of freedom.  ``step`` is accepted
        for API compatibility; unlike the validation-only centered difference,
        the production derivative has no subtractive step-size error.
        """
        del step
        # The shell matrix is assembled from sums of transformed element
        # terms.  An extremely small imaginary increment (for example 1e-30)
        # is lost when those terms cancel at machine precision, even though
        # the final energy evaluation itself has no subtraction.  A
        # sqrt(machine-epsilon)-scale geometric increment retains the matrix
        # derivative while keeping the O(h^2) complex-step truncation tiny.
        complex_step = 1.0e-8*max(self.width, self.height, 1.0)
        gradient = np.zeros((len(ribs), 4), float)
        for rib_index, (rib, thickness) in enumerate(zip(ribs, thicknesses)):
            cached_bottom, matrix = self._rib_sparse(
                rib, float(thickness), cache_basis=True
            )
            bottom, shape_derivatives = (
                self._rib_bottom_points_and_shape_derivatives(rib)
            )
            _, positive_shape_derivatives = (
                self._rib_bottom_points_and_shape_derivatives(rib, side=+1)
            )
            _, negative_shape_derivatives = (
                self._rib_bottom_points_and_shape_derivatives(rib, side=-1)
            )
            if (
                bottom.shape != cached_bottom.shape
                or not np.allclose(bottom, cached_bottom, rtol=0.0, atol=1.0e-12)
            ):
                raise RuntimeError(
                    "shape-derivative root trace does not match rib discretization"
                )
            local = self._equilibrated_local_displacements(bottom, matrix, result)
            retained_count = 6*len(bottom)
            internal = local[retained_count:, :]
            for coordinate in range(4):
                point_derivatives = shape_derivatives[:, :, coordinate]
                positive_derivatives = positive_shape_derivatives[:, :, coordinate]
                negative_derivatives = negative_shape_derivatives[:, :, coordinate]
                complex_bottom = (
                    np.asarray(bottom, complex)
                    + 1j*complex_step*point_derivatives
                )
                positive_complex_bottom = (
                    np.asarray(bottom, complex)
                    + 1j*complex_step*positive_derivatives
                )
                negative_complex_bottom = (
                    np.asarray(bottom, complex)
                    + 1j*complex_step*negative_derivatives
                )
                retained = np.column_stack([
                    np.concatenate([
                        self._response_at_complex_point(
                            displacement,
                            reference_point,
                            positive_complex_point,
                            positive_derivative,
                            negative_complex_point,
                            negative_derivative,
                        )
                        for (
                            reference_point,
                            positive_complex_point,
                            positive_derivative,
                            negative_complex_point,
                            negative_derivative,
                        ) in zip(
                            bottom,
                            positive_complex_bottom,
                            positive_derivatives,
                            negative_complex_bottom,
                            negative_derivatives,
                        )
                    ])
                    for displacement in result.displacements
                ])
                complex_local = np.vstack((retained, internal.astype(complex)))
                energies = self._rib_element_energies_from_bottom_points(
                    complex_bottom,
                    rib.height,
                    float(thickness),
                    complex_local,
                )
                weighted_energy = self.load_weights@energies
                gradient[rib_index, coordinate] = -float(
                    np.imag(weighted_energy)/complex_step
                )
        return gradient

    def geometry_gradient_centered_difference(self, ribs: Sequence[Rib], thicknesses: Sequence[float], result: AnalysisResult, step: float) -> np.ndarray:
        """Validation-only legacy centered-difference shape sensitivity."""
        jobs: list[tuple[int, int, Rib, Rib, float, float]] = []
        local_states: list[tuple[np.ndarray,np.ndarray]] = []
        for i,(rib,t) in enumerate(zip(ribs,thicknesses)):
            bottom,matrix=self._rib_sparse(rib,float(t),cache_basis=True)
            local=self._equilibrated_local_displacements(bottom,matrix,result)
            retained_count=6*len(bottom)
            local_states.append((
                self._rib_parameters(rib,bottom),
                local[retained_count:,:],
            ))
            points=np.array([rib.p0,rib.p1],float)
            for endpoint in range(2):
                for axis in range(2):
                    limit=[self.width,self.height][axis]
                    on_boundary=(
                        points[endpoint,axis] <= 1.0e-12*max(limit,1.0)
                        or points[endpoint,axis] >= limit-1.0e-12*max(limit,1.0)
                    )
                    # Moving a rib away from a ground-element boundary changes
                    # a one-sided master-surface trace. Its derivative is the
                    # h->0+ limit and needs a much smaller step than an interior
                    # centered difference.
                    boundary_step=1.0e-6*min(self.dx,self.dy)
                    delta=min(boundary_step if on_boundary else step,0.25*max(rib.length,step))
                    plus=points.copy(); minus=points.copy()
                    plus[endpoint,axis]=np.clip(plus[endpoint,axis]+delta,0,[self.width,self.height][axis])
                    minus[endpoint,axis]=np.clip(minus[endpoint,axis]-delta,0,[self.width,self.height][axis])
                    denominator=plus[endpoint,axis]-minus[endpoint,axis]
                    if denominator<=1e-12: continue
                    rp=Rib(tuple(plus[0]),tuple(plus[1]),rib.height,rib.name,rib.segments)
                    rm=Rib(tuple(minus[0]),tuple(minus[1]),rib.height,rib.name,rib.segments)
                    jobs.append((i,2*endpoint+axis,rp,rm,float(t),denominator))

        def component(job: tuple[int, int, Rib, Rib, float, float]) -> tuple[int, int, float]:
            i, coordinate, rp, rm, thickness, denominator = job
            source_parameters,source_internal=local_states[i]
            energy_plus=self._energy_with_interpolated_internal_field(
                rp,thickness,result,source_parameters,source_internal
            )
            energy_minus=self._energy_with_interpolated_internal_field(
                rm,thickness,result,source_parameters,source_internal
            )
            return i, coordinate, -(energy_plus-energy_minus)/denominator

        if self.sensitivity_workers > 1 and len(jobs) > 1:
            with ThreadPoolExecutor(max_workers=self.sensitivity_workers) as pool:
                values = list(pool.map(component, jobs))
        else:
            values = [component(job) for job in jobs]
        gradient=np.zeros((len(ribs),4))
        for i, coordinate, value in values:
            gradient[i,coordinate] = value
        return gradient

    def response_at(self, displacement: np.ndarray, point: Sequence[float]) -> np.ndarray:
        nodes,weights=self.interpolation(point)
        return sum(float(w)*displacement[6*n:6*n+6] for n,w in zip(nodes,weights))
