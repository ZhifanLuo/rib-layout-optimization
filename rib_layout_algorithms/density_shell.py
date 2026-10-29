"""SIMP optimization of a bonded parallel shell layer on an unchanged base.

The design layer occupies z=[t_base/2, t_base/2+t_layer]. It shares rotations
with the base and its midsurface translations follow the rigid-offset map.
This is an alternative bonded coating/face layer, not a vertical-rib model.
"""
from __future__ import annotations

from copy import deepcopy
import time
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.spatial import cKDTree

from rib_layout_env import configure_runtime
from .model_shell import ShellStiffenedPlateModel
from .shell import shell_q4_stiffness

configure_runtime()


def offset_transform(offset: float) -> np.ndarray:
    """Map base-reference DOFs to a parallel plane: u(z)=u(0)+theta cross r."""
    if not np.isfinite(offset):
        raise ValueError("offset must be finite")
    block = np.eye(6)
    block[0, 4], block[1, 3] = float(offset), -float(offset)
    return np.kron(np.eye(4), block)


class ParallelLayerModel:
    """One density per Q4 cell, shared base load/support semantics and SIMP."""

    def __init__(self, config: dict, *, layer_thickness: float | None = None,
                 penal: float = 3.0, emin_ratio: float = 0.0,
                 filter_radius: float | None = None, linear_solver: str = "auto"):
        self.config = deepcopy(config)
        self.nx, self.ny = map(int, config["mesh"])
        self.width, self.height = map(float, config["domain"])
        self.base_thickness = float(config["wall_thickness"])
        self.layer_thickness = float(layer_thickness if layer_thickness is not None
                                     else config["rib"]["height"]-self.base_thickness/2)
        self.penal, self.emin_ratio = float(penal), float(emin_ratio)
        self.budget = float(config["volume_bound"])
        values = [self.width, self.height, self.base_thickness, self.layer_thickness,
                  self.budget, self.penal]
        if not all(np.isfinite(values)) or min(values) <= 0:
            raise ValueError("geometry, volume budget and SIMP settings must be finite and positive")
        if min(self.nx, self.ny) < 1 or self.penal < 1 or not np.isfinite(self.emin_ratio) or not 0 <= self.emin_ratio < 1:
            raise ValueError("positive mesh, penal>=1 and 0<=emin_ratio<1 required")
        material = config["material"]
        young, poisson = float(material["E"]), float(material["nu"])
        if not np.isfinite(young) or young <= 0 or not np.isfinite(poisson) or not -1 < poisson < .5:
            raise ValueError("isotropic material requires finite E>0 and -1<nu<0.5")
        loads = deepcopy(config["load_cases"])
        weights = np.asarray([x.get("weight", 1.0) for x in loads], float)
        if not len(loads) or not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.max() <= 0:
            raise ValueError("load weights must be finite, nonnegative, with positive sum")
        # Scale before summing: finite weights can otherwise overflow their sum
        # and silently become all zero in the base model's normalization.
        weights = weights/weights.max()
        weights /= weights.sum()
        for load, weight in zip(loads, weights):
            load["weight"] = float(weight)
        self.base = ShellStiffenedPlateModel(
            width=self.width, height=self.height, nx=self.nx, ny=self.ny,
            wall_thickness=self.base_thickness, E=young,
            nu=poisson, loads=loads, supports=config["supports"],
            linear_solver=linear_solver)
        self.dx, self.dy = self.base.dx, self.base.dy
        self.nelem = self.nx*self.ny
        self.cell_volume = self.dx*self.dy*self.layer_thickness
        self.capacity = self.nelem*self.cell_volume
        self.volume_fraction = self.budget/self.capacity
        if self.volume_fraction > 1+1e-12:
            raise ValueError("material budget exceeds full design-layer capacity; choose a thicker layer")
        self.offset = (self.base_thickness+self.layer_thickness)/2
        self.transform = offset_transform(self.offset)
        xyz = np.array([[0, 0, 0], [self.dx, 0, 0],
                        [self.dx, self.dy, 0], [0, self.dy, 0]], float)
        solid = shell_q4_stiffness(xyz, self.layer_thickness, self.base.E, self.base.nu)
        self.layer_element = self.transform.T@solid@self.transform
        elements = []
        for iy in range(self.ny):
            for ix in range(self.nx):
                nodes = [self.base.node(ix, iy), self.base.node(ix+1, iy),
                         self.base.node(ix+1, iy+1), self.base.node(ix, iy+1)]
                elements.append([6*n+j for n in nodes for j in range(6)])
        self.edofs = np.asarray(elements, int)
        self.rows = np.repeat(self.edofs, 24, axis=1).ravel()
        self.cols = np.tile(self.edofs, (1, 24)).ravel()
        self.filter_radius = float(filter_radius if filter_radius is not None
                                   else 1.5*max(self.dx, self.dy))
        if not np.isfinite(self.filter_radius) or self.filter_radius <= 0:
            raise ValueError("filter radius must be finite and positive in physical length units")
        centers = np.array([[(ix+.5)*self.dx, (iy+.5)*self.dy]
                            for iy in range(self.ny) for ix in range(self.nx)])
        tree = cKDTree(centers)
        pairs = tree.sparse_distance_matrix(tree, self.filter_radius, output_type="coo_matrix")
        h = coo_matrix((np.maximum(0., self.filter_radius-pairs.data),
                        (pairs.row, pairs.col)), shape=(self.nelem, self.nelem)).tocsr()
        self.filter = diags(1/np.asarray(h.sum(axis=1)).ravel())@h
        self.symmetry = tuple(config.get("mirror_symmetry", []))
        if any(axis not in ("x", "y") for axis in self.symmetry):
            raise ValueError("mirror_symmetry axes must be x or y")
        self.analysis_count = 0

    def symmetric(self, vector: np.ndarray) -> np.ndarray:
        grid = np.asarray(vector, float).reshape(self.ny, self.nx).copy()
        if "x" in self.symmetry:
            grid = .5*(grid+grid[:, ::-1])
        if "y" in self.symmetry:
            grid = .5*(grid+grid[::-1, :])
        return grid.ravel()

    def physical_density(self, raw: np.ndarray) -> np.ndarray:
        physical = np.asarray(self.filter@self.symmetric(self._density(raw))).ravel()
        # A row-normalized nonnegative filter maps [0,1] into [0,1] exactly.
        # Sparse normalization/summation can overshoot by a few floating-point
        # ulps at a solid cell (observed 9e-16 on the 160x80 case-II grid).
        # Repair only that arithmetic noise, retaining strict raw-input and
        # direct physical-density validation and rejecting genuine bad filters.
        tolerance = 64*np.finfo(float).eps
        if (physical.shape != (self.nelem,) or not np.all(np.isfinite(physical))
                or np.any(physical < -tolerance) or np.any(physical > 1+tolerance)):
            raise ValueError("physical-density filter violated [0,1] beyond floating-point roundoff")
        return np.clip(physical, 0., 1.)

    def _density(self, values: np.ndarray) -> np.ndarray:
        rho = np.asarray(values, float)
        if rho.shape != (self.nelem,) or not np.all(np.isfinite(rho)) or np.any(rho < 0) or np.any(rho > 1):
            raise ValueError("density must be a finite element-length vector in [0,1]")
        return rho

    def volume_gradient(self) -> np.ndarray:
        return self.symmetric(np.asarray(self.filter.T@np.full(self.nelem, self.cell_volume)).ravel())

    def stiffness(self, physical: np.ndarray):
        rho = self._density(physical)
        scale = self.emin_ratio+(1-self.emin_ratio)*rho**self.penal
        data = (scale[:, None]*self.layer_element.ravel()[None, :]).ravel()
        added = coo_matrix((data, (self.rows, self.cols)),
                           shape=(self.base.ndof, self.base.ndof)).tocsc()
        return self.base.base_stiffness+added

    def _equilibrium_solve(self, matrix, rhs: np.ndarray) -> tuple[np.ndarray, dict]:
        """Retain the normal solve; repair numerical failure on the same equation.

        Thin-base refined meshes can lose accuracy from mixed translation and
        rotation scales. Only if the ordinary solve fails its original 1e-7
        residual check, solve D K D y = D f and recover u=D y. Corrections use
        the original residual f-Ku, and acceptance always uses the original
        matrix and RHS. No diagonal regularization or tolerance relaxation.
        """
        norm = max(float(np.linalg.norm(rhs)), 1e-30)
        def residual(solution):
            if not np.all(np.isfinite(solution)):
                return float("inf")
            return float(np.linalg.norm(matrix@solution-rhs)/norm)
        def solve(system, force):
            solution = self.base._solve_global(system, force)
            return solution[:, None] if solution.ndim == 1 else solution
        initial_error = None
        try:
            original = solve(matrix, rhs)
            original_residual = residual(original)
            if original_residual <= 1e-7:
                return original, dict(method="original", refinement_steps=0,
                                      relative_residual=original_residual)
        except (RuntimeError, np.linalg.LinAlgError) as error:
            initial_error = str(error)
            original_residual = float("inf")
        diagonal = matrix.diagonal()
        if not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0):
            raise RuntimeError("density-layer equilibrium cannot equilibrate a nonpositive/nonfinite diagonal")
        scale = 1/np.sqrt(diagonal)
        d = diags(scale)
        equilibrated = (d@matrix@d).tocsc()
        solution = scale[:, None]*solve(equilibrated, scale[:, None]*rhs)
        best = solution
        best_residual = residual(solution)
        steps = 0
        while best_residual > 1e-7 and steps < 4:
            correction_rhs = rhs-matrix@best
            correction = solve(equilibrated, scale[:, None]*correction_rhs)
            candidate = best+scale[:, None]*correction
            candidate_residual = residual(candidate)
            steps += 1
            if candidate_residual < best_residual:
                best, best_residual = candidate, candidate_residual
        if best_residual > 1e-7 or not np.isfinite(best_residual):
            raise RuntimeError(f"density-layer equilibrium failed original-equation residual check after "
                               f"equilibration/refinement: {best_residual}")
        return best, dict(method="equilibrated_fallback", refinement_steps=steps,
                          initial_relative_residual=original_residual if np.isfinite(original_residual) else None,
                          initial_numerical_error=initial_error, relative_residual=best_residual)

    def evaluate_physical(self, physical: np.ndarray) -> dict:
        """Analyze supplied physical density without filtering it a second time."""
        rho = self._density(physical)
        k = self.stiffness(rho)
        free = self.base.free_dofs
        forces = np.column_stack(self.base.load_vectors)
        reduced = k[free][:, free].tocsc()
        rhs = forces[free]
        solved, solve_details = self._equilibrium_solve(reduced, rhs)
        residual = solve_details["relative_residual"]
        displacement = np.zeros_like(forces)
        displacement[free] = solved
        compliances = np.sum(forces*displacement, axis=0)
        objective = float(self.base.load_weights@compliances)
        local_u = displacement[self.edofs]
        energy = np.einsum("eim,ij,ejm,m->e", local_u, self.layer_element,
                           local_u, self.base.load_weights, optimize=True)
        grad = -self.penal*(1-self.emin_ratio)*rho**(self.penal-1)*energy
        if not np.isfinite(objective) or objective < 0:
            raise RuntimeError("invalid compliance")
        self.analysis_count += 1
        return dict(compliance=objective, load_compliances=compliances,
                    load_weights=self.base.load_weights.copy(),
                    max_downward_displacement=np.maximum(0., -displacement[2::6].min(axis=0)),
                    displacement=displacement, physical_density=rho.copy(),
                    volume=float(self.cell_volume*rho.sum()),
                    physical_gradient=grad, relative_residual=float(residual),
                    linear_solution=solve_details)

    def analyze(self, raw: np.ndarray) -> dict:
        response = self.evaluate_physical(self.physical_density(raw))
        response["gradient"] = self.symmetric(np.asarray(
            self.filter.T@response["physical_gradient"]).ravel())
        return response

    def _oc_update(self, raw: np.ndarray, gradient: np.ndarray, move: float) -> np.ndarray:
        dv = self.volume_gradient()
        lower, upper = np.maximum(0., raw-move), np.minimum(1., raw+move)
        # A positive seed permits re-entry if a zero raw cell retains a negative
        # adjoint gradient through neighbors. Entirely void filter neighborhoods
        # have zero SIMP derivative for p>1 and cannot re-enter under OC alone.
        seed = np.maximum(raw, 1e-9)
        def trial(multiplier):
            return self.symmetric(np.clip(seed*np.sqrt(np.maximum(-gradient, 0.)/
                                  (np.maximum(multiplier, 1e-300)*dv)), lower, upper))
        def volume(x):
            return float(self.cell_volume*self.physical_density(x).sum())
        lo, hi = 0., max(float(np.max(-gradient/dv)), 1e-30)
        while volume(trial(hi)) > self.budget:
            hi *= 2
            if not np.isfinite(hi):
                raise RuntimeError("failed to bracket OC volume multiplier")
        for _ in range(100):
            mid = .5*(lo+hi)
            if volume(trial(mid)) > self.budget:
                lo = mid
            else:
                hi = mid
        return trial(hi)

    def optimize(self, *, max_iterations: int = 300, move: float = .1,
                 density_tolerance: float = .005, raw_density_tolerance: float = .01,
                 objective_tolerance: float = 1e-4,
                 initial_raw: np.ndarray | None = None,
                 callback=None) -> dict:
        if int(max_iterations) != max_iterations or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if not all(np.isfinite([move, density_tolerance, raw_density_tolerance, objective_tolerance])) or not (
                0 < move <= 1 and density_tolerance > 0 and raw_density_tolerance > 0 and objective_tolerance > 0):
            raise ValueError("positive finite convergence tolerances and 0<move<=1 required")
        started = time.perf_counter()
        if initial_raw is None:
            raw = np.full(self.nelem, min(1., self.volume_fraction))
            initialization_type = "uniform_at_budget"
        else:
            raw = self.symmetric(self._density(initial_raw))
            initial_volume = self.cell_volume*self.physical_density(raw).sum()
            if initial_volume > self.budget*(1+1e-10):
                raise ValueError("initial raw density exceeds the physical-volume budget after symmetry/filtering")
            initialization_type = "provided_raw_density"
        # For SIMP p>1, an entirely void start has zero sensitivities, so OC
        # cannot add material and change-based stopping would mislabel it.
        if not np.any(raw > 0):
            raise ValueError("optimization requires a nonzero initial density; an all-zero start can trap SIMP/OC at zero sensitivity")
        initial = self.analyze(raw)
        current = initial
        history = []
        stable = 0
        converged = False
        for iteration in range(1, int(max_iterations)+1):
            trial = self._oc_update(raw, current["gradient"], move)
            updated = self.analyze(trial)
            raw_change = float(np.max(np.abs(trial-raw)))
            change = float(np.max(np.abs(updated["physical_density"]-current["physical_density"])))
            objective_change = abs(updated["compliance"]-current["compliance"])/max(current["compliance"], 1e-30)
            feasible = updated["volume"] <= self.budget*(1+1e-10)
            if not feasible:
                raise RuntimeError("OC update violated the physical-volume constraint")
            stable = stable+1 if (raw_change <= raw_density_tolerance and
                                  change <= density_tolerance and
                                  objective_change <= objective_tolerance) else 0
            row = dict(iteration=iteration, compliance=updated["compliance"],
                       volume=updated["volume"], raw_change=raw_change, physical_change=change,
                       relative_objective_change=objective_change,
                       relative_residual=updated["relative_residual"], analyses=self.analysis_count)
            history.append(row)
            if callback:
                callback(row)
            raw, current = trial, updated
            if stable >= 3:
                converged = True
                break
        return dict(initial=initial, final=current, raw_density=raw, history=history,
                    initialization=dict(type=initialization_type,
                                        physical_volume=float(initial["volume"])),
                    converged=converged, termination_reason="convergence" if converged else "iteration_limit",
                    iterations=len(history), analysis_count=self.analysis_count,
                    elapsed_seconds=time.perf_counter()-started)

    def binary_design(self, physical: np.ndarray) -> np.ndarray:
        """Highest-density mirror orbits within the budget; no fractional cells."""
        rho = self._density(physical)
        seen, orbits = set(), []
        for index in range(self.nelem):
            if index in seen:
                continue
            iy, ix = divmod(index, self.nx)
            xs = {ix, self.nx-1-ix} if "x" in self.symmetry else {ix}
            ys = {iy, self.ny-1-iy} if "y" in self.symmetry else {iy}
            group = sorted(y*self.nx+x for y in ys for x in xs)
            seen.update(group)
            orbits.append((float(np.mean(rho[group])), group))
        out = np.zeros(self.nelem)
        used = 0
        max_cells = int(np.floor(self.budget/self.cell_volume+1e-10))
        for _, group in sorted(orbits, key=lambda item: (-item[0], item[1][0])):
            if used+len(group) <= max_cells:
                out[group] = 1.
                used += len(group)
        return out

    def metadata(self) -> dict:
        return dict(model="fixed base plus bonded parallel design shell",
                    layer_thickness=self.layer_thickness, layer_centroid_z=self.offset,
                    layer_bottom_z=self.base_thickness/2,
                    layer_top_z=self.base_thickness/2+self.layer_thickness,
                    fixed_base_thickness=self.base_thickness,
                    fixed_base_volume=self.width*self.height*self.base_thickness,
                    design_volume_budget=self.budget, solid_design_capacity=self.capacity,
                    target_volume_fraction=self.volume_fraction, penal=self.penal,
                    emin_ratio=self.emin_ratio, filter_radius=self.filter_radius,
                    filter="row-normalized physical-density filter with exact adjoint",
                    mirror_symmetry=list(self.symmetry), mesh=[self.nx, self.ny],
                    layer_kinematics="perfect bond, common rotations, rigid midsurface offset",
                    comparison_limit="horizontal face-layer alternative, not vertical ribs; no optimizer ranking",
                    manufacturing_constraints="none; no draw constraint or connectivity/feature certification")
