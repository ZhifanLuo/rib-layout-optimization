"""Internal multi-phase active-set rib-layout optimization."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares, minimize

from .model import AnalysisResult, Rib, StiffenedPlateModel
from .move_limit import EnhancedMMAMoveLimit
from .symmetry import (
    build_mirror_variable_map,
    mirror_axes,
    mirror_groups,
    mirror_rib,
)


@dataclass
class Stage:
    name: str
    ribs: list[Rib]
    thicknesses: np.ndarray
    compliance: float
    analyses: int
    note: str = ""


@dataclass
class OptimizationRun:
    stages: list[Stage] = field(default_factory=list)
    active_history: list[Stage] = field(default_factory=list)
    rationalization_histories: dict[str, list[dict]] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)


def smooth_member_count(
    thicknesses: Sequence[float],
    threshold: float,
    beta: float,
) -> float:
    """Paper Eq. (15): smooth approximation of the number of active ribs."""
    safe = np.maximum(np.asarray(thicknesses, float), np.finfo(float).tiny)
    z = np.clip(float(beta) * (safe / float(threshold) - 1.0), -30.0, 30.0)
    return float(np.sum(0.5 * (1.0 + np.tanh(z))))


def smooth_member_count_gradient(
    thicknesses: Sequence[float],
    threshold: float,
    beta: float,
) -> np.ndarray:
    """Derivative of the smooth member count with respect to thickness."""
    safe = np.maximum(np.asarray(thicknesses, float), np.finfo(float).tiny)
    z = np.clip(float(beta) * (safe / float(threshold) - 1.0), -30.0, 30.0)
    tanh_z = np.tanh(z)
    return 0.5 * float(beta) / float(threshold) * (1.0 - tanh_z**2)


def sca_step_converged(
    feasible: bool,
    objective_change: float,
    design_change: float,
    objective_tolerance: float,
    design_tolerance: float,
    design_guard_tolerance: float,
) -> bool:
    """Common SCA outer-step convergence test.

    The 1%-style design guard prevents a temporary objective plateau from
    terminating an optimization while its design variables are still moving
    appreciably.  Inside that guard the original objective/design OR test is
    retained.
    """
    if design_guard_tolerance < design_tolerance:
        raise ValueError("SCA design guard tolerance must not be smaller than the design tolerance")
    return bool(
        feasible
        and design_change < design_guard_tolerance
        and (
            objective_change < objective_tolerance
            or design_change < design_tolerance
        )
    )


def solve_geometry_convex_subproblem(
    current: np.ndarray,
    reciprocal_coefficients: np.ndarray,
    geometry_gradient: np.ndarray,
    volume_gradient: np.ndarray,
    volume_at_current: float,
    volume_bound: float,
    proximal: float,
    compliance_scale: float,
    coordinate_scale: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Solve the Eq. (7) separable convex subproblem by dual bisection.

    The approximation has one affine volume constraint and box move limits.
    For a fixed volume multiplier every thickness and coordinate has a
    closed-form minimizer. A two-sided multiplier search makes the linearized
    volume as close as possible to the prescribed bound: it reaches equality
    when the box permits and otherwise returns the minimum-residual bound
    solution. Thus a volume-slack unconstrained minimizer is not returned
    without first solving the closest-volume problem.
    """
    x = np.asarray(current, float)
    a = np.asarray(reciprocal_coefficients, float)
    gp = np.asarray(geometry_gradient, float)
    vg = np.asarray(volume_gradient, float)
    lo = np.asarray(lower, float)
    hi = np.asarray(upper, float)
    n = len(a)
    if (
        x.shape != lo.shape
        or x.shape != hi.shape
        or vg.shape != x.shape
        or len(x) != n+len(gp)
        or len(coordinate_scale) != len(gp)
    ):
        raise ValueError("inconsistent separable convex-subproblem dimensions")
    if proximal <= 0.0 or compliance_scale <= 0.0:
        raise ValueError("geometry convex subproblem requires positive proximal scaling")

    def minimizer(multiplier: float) -> np.ndarray:
        y = np.empty_like(x)
        weighted_volume = multiplier*vg[:n]
        positive = (a > 0.0) & (weighted_volume > 0.0)
        y[:n] = x[:n]
        use_upper = (a > 0.0) | (weighted_volume < 0.0)
        y[:n][use_upper] = hi[:n][use_upper]
        y[:n][positive] = np.sqrt(a[positive]/weighted_volume[positive])
        y[:n][(a <= 0.0) & (weighted_volume > 0.0)] = lo[:n][
            (a <= 0.0) & (weighted_volume > 0.0)
        ]
        y[:n] = np.clip(y[:n], lo[:n], hi[:n])
        coordinate_linear = gp+multiplier*vg[n:]
        y[n:] = x[n:] - (
            coordinate_linear*coordinate_scale**2/(proximal*compliance_scale)
        )
        y[n:] = np.clip(y[n:], lo[n:], hi[n:])
        return y

    def linearized_volume(y: np.ndarray) -> float:
        return float(volume_at_current+vg@(y-x))

    candidate = minimizer(0.0)
    tolerance = 1.0e-12*max(abs(volume_bound), 1.0)
    candidate_volume = linearized_volume(candidate)
    if abs(candidate_volume-volume_bound) <= tolerance:
        return candidate

    if candidate_volume > volume_bound:
        lower_multiplier = 0.0
        upper_multiplier = 1.0
        upper_candidate = minimizer(upper_multiplier)
        while linearized_volume(upper_candidate) > volume_bound+tolerance:
            upper_multiplier *= 10.0
            if upper_multiplier > 1.0e30:
                # Even the minimum-volume box solution exceeds the bound.
                return upper_candidate
            upper_candidate = minimizer(upper_multiplier)
    else:
        lower_multiplier = -1.0
        upper_multiplier = 0.0
        lower_candidate = minimizer(lower_multiplier)
        while linearized_volume(lower_candidate) < volume_bound-tolerance:
            lower_multiplier *= 10.0
            if lower_multiplier < -1.0e30:
                # Equality is unreachable; this is the maximum-volume box
                # solution and therefore has the smallest absolute residual.
                return lower_candidate
            lower_candidate = minimizer(lower_multiplier)

    for _ in range(80):
        multiplier = 0.5*(lower_multiplier+upper_multiplier)
        candidate = minimizer(multiplier)
        if linearized_volume(candidate) > volume_bound:
            lower_multiplier = multiplier
        else:
            upper_multiplier = multiplier
    lower_candidate = minimizer(lower_multiplier)
    upper_candidate = minimizer(upper_multiplier)
    lower_residual = abs(linearized_volume(lower_candidate)-volume_bound)
    upper_residual = abs(linearized_volume(upper_candidate)-volume_bound)
    return lower_candidate if lower_residual < upper_residual else upper_candidate


@dataclass
class RationalizationDualResult:
    """Solution and diagnostics for one convex Eq. (18) approximation."""

    x: np.ndarray
    success: bool
    status: int
    iterations: int
    message: str
    compliance_residual: float
    volume_residual: float
    compliance_multiplier: float
    volume_multiplier: float


def solve_rationalization_convex_subproblem(
    current: np.ndarray,
    count_gradient: np.ndarray,
    reciprocal_coefficients: np.ndarray,
    geometry_gradient: np.ndarray,
    volume_gradient: np.ndarray,
    compliance_at_current: float,
    compliance_bound: float,
    volume_at_current: float,
    volume_bound: float,
    proximal: float,
    compliance_scale: float,
    thickness_scale: np.ndarray,
    coordinate_scale: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    dual_iterations: int = 70,
    constraint_tolerance: float = 1.0e-9,
) -> RationalizationDualResult:
    """Solve one convex Eq. (18) approximation through its two-variable dual.

    Given the compliance and volume multipliers, every coordinate has an
    explicit minimizer and every thickness is the unique positive root of a
    monotone cubic KKT equation.  Nested multiplier bisections therefore
    replace the former high-dimensional SLSQP iteration.
    """
    x = np.asarray(current, float)
    count = np.asarray(count_gradient, float)
    a = np.asarray(reciprocal_coefficients, float)
    gp = np.asarray(geometry_gradient, float)
    vg = np.asarray(volume_gradient, float)
    thickness_scale = np.asarray(thickness_scale, float)
    coordinate_scale = np.asarray(coordinate_scale, float)
    lo = np.asarray(lower, float)
    hi = np.asarray(upper, float)
    n = len(a)
    if (
        x.shape != count.shape
        or x.shape != vg.shape
        or x.shape != lo.shape
        or x.shape != hi.shape
        or len(x) != n+len(gp)
        or thickness_scale.shape != (n,)
        or coordinate_scale.shape != gp.shape
    ):
        raise ValueError("inconsistent rationalization dual dimensions")
    if proximal <= 0.0 or compliance_scale <= 0.0:
        raise ValueError("rationalization dual requires positive proximal scaling")
    if np.any(lo[:n] <= 0.0) or np.any(thickness_scale <= 0.0):
        raise ValueError("rationalization thickness bounds and scales must be positive")
    if dual_iterations < 20:
        raise ValueError("rationalization dual_iterations must be at least 20")
    if constraint_tolerance <= 0.0:
        raise ValueError("rationalization dual constraint tolerance must be positive")

    thickness_quadratic = proximal/thickness_scale**2
    coordinate_count_gradient = count[n:]
    evaluations = 0

    def minimizer(compliance_multiplier: float, volume_multiplier: float) -> np.ndarray:
        nonlocal evaluations
        evaluations += 1
        candidate = np.empty_like(x)

        # The derivative is strictly increasing on t>0:
        # q(t-tk)+c+mu*v-lambda*a/t^2 = 0.  Bound tests plus a
        # vectorized bisection give the unique box-constrained minimizer.
        forcing = count[:n]+volume_multiplier*vg[:n]
        left = lo[:n].copy()
        right = hi[:n].copy()

        def derivative(values: np.ndarray) -> np.ndarray:
            return (
                thickness_quadratic*(values-x[:n])
                + forcing
                - compliance_multiplier*a/values**2
            )

        derivative_left = derivative(left)
        derivative_right = derivative(right)
        at_lower = derivative_left >= 0.0
        at_upper = derivative_right <= 0.0
        active = ~(at_lower | at_upper)
        for _ in range(55):
            midpoint = 0.5*(left+right)
            derivative_midpoint = derivative(midpoint)
            move_left = active & (derivative_midpoint < 0.0)
            move_right = active & ~move_left
            left[move_left] = midpoint[move_left]
            right[move_right] = midpoint[move_right]
        candidate[:n] = 0.5*(left+right)
        candidate[:n][at_lower] = lo[:n][at_lower]
        candidate[:n][at_upper] = hi[:n][at_upper]

        denominator = proximal*(1.0+compliance_multiplier*compliance_scale)
        coordinate_forcing = (
            coordinate_count_gradient
            + compliance_multiplier*gp
            + volume_multiplier*vg[n:]
        )
        candidate[n:] = x[n:] - (
            coordinate_forcing*coordinate_scale**2/denominator
        )
        candidate[n:] = np.clip(candidate[n:], lo[n:], hi[n:])
        return candidate

    def residuals(candidate: np.ndarray) -> tuple[float, float]:
        safe = np.maximum(candidate[:n], lo[:n])
        dp = candidate[n:]-x[n:]
        compliance = (
            compliance_at_current
            + np.sum(a*(1.0/safe-1.0/x[:n]))
            + gp@dp
            + 0.5*proximal*compliance_scale
            * np.sum((dp/coordinate_scale)**2)
        )
        volume = volume_at_current+vg@(candidate-x)
        return float(compliance-compliance_bound), float(volume-volume_bound)

    compliance_normalizer = max(abs(compliance_bound), 1.0)
    volume_normalizer = max(abs(volume_bound), 1.0)
    compliance_tolerance = constraint_tolerance*compliance_normalizer
    volume_tolerance = constraint_tolerance*volume_normalizer

    def dual_value_and_gradient(
        normalized_multipliers: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        compliance_multiplier = (
            normalized_multipliers[0]/compliance_normalizer
        )
        volume_multiplier = normalized_multipliers[1]/volume_normalizer
        candidate = minimizer(compliance_multiplier, volume_multiplier)
        compliance_residual, volume_residual = residuals(candidate)
        difference = candidate-x
        objective = (
            count@difference
            + 0.5*proximal
            * np.sum((difference[:n]/thickness_scale)**2)
            + 0.5*proximal
            * np.sum((difference[n:]/coordinate_scale)**2)
        )
        normalized_compliance = compliance_residual/compliance_normalizer
        normalized_volume = volume_residual/volume_normalizer
        dual_value = (
            objective
            + normalized_multipliers[0]*normalized_compliance
            + normalized_multipliers[1]*normalized_volume
        )
        # Envelope theorem: the dual derivatives are the constraint residuals.
        return -float(dual_value), -np.array([
            normalized_compliance, normalized_volume
        ])

    dual_optimum = minimize(
        dual_value_and_gradient,
        np.zeros(2),
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, None), (0.0, None)],
        options={
            "maxiter": max(100, 2*dual_iterations),
            "ftol": 1.0e-15,
            "gtol": 1.0e-10,
            "maxls": 50,
        },
    )
    normalized_multipliers = np.maximum(np.asarray(dual_optimum.x, float), 0.0)

    def normalized_constraint_residuals(
        active_values: np.ndarray,
        active_indices: np.ndarray,
    ) -> np.ndarray:
        multipliers = normalized_multipliers.copy()
        multipliers[active_indices] = active_values
        candidate = minimizer(
            multipliers[0]/compliance_normalizer,
            multipliers[1]/volume_normalizer,
        )
        compliance_residual, volume_residual = residuals(candidate)
        all_residuals = np.array([
            compliance_residual/compliance_normalizer,
            volume_residual/volume_normalizer,
        ])
        return all_residuals[active_indices]

    initial_candidate = minimizer(
        normalized_multipliers[0]/compliance_normalizer,
        normalized_multipliers[1]/volume_normalizer,
    )
    initial_compliance, initial_volume = residuals(initial_candidate)
    initial_normalized_residuals = np.array([
        initial_compliance/compliance_normalizer,
        initial_volume/volume_normalizer,
    ])
    active_indices = np.flatnonzero(
        (normalized_multipliers > 1.0e-10)
        | (initial_normalized_residuals > 1.0e-10)
    )
    if len(active_indices):
        refinement = least_squares(
            lambda values: normalized_constraint_residuals(
                values, active_indices
            ),
            normalized_multipliers[active_indices],
            bounds=(np.zeros(len(active_indices)), np.full(len(active_indices), np.inf)),
            xtol=1.0e-13,
            ftol=1.0e-13,
            gtol=1.0e-13,
            max_nfev=100,
        )
        normalized_multipliers[active_indices] = refinement.x

    compliance_multiplier = float(
        normalized_multipliers[0]/compliance_normalizer
    )
    volume_multiplier = float(normalized_multipliers[1]/volume_normalizer)
    candidate = minimizer(compliance_multiplier, volume_multiplier)
    compliance_residual, volume_residual = residuals(candidate)
    success = bool(
        compliance_residual <= compliance_tolerance
        and volume_residual <= volume_tolerance
    )
    message = (
        "two-multiplier dual KKT solution"
        if success
        else "dual solution exceeds a convex constraint tolerance: "
        + str(dual_optimum.message)
    )
    return RationalizationDualResult(
        candidate,
        success,
        0 if success else int(getattr(dual_optimum, "status", 1))+1,
        evaluations,
        message,
        compliance_residual,
        volume_residual,
        compliance_multiplier,
        volume_multiplier,
    )


def maximum_normalized_design_change(
    previous: np.ndarray,
    current: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Return the maximum design step normalized by each global range."""
    scale = np.maximum(np.asarray(upper, float)-np.asarray(lower, float), 1.0e-30)
    return float(np.max(np.abs(np.asarray(current)-np.asarray(previous))/scale))


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab, ac = b - a, c - a
    return float(ab[0] * ac[1] - ab[1] * ac[0])


def collinear_overlap(a: Rib, b: Rib, tolerance: float = 1.0e-8) -> bool:
    p, q = np.asarray(a.p0), np.asarray(a.p1)
    r, s = np.asarray(b.p0), np.asarray(b.p1)
    a_length = float(np.linalg.norm(q-p))
    b_length = float(np.linalg.norm(s-r))
    if a_length <= tolerance or b_length <= tolerance:
        return False
    scale = max(a.length, b.length, 1.0)
    if abs(_orientation(p, q, r)) > tolerance * scale or abs(_orientation(p, q, s)) > tolerance * scale:
        return False
    axis = (q - p) / a_length
    a0, a1 = sorted((float(p @ axis), float(q @ axis)))
    b0, b1 = sorted((float(r @ axis), float(s @ axis)))
    return min(a1, b1) - max(a0, b0) > tolerance


def collinear_covered(candidate: Rib, existing: Rib, tolerance: float = 1.0e-8) -> bool:
    """Return whether ``candidate`` is wholly covered by ``existing``.

    Unlike :func:`collinear_overlap`, this directional predicate does not
    reject a candidate that merely extends across part of an existing rib.
    """
    p, q = np.asarray(candidate.p0, float), np.asarray(candidate.p1, float)
    r, s = np.asarray(existing.p0, float), np.asarray(existing.p1, float)
    existing_length = float(np.linalg.norm(s-r))
    scale = max(candidate.length, existing_length, 1.0)
    if existing_length <= tolerance:
        return False
    if (
        abs(_orientation(r, s, p)) > tolerance*scale
        or abs(_orientation(r, s, q)) > tolerance*scale
    ):
        return False
    axis = (s-r)/existing_length
    candidate_min, candidate_max = sorted((float(p@axis), float(q@axis)))
    existing_min, existing_max = sorted((float(r@axis), float(s@axis)))
    return (
        candidate_min >= existing_min-tolerance
        and candidate_max <= existing_max+tolerance
    )


def collinear_union_covering_ribs(
    candidate: Rib,
    existing_ribs: Sequence[Rib],
    tolerance: float = 1.0e-8,
) -> list[Rib]:
    """Return existing collinear ribs whose union fully covers ``candidate``.

    Coverage may be supplied by one rib or by several adjacent/overlapping
    ribs.  A positive gap larger than ``tolerance`` leaves the candidate
    uncovered and returns an empty list.
    """
    p = np.asarray(candidate.p0, float)
    q = np.asarray(candidate.p1, float)
    candidate_length = float(np.linalg.norm(q-p))
    if candidate_length <= tolerance:
        return []
    axis = (q-p)/candidate_length
    intervals: list[tuple[float, float, Rib]] = []
    for rib in existing_ribs:
        r = np.asarray(rib.p0, float)
        s = np.asarray(rib.p1, float)
        scale = max(candidate_length, rib.length, 1.0)
        if (
            abs(_orientation(p, q, r)) > tolerance*scale
            or abs(_orientation(p, q, s)) > tolerance*scale
        ):
            continue
        start, end = sorted((float((r-p)@axis), float((s-p)@axis)))
        start = max(start, 0.0)
        end = min(end, candidate_length)
        if end >= start-tolerance:
            intervals.append((start, end, rib))

    covered_until = 0.0
    contributors: list[Rib] = []
    for start, end, rib in sorted(
        intervals, key=lambda item: (item[0], -item[1], item[2].name)
    ):
        if start > covered_until+tolerance:
            break
        if end > covered_until+tolerance:
            contributors.append(rib)
            covered_until = end
        if covered_until >= candidate_length-tolerance:
            return contributors
    return []


def geometry_move_freeze_reasons(
    current_ribs: Sequence[Rib],
    candidate_ribs: Sequence[Rib],
    minimum_length: float,
) -> dict[int, set[str]]:
    """Identify only the rib positions responsible for invalid geometry.

    A newly short rib freezes its own endpoint coordinates.  For a newly
    collinear-overlapping pair, compare each moved rib against the other rib's
    current position.  This freezes only the causative rib when it can be
    identified uniquely; otherwise both positions are frozen.  Geometry that
    was already present at the current outer iterate is treated as the valid
    baseline and is not repeatedly flagged.
    """
    if len(current_ribs) != len(candidate_ribs):
        raise ValueError("current and candidate rib sets must have equal length")
    reasons: dict[int, set[str]] = {}

    def add(index: int, reason: str) -> None:
        reasons.setdefault(int(index), set()).add(reason)

    for index, (current, candidate) in enumerate(
        zip(current_ribs, candidate_ribs)
    ):
        if current.length >= minimum_length and candidate.length < minimum_length:
            add(index, "short")

    for i in range(len(candidate_ribs)):
        for j in range(i+1, len(candidate_ribs)):
            if not collinear_overlap(candidate_ribs[i], candidate_ribs[j]):
                continue
            if collinear_overlap(current_ribs[i], current_ribs[j]):
                continue
            i_causes = collinear_overlap(candidate_ribs[i], current_ribs[j])
            j_causes = collinear_overlap(current_ribs[i], candidate_ribs[j])
            if i_causes and not j_causes:
                add(i, f"overlap:{candidate_ribs[j].name}")
            elif j_causes and not i_causes:
                add(j, f"overlap:{candidate_ribs[i].name}")
            else:
                add(i, f"overlap:{candidate_ribs[j].name}")
                add(j, f"overlap:{candidate_ribs[i].name}")
    return reasons


class RibLayoutOptimizer:
    def __init__(self, model: StiffenedPlateModel, cfg: dict) -> None:
        self.model, self.cfg = model, cfg
        self.volume_bound = float(cfg["volume_bound"])
        self.t0 = float(cfg["rib"]["initial"])
        self.t_lower = float(cfg["rib"]["lower"])
        self.t_upper = float(cfg["rib"]["upper"])
        self.analysis_count = 0
        self.log: list[str] = []
        self.active_history: list[Stage] = []
        self.sizing_history: list[dict] = []
        self.rationalization_history: list[dict] = []
        self.sizing_termination_reason: str | None = None
        self.geometry_termination_reason: str | None = None
        self.rationalization_termination_reason: str | None = None
        self.mirror_axes = mirror_axes(cfg)
        self.symmetry_width, self.symmetry_height = map(float, cfg["domain"])

    def _mirror_groups(self, ribs: Sequence[Rib]) -> list[list[int]]:
        return mirror_groups(
            ribs,
            self.mirror_axes,
            self.symmetry_width,
            self.symmetry_height,
        )

    def _mirror_expand_mask(
        self,
        ribs: Sequence[Rib],
        mask: Sequence[bool],
    ) -> np.ndarray:
        """Expand any discrete rib decision to its complete mirror group."""
        expanded = np.asarray(mask, bool).copy()
        if len(expanded) != len(ribs):
            raise ValueError("mirror mask and rib counts must match")
        for group in self._mirror_groups(ribs):
            if np.any(expanded[group]):
                expanded[group] = True
        return expanded

    def _mirror_group_removal_mask(
        self,
        ribs: Sequence[Rib],
        mask: Sequence[bool],
    ) -> np.ndarray:
        """Delete an orbit only when every available mirror member qualifies."""
        qualifying = np.asarray(mask, bool)
        if len(qualifying) != len(ribs):
            raise ValueError("mirror mask and rib counts must match")
        grouped = np.zeros(len(ribs), dtype=bool)
        for group in self._mirror_groups(ribs):
            if np.all(qualifying[group]):
                grouped[group] = True
        return grouped

    def _mirror_complete_addition_batch(
        self,
        chosen: Sequence[Rib],
        candidates: Sequence[Rib],
        active: Sequence[Rib],
        limit: int,
    ) -> list[Rib]:
        """Complete at most ``limit`` ranked seeds without exceeding one extra rib.

        With the standard two-seed limit, a first seed that needs a mirror
        partner consumes the batch immediately (two ribs). Only when the first
        seed needs no new partner is the second seed considered; its partner
        may then enlarge the completed batch to three ribs.
        """
        if not self.mirror_axes or not chosen:
            return list(chosen)
        by_key = {candidate.key: candidate for candidate in candidates}
        completed: list[Rib] = []
        completed_keys: set[tuple] = set()
        seed_limit = max(int(limit), 0)
        completed_limit = seed_limit+1 if seed_limit > 0 else 0
        for seed in chosen:
            orbit = [seed]
            frontier = [seed]
            for axis in self.mirror_axes:
                additions = [
                    mirror_rib(
                        rib, axis, self.symmetry_width, self.symmetry_height
                    )
                    for rib in frontier
                ]
                frontier += additions
            for reflected in frontier:
                candidate = by_key.get(reflected.key)
                if candidate is not None and candidate.key not in {rib.key for rib in orbit}:
                    orbit.append(candidate)
            orbit = [
                rib for rib in orbit
                if not collinear_union_covering_ribs(rib, active)
            ]
            new_members = [rib for rib in orbit if rib.key not in completed_keys]
            if len(completed)+len(new_members) > completed_limit:
                self.log.append(
                    "mirror-symmetric candidate group skipped: "
                    f"seed={seed.name}, required_new_ribs={len(new_members)}, "
                    f"completed={len(completed)}, maximum={completed_limit}"
                )
                continue
            for rib in new_members:
                completed.append(rib)
                completed_keys.add(rib.key)
            if len(completed) >= seed_limit:
                break
        if completed:
            self.log.append(
                "mirror-symmetric addition batch: "
                f"axes={list(self.mirror_axes)}, ribs={[rib.name for rib in completed]}"
            )
        return completed

    def analyze(self, ribs: Sequence[Rib], thicknesses: Sequence[float]) -> AnalysisResult:
        self.analysis_count += 1
        return self.model.analyze(ribs, thicknesses)

    def _report_progress(
        self,
        phase: str,
        ribs: Sequence[Rib],
        result: AnalysisResult,
    ) -> None:
        """Emit and retain an immediately visible phase-completion message."""
        message = (
            f"[progress] {phase} is finished, with compliance = "
            f"{float(result.compliance):.7g}, ribs = {len(ribs)}"
        )
        self.log.append(message)
        print(message, flush=True)

    def _new_move_limit(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        step_scale: np.ndarray | None = None,
        initial_global_step: float | None = None,
    ) -> EnhancedMMAMoveLimit:
        settings = self.cfg["algorithm"]
        return EnhancedMMAMoveLimit(
            lower=np.asarray(lower, float),
            upper=np.asarray(upper, float),
            step_scale=None if step_scale is None else np.asarray(step_scale, float),
            initial_global_step=float(
                settings["move_limit_initial"]
                if initial_global_step is None else initial_global_step
            ),
            same_direction_increase=float(settings["move_limit_direction_increase"]),
            oscillation_decrease=float(settings["move_limit_direction_decrease"]),
            unsuccessful_decrease=float(settings["move_limit_unsuccessful_decrease"]),
            maximum_global_step=float(settings["move_limit_maximum_global"]),
            direction_zero_tolerance=float(
                settings.get("move_limit_direction_zero_tolerance", 1.0e-6)
            ),
        )

    def _positive_move_limit_setting(
        self, name: str, default: float
    ) -> float:
        value = float(self.cfg["algorithm"].get(name, default))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return value

    def _outer_objective_rollback_settings(self) -> tuple[float, int, float]:
        """Return validated phase-independent severe-step rollback settings."""
        settings = self.cfg["algorithm"]
        threshold = float(
            settings.get("outer_objective_rollback_threshold", 0.50)
        )
        retries_raw = settings.get("outer_objective_rollback_max_retries", 4)
        retries = int(retries_raw)
        minimum_move = float(
            settings.get("outer_objective_rollback_minimum_move", 1.0e-3)
        )
        contraction = float(settings["move_limit_unsuccessful_decrease"])
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError(
                "outer_objective_rollback_threshold must be finite and nonnegative"
            )
        if isinstance(retries_raw, bool) or retries != retries_raw or retries < 0:
            raise ValueError(
                "outer_objective_rollback_max_retries must be a nonnegative integer"
            )
        if not np.isfinite(minimum_move) or minimum_move <= 0.0:
            raise ValueError(
                "outer_objective_rollback_minimum_move must be finite and positive"
            )
        if not 0.0 < contraction < 1.0:
            raise ValueError(
                "move_limit_unsuccessful_decrease must lie strictly between zero and one"
            )
        return threshold, retries, minimum_move

    @staticmethod
    def _objective_worsening(start: float, trial: float) -> float:
        """Signed relative worsening of a minimization objective."""
        start_value = float(start)
        trial_value = float(trial)
        if not np.isfinite(start_value):
            raise ValueError("accepted outer objective must be finite")
        if not np.isfinite(trial_value):
            return float("inf")
        return (trial_value-start_value)/max(abs(start_value), 1.0e-16)

    @classmethod
    def _severe_objective_response(
        cls, start: float, trial: float, threshold: float
    ) -> tuple[float, bool, bool]:
        """Classify a true trial objective, rejecting every non-finite value."""
        trial_finite = bool(np.isfinite(float(trial)))
        worsening = cls._objective_worsening(start, trial)
        rejected = bool(not trial_finite or worsening > float(threshold))
        return worsening, trial_finite, rejected

    @staticmethod
    def _candidate_design_is_finite(
        candidate: Sequence[float],
        ribs: Sequence[Rib],
        thicknesses: Sequence[float],
    ) -> bool:
        """Return whether a candidate is safe to submit to finite-element analysis."""
        if not np.all(np.isfinite(np.asarray(candidate, float))):
            return False
        if not np.all(np.isfinite(np.asarray(thicknesses, float))):
            return False
        return all(
            np.all(np.isfinite(np.asarray([*rib.p0, *rib.p1, rib.height], float)))
            for rib in ribs
        )

    @staticmethod
    def _contract_outer_move_for_retry(
        move_limit: EnhancedMMAMoveLimit, minimum_move: float
    ) -> bool:
        """Contract a global move, respecting a positive retry floor."""
        if move_limit.global_step <= minimum_move*(1.0+1.0e-12):
            return False
        move_limit.contract()
        move_limit.global_step = max(float(move_limit.global_step), minimum_move)
        return True

    def _coordinate_move_step_scale(self, rib_count: int) -> np.ndarray:
        """Return the endpoint-coordinate scale used by geometry move limits.

        The scale is 0.75 times the local ground-shell cell dimension. With the
        default initial global step of 0.5 and local step of 1.0, the initial
        coordinate half-width is therefore 0.375 shell cells in each direction.
        """
        return np.tile(
            [
                0.75*self.model.dx,
                0.75*self.model.dy,
                0.75*self.model.dx,
                0.75*self.model.dy,
            ],
            int(rib_count),
        )

    @staticmethod
    def volume(ribs: Sequence[Rib], thicknesses: Sequence[float]) -> float:
        return float(sum(r.length * r.height * float(t) for r, t in zip(ribs, thicknesses)))

    def short_rib_length_threshold(self) -> float:
        """Mesh-aware length below which a rib may enter filtering."""
        settings = self.cfg["algorithm"]
        shell_cells = float(settings.get("short_rib_shell_cells", 3.0))
        cell_fraction = float(settings.get("short_rib_cell_fraction", 0.25))
        if shell_cells <= 0.0 or cell_fraction <= 0.0:
            raise ValueError("short-rib length factors must be positive")
        return max(
            shell_cells*min(float(self.model.dx), float(self.model.dy)),
            cell_fraction*float(self.cfg["initial_rib_cell_size"]),
        )

    def short_rib_thickness_factor(self) -> float:
        factor = float(
            self.cfg["algorithm"].get("short_rib_thickness_factor", 5.0)
        )
        if factor <= 1.0:
            raise ValueError("short_rib_thickness_factor must exceed one")
        return factor

    def rationalization_reference_quantile(
        self, relaxation: float, n_rib: int
    ) -> float:
        """Return ``q = 1/n_rib + rho`` for the ``tref`` quantile."""
        rib_count = int(n_rib)
        if rib_count <= 0 or rib_count != n_rib:
            raise ValueError("n_rib must be a positive integer")
        quantile = 1.0/rib_count+float(relaxation)
        if not 0.0 < quantile < 1.0:
            raise ValueError(
                "rationalization reference quantile q=1/n_rib+rho must "
                f"lie strictly between zero and one; got q={quantile:.7g} "
                f"for n_rib={rib_count}, rho={float(relaxation):.7g}"
            )
        return quantile

    def _feasible_start(self, ribs: Sequence[Rib], initial: Sequence[float] | None) -> np.ndarray:
        x = np.full(len(ribs), self.t0) if initial is None else np.clip(np.asarray(initial, float), self.t_lower, self.t_upper)
        coefficients = np.array([r.length * r.height for r in ribs], float)
        available = self.volume_bound - self.t_lower * coefficients.sum()
        if available < 0:
            raise ValueError("volume bound is below the all-lower-bound rib volume")
        excess = np.maximum(x - self.t_lower, 0.0)
        used = float(coefficients @ excess)
        if used > available and used > 0:
            x = self.t_lower + excess * available / used
        return x

    def size(self, ribs: Sequence[Rib], initial: Sequence[float] | None = None, maxiter: int | None = None) -> tuple[np.ndarray, AnalysisResult]:
        if not ribs:
            raise ValueError("cannot size an empty rib set")
        self.sizing_termination_reason = None
        (
            rollback_threshold,
            rollback_max_retries,
            rollback_minimum_move,
        ) = self._outer_objective_rollback_settings()
        variable_map = build_mirror_variable_map(
            ribs, self.mirror_axes, self.symmetry_width, self.symmetry_height
        )
        full_x = self._feasible_start(ribs, initial)
        x = variable_map.reduce_thicknesses(full_x)
        full_x = variable_map.expand_thicknesses(x)
        full_coeff = np.array([r.length * r.height for r in ribs], float)
        coeff = variable_map.reduce_thickness_gradient(full_coeff)
        if maxiter is None:
            maxiter = int(self.cfg["algorithm"]["sizing_max_iterations"])
        objective_tolerance = float(self.cfg["algorithm"]["sca_objective_tolerance"])
        design_tolerance = float(
            self.cfg["algorithm"].get("sca_design_tolerance", 0.001)
        )
        design_guard_tolerance = float(
            self.cfg["algorithm"].get("sca_design_guard_tolerance", 0.010)
        )
        constraint_tolerance = float(self.cfg["algorithm"]["sca_constraint_tolerance"])
        consecutive_required = int(
            self.cfg["algorithm"].get("sca_consecutive_convergence_steps", 2)
        )
        current = self.analyze(ribs, full_x)
        initial_volume = self.volume(ribs, full_x)
        initial_feasible = initial_volume <= self.volume_bound * (
            1.0+constraint_tolerance
        )
        best_x: np.ndarray | None = full_x.copy() if initial_feasible else None
        best_result: AnalysisResult | None = current if initial_feasible else None
        best_outer: int | None = 0 if initial_feasible else None
        move_limit = self._new_move_limit(
            np.full(len(x), self.t_lower), np.full(len(x), self.t_upper)
        )
        violation = max(initial_volume/self.volume_bound-1.0, 0.0)
        move_lower, move_upper = move_limit.update(
            x, current.compliance, violation, constraint_tolerance
        )

        lower_global = np.full(len(x), self.t_lower)
        upper_global = np.full(len(x), self.t_upper)
        consecutive_converged = 0
        for outer in range(1, int(maxiter) + 1):
            # One FEA has already supplied current C and displacement. Obtain
            # sensitivities and build a reciprocal convex approximation:
            # C~ = Ck + sum a_i(1/t_i - 1/t_ki), a_i=-g_i*t_ki^2 >= 0.
            full_gradient = self.model.compliance_gradient(ribs, full_x, current)
            gradient = variable_map.reduce_thickness_gradient(full_gradient)
            a = np.maximum(-gradient, 0.0) * x**2
            # Eq. (9) is the thickness-only specialization of the same
            # separable dual problem used for Eq. (7): the geometry-gradient
            # and coordinate arrays are empty, leaving the closed-form
            # t_i(lambda)=sqrt(a_i/(lambda*v_i)) update with box move limits.
            response_trials: list[dict] = []
            accepted = False
            rollback_failure_reason = "unknown"
            for response_retry in range(rollback_max_retries+1):
                if response_retry:
                    move_lower, move_upper = move_limit.current_bounds(x)
                candidate = solve_geometry_convex_subproblem(
                    x,
                    a,
                    np.empty(0),
                    coeff,
                    float(coeff@x),
                    self.volume_bound,
                    1.0,  # unused when no coordinate variables are present
                    max(current.compliance, 1.0e-16),
                    np.empty(0),
                    move_lower,
                    move_upper,
                )
                full_candidate = variable_map.expand_thicknesses(candidate)
                full_candidate = self._feasible_start(ribs, full_candidate)
                candidate = variable_map.reduce_thicknesses(full_candidate)
                full_candidate = variable_map.expand_thicknesses(candidate)
                move_global_trial = float(move_limit.global_step)
                if not self._candidate_design_is_finite(
                    candidate, ribs, full_candidate
                ):
                    response_trials.append({
                        "retry": int(response_retry),
                        "objective": None,
                        "objective_relative_change_signed": None,
                        "objective_finite": False,
                        "design_finite": False,
                        "fea_performed": False,
                        "move_global": move_global_trial,
                        "accepted": False,
                        "rejection_reasons": ["nonfinite_candidate_design"],
                    })
                    if response_retry >= rollback_max_retries:
                        rollback_failure_reason = "retry_budget_exhausted"
                        break
                    if not self._contract_outer_move_for_retry(
                        move_limit, rollback_minimum_move
                    ):
                        rollback_failure_reason = "minimum_move_reached"
                        break
                    continue
                trial = self.analyze(ribs, full_candidate)
                (
                    signed_trial_change,
                    trial_objective_finite,
                    severe_worsening,
                ) = self._severe_objective_response(
                    current.compliance, trial.compliance, rollback_threshold
                )
                response_trials.append({
                    "retry": int(response_retry),
                    "objective": (
                        float(trial.compliance) if trial_objective_finite else None
                    ),
                    "objective_relative_change_signed": (
                        float(signed_trial_change) if trial_objective_finite else None
                    ),
                    "objective_finite": trial_objective_finite,
                    "design_finite": True,
                    "fea_performed": True,
                    "move_global": move_global_trial,
                    "accepted": not severe_worsening,
                })
                if not severe_worsening:
                    accepted = True
                    break
                self.log.append(
                    "sizing severe-objective trial rejected: "
                    f"outer={outer}, retry={response_retry}, "
                    f"dC={100*signed_trial_change:.4f}%, "
                    f"move_global={move_global_trial:.5g}"
                )
                if response_retry >= rollback_max_retries:
                    rollback_failure_reason = "retry_budget_exhausted"
                    break
                if not self._contract_outer_move_for_retry(
                    move_limit, rollback_minimum_move
                ):
                    rollback_failure_reason = "minimum_move_reached"
                    break

            if not accepted:
                self.sizing_termination_reason = "objective_rollback_failed"
                current_feasible = self.volume(
                    ribs, full_x
                ) <= self.volume_bound*(1.0+constraint_tolerance)
                self.sizing_history.append({
                    "outer": int(outer),
                    "compliance": float(current.compliance),
                    "feasible": bool(current_feasible),
                    "accepted": False,
                    "termination_reason": "objective_rollback_failed",
                    "rollback_failure_reason": rollback_failure_reason,
                    "response_trials": response_trials,
                    "response_retry_count": int(len(response_trials)-1),
                    "best_feasible_outer": best_outer,
                    "best_feasible_compliance": (
                        None if best_result is None else float(best_result.compliance)
                    ),
                    "is_best_feasible": bool(best_outer == outer),
                })
                self.log.append(
                    "sizing SCA stopped: no acceptable severe-objective step "
                    f"after {len(response_trials)} trials at outer={outer}; "
                    "the accepted outer start was restored"
                )
                if best_result is None or best_x is None:
                    returned_x, returned_result = full_x, current
                else:
                    returned_x, returned_result = best_x, best_result
                self._report_progress("sizing optimization", ribs, returned_result)
                return returned_x, returned_result

            relative_change = abs(trial.compliance - current.compliance) / max(abs(current.compliance), 1.0e-16)
            design_change = maximum_normalized_design_change(
                x, candidate, lower_global, upper_global
            )
            feasible = self.volume(ribs, full_candidate) <= self.volume_bound * (1.0 + constraint_tolerance)
            if feasible and (
                best_result is None or trial.compliance < best_result.compliance
            ):
                best_x = full_candidate.copy()
                best_result = trial
                best_outer = outer
            x, full_x, current = candidate, full_candidate, trial
            violation = max(self.volume(ribs, full_x)/self.volume_bound-1.0, 0.0)
            move_lower, move_upper = move_limit.update(
                x, current.compliance, violation, constraint_tolerance
            )
            step_converged = sca_step_converged(
                feasible,
                relative_change,
                design_change,
                objective_tolerance,
                design_tolerance,
                design_guard_tolerance,
            )
            consecutive_converged = (
                consecutive_converged+1 if step_converged else 0
            )
            self.sizing_history.append({
                "outer": int(outer),
                "compliance": float(trial.compliance),
                "feasible": bool(feasible),
                "accepted": True,
                "response_trials": response_trials,
                "response_retry_count": int(len(response_trials)-1),
                "best_feasible_outer": best_outer,
                "best_feasible_compliance": (
                    None if best_result is None else float(best_result.compliance)
                ),
                "is_best_feasible": bool(best_outer == outer),
            })
            if consecutive_converged >= consecutive_required:
                self.sizing_termination_reason = "converged"
                self.log.append(
                    f"sizing SCA converged: outer={outer}, "
                    f"dC={100*relative_change:.4f}%, dx={100*design_change:.4f}%, "
                    f"volume={self.volume(ribs,full_x):.7g}"
                )
                if best_result is None or best_x is None:
                    self.log.append(
                        "sizing SCA warning: no feasible true-FEA incumbent; "
                        "returning the last iterate"
                    )
                    returned_x, returned_result = full_x, current
                else:
                    returned_x, returned_result = best_x, best_result
                if best_outer is not None and best_outer != outer:
                    self.log.append(
                        "sizing SCA returned best feasible true-FEA incumbent: "
                        f"outer={best_outer}, C={best_result.compliance:.7g}"
                    )
                self._report_progress("sizing optimization", ribs, returned_result)
                return returned_x, returned_result
        self.log.append(f"sizing SCA warning: outer iteration limit {maxiter} reached")
        self.sizing_termination_reason = "iteration_limit"
        if best_result is None or best_x is None:
            self.log.append(
                "sizing SCA warning: no feasible true-FEA incumbent; "
                "returning the last iterate"
            )
            returned_x, returned_result = full_x, current
        else:
            returned_x, returned_result = best_x, best_result
        if best_outer is not None and best_outer != int(maxiter):
            self.log.append(
                "sizing SCA returned best feasible true-FEA incumbent: "
                f"outer={best_outer}, C={best_result.compliance:.7g}"
            )
        self._report_progress("sizing optimization", ribs, returned_result)
        return returned_x, returned_result

    def filter(
        self,
        ribs: list[Rib],
        thicknesses: np.ndarray,
        pre: AnalysisResult,
        threshold_ratios: Sequence[float] | None = None,
        reference_compliance: float | None = None,
    ) -> tuple[list[Rib], np.ndarray, AnalysisResult]:
        tolerance = float(self.cfg["algorithm"]["filter_tolerance"])
        reference = (
            float(pre.compliance)
            if reference_compliance is None
            else float(reference_compliance)
        )
        ratios = threshold_ratios
        if ratios is None:
            ratios = self.cfg["algorithm"].get("filter_threshold_ratios")
        if ratios is None:
            max_decade = int(self.cfg["algorithm"]["filter_max_decade"])
            ratios = [10.0 ** (-decade) for decade in range(2, max_decade + 1)]
        for ratio in map(float, ratios):
            threshold = ratio * self.t0
            lengths = np.asarray([rib.length for rib in ribs], float)
            thin = thicknesses < threshold
            short = lengths < self.short_rib_length_threshold()
            short_and_light = short & (
                thicknesses < self.short_rib_thickness_factor()*threshold
            )
            remove = self._mirror_group_removal_mask(
                ribs, thin | short_and_light
            )
            keep = ~remove
            if keep.all() or not keep.any():
                continue
            trial_ribs = [r for r, flag in zip(ribs, keep) if flag]
            trial_x, trial_result = self.size(trial_ribs, thicknesses[keep])
            increase = (trial_result.compliance - reference) / reference
            if increase <= tolerance:
                self.log.append(
                    f"filter accepted tau={threshold:g}: {len(ribs)} -> "
                    f"{len(trial_ribs)}, dC_ref={100*increase:.3f}%, "
                    f"thin={int(np.count_nonzero(thin))}, "
                    f"short_light={int(np.count_nonzero(short_and_light & ~thin))}, "
                    f"removed={[rib.name for rib, flag in zip(ribs, remove) if flag]}"
                )
                return trial_ribs, trial_x, trial_result
        return ribs, thicknesses, pre

    def filter_until_stable(
        self,
        ribs: list[Rib],
        thicknesses: np.ndarray,
        current: AnalysisResult,
        threshold_ratios: Sequence[float] | None = None,
    ) -> tuple[list[Rib], np.ndarray, AnalysisResult, int]:
        """Repeat sizing/filtering until no additional rib can be deleted."""
        deletion_rounds = 0
        # Every deletion attempt in this repeated filtering phase is checked
        # against the same compliance measured at phase entry. This prevents
        # the allowed degradation from accumulating over successive passes.
        reference_compliance = float(current.compliance)
        for _ in range(len(ribs) + 1):
            before = {r.key for r in ribs}
            if threshold_ratios is None:
                ribs, thicknesses, current = self.filter(
                    ribs,
                    thicknesses,
                    current,
                    reference_compliance=reference_compliance,
                )
            else:
                ribs, thicknesses, current = self.filter(
                    ribs,
                    thicknesses,
                    current,
                    threshold_ratios,
                    reference_compliance=reference_compliance,
                )
            if {r.key for r in ribs} == before:
                break
            deletion_rounds += 1
        self.log.append(
            f"sizing/filtering converged after {deletion_rounds} deletion rounds; "
            f"ribs={len(ribs)}, C={current.compliance:.7g}"
        )
        self._report_progress("filtering", ribs, current)
        return ribs, thicknesses, current, deletion_rounds

    def _select_addition_candidates(
        self,
        candidates: Sequence[Rib],
        active: Sequence[Rib],
        current: AnalysisResult,
        limit: int = 2,
    ) -> tuple[list[Rib], list[Rib]]:
        """Rank candidates and scan until a strong nonconflicting batch is found.

        Every candidate is ranked directly by its frozen-field stiffness
        contribution per added rib volume. Fully covered or mutually
        overlapping candidates are skipped without consuming a batch slot.
        When one collinear candidate completely covers another, the covering
        longer candidate is preferred for manufacturability regardless of
        score gap.
        """
        workers = max(
            1, int(self.cfg["algorithm"].get("sensitivity_workers", 1))
        )
        def score(candidate: Rib) -> tuple[float, Rib]:
            return (
                self.model.candidate_efficiency(candidate, self.t0, current),
                candidate,
            )

        if workers > 1 and len(candidates) > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                scored = list(pool.map(score, candidates))
        else:
            scored = [score(candidate) for candidate in candidates]
        scored.sort(
            key=lambda item: (
                -item[0] if np.isfinite(item[0]) else np.inf,
                -item[1].length,
                item[1].name,
                item[1].key,
            )
        )
        inspected: list[Rib] = []
        chosen: list[Rib] = []
        chosen_factors: list[float] = []
        minimum_factor_ratio = float(
            self.cfg["algorithm"].get(
                "addition_factor_min_ratio",
                self.cfg["algorithm"].get(
                    "addition_second_factor_min_ratio", 0.70
                ),
            )
        )
        if not 0.0 <= minimum_factor_ratio <= 1.0:
            raise ValueError("addition_factor_min_ratio must be in [0,1]")
        eligible_reference_scores = [
            (float(factor), candidate)
            for factor, candidate in scored
            if (
                np.isfinite(factor)
                and factor > 0.0
                and not collinear_union_covering_ribs(candidate, active)
            )
        ]
        positive_factors = [factor for factor, _ in eligible_reference_scores]
        maximum_factor = max(positive_factors, default=-np.inf)
        if positive_factors:
            self.log.append(
                f"member candidate eligible stiffness-per-volume score reference: "
                f"eligible={len(eligible_reference_scores)} of {len(scored)}, "
                f"maximum={maximum_factor:.7g}, "
                f"minimum selection ratio={minimum_factor_ratio:.4f}, "
                f"minimum score={minimum_factor_ratio*maximum_factor:.7g}"
            )
        else:
            self.log.append(
                "member candidate scan stopped: no positive uncovered "
                "candidate is eligible for the stiffness-per-volume score reference"
            )
            return inspected, chosen
        for factor, candidate in scored:
            if len(chosen) >= max(int(limit), 0):
                break
            inspected.append(candidate)
            self.log.append(
                f"member candidate ranked: {candidate.name}, L={candidate.length:.7g}, "
                f"stiffness_per_volume_score={factor:.7g}"
            )
            if not np.isfinite(factor) or factor <= 0.0:
                self.log.append(
                    "member candidate scan stopped: no positive "
                    "stiffness-per-volume score remains"
                )
                break
            ratio_to_maximum = float(factor)/maximum_factor
            if ratio_to_maximum < minimum_factor_ratio:
                self.log.append(
                    f"member candidate scan stopped: {candidate.name} score ratio="
                    f"{ratio_to_maximum:.4f} < {minimum_factor_ratio:.4f} relative "
                    f"to eligible maximum {maximum_factor:.7g}; "
                    "no later candidate is eligible"
                )
                break

            covering_ribs = collinear_union_covering_ribs(candidate, active)
            if covering_ribs:
                covering_names = [rib.name for rib in covering_ribs]
                if len(covering_ribs) == 1:
                    coverage_description = f"existing rib {covering_names[0]}"
                else:
                    coverage_description = (
                        f"union of existing ribs {covering_names}"
                    )
                self.log.append(
                    f"member candidate skipped: {candidate.name} is fully "
                    f"covered by {coverage_description}"
                )
                continue

            overlap_index = next(
                (
                    index for index, selected in enumerate(chosen)
                    if collinear_overlap(candidate, selected)
                ),
                None,
            )
            if overlap_index is not None:
                selected = chosen[overlap_index]
                selected_factor = chosen_factors[overlap_index]
                relative_factor_difference = abs(
                    float(factor)-selected_factor
                )/max(abs(float(factor)), abs(selected_factor))
                candidate_covers_selected = collinear_covered(
                    selected, candidate
                )
                if (
                    candidate_covers_selected
                    and candidate.length > selected.length
                ):
                    chosen[overlap_index] = candidate
                    chosen_factors[overlap_index] = float(factor)
                    self.log.append(
                        f"member candidate preferred for length: {candidate.name} "
                        f"(L={candidate.length:.7g}, score={factor:.7g}) replaces "
                        f"overlapping collinear {selected.name} "
                        f"(L={selected.length:.7g}, score={selected_factor:.7g}); "
                        f"longer rib completely covers shorter rib and is "
                        f"preferred regardless of "
                        f"score difference={100*relative_factor_difference:.3f}%"
                    )
                else:
                    selected_covers_candidate = collinear_covered(
                        candidate, selected
                    )
                    overlap_reason = (
                        "is completely covered by"
                        if selected_covers_candidate
                        else "partially overlaps"
                    )
                    self.log.append(
                        f"member candidate skipped: {candidate.name} "
                        f"{overlap_reason} preferred rib {selected.name} "
                        f"(score difference={100*relative_factor_difference:.3f}%, "
                        f"lengths={candidate.length:.7g}/{selected.length:.7g})"
                    )
                continue

            chosen.append(candidate)
            chosen_factors.append(float(factor))
            self.log.append(
                f"member candidate accepted for batch: {candidate.name}, "
                f"stiffness_per_volume_score={factor:.7g}"
            )
        chosen = self._mirror_complete_addition_batch(
            chosen, candidates, active, limit
        )
        return inspected, chosen

    def adapt(self, ribs: list[Rib], thicknesses: np.ndarray, current: AnalysisResult, candidates: Sequence[Rib]) -> tuple[list[Rib], np.ndarray, AnalysisResult]:
        settings = self.cfg["algorithm"]
        max_iterations = int(settings["active_set_max_iterations"])
        minimum_sizing_improvement = float(settings["addition_sizing_improvement_min"])
        minimum_cycle_improvement = float(settings["active_cycle_improvement_min"])
        ribs, thicknesses, current, _ = self.filter_until_stable(ribs, thicknesses, current)
        self.active_history.append(
            Stage("filtering_converged_0", list(ribs), thicknesses.copy(), current.compliance, self.analysis_count,
                  "initial sizing/filtering convergence")
        )
        attempted_keys: set[tuple] = set()
        for iteration in range(max_iterations):
            before_keys = {r.key for r in ribs}
            compliance_before = current.compliance
            untried = [candidate for candidate in candidates if candidate.key not in attempted_keys]
            if not untried:
                self.log.append("active set converged: no untried candidate")
                break
            inspected, chosen = self._select_addition_candidates(
                untried,
                ribs,
                current,
                limit=int(settings.get("additions_per_iteration", 2)),
            )
            attempted_keys.update(candidate.key for candidate in chosen)
            if not chosen:
                self.log.append(
                    "active set converged: no valid uncovered candidate remains"
                )
                break
            replaced_indices = {
                index
                for index, active_rib in enumerate(ribs)
                if any(
                    collinear_covered(active_rib, candidate)
                    for candidate in chosen
                )
            }
            retained_mask = np.array(
                [index not in replaced_indices for index in range(len(ribs))],
                dtype=bool,
            )
            replaced_names = [ribs[index].name for index in sorted(replaced_indices)]
            if replaced_names:
                self.log.append(
                    "member candidate replacement: removed fully covered existing "
                    f"ribs={replaced_names}, added={[candidate.name for candidate in chosen]}"
                )
            trial_ribs = [
                rib for rib, retain in zip(ribs, retained_mask) if retain
            ] + chosen
            trial_x0 = np.r_[
                thicknesses[retained_mask], np.full(len(chosen), self.t0)
            ]
            trial_x, trial_result = self.size(trial_ribs, trial_x0)
            sizing_improvement = (compliance_before-trial_result.compliance)/compliance_before
            rejected = sizing_improvement < minimum_sizing_improvement
            self.log.append(
                f"member-addition sizing round {iteration+1}: added={len(chosen)}, "
                f"replaced={len(replaced_indices)}, "
                f"C={trial_result.compliance:.7g}, decrease={100*sizing_improvement:.3f}%"
            )
            self.active_history.append(
                Stage(
                    f"member_addition_sizing_round_{iteration+1}{'_rejected' if rejected else ''}",
                    list(trial_ribs),trial_x.copy(),trial_result.compliance,self.analysis_count,
                    f"added={len(chosen)}, replaced={len(replaced_indices)}, "
                    f"sizing decrease={100*sizing_improvement:.3f}%"
                    +(", rejected and restored" if rejected else ""),
                )
            )
            if rejected:
                self.log.append(
                    f"active set converged: post-addition sizing decrease "
                    f"{100*sizing_improvement:.3f}% < {100*minimum_sizing_improvement:.3f}%; "
                    f"reverted addition and terminated"
                )
                break

            # The addition has passed the 1% sizing-only check. Accept it,
            # then repeat filtering/sizing until no further deletion occurs.
            ribs,thicknesses,current=trial_ribs,trial_x,trial_result
            ribs,thicknesses,current,deletion_rounds=self.filter_until_stable(
                ribs,thicknesses,current
            )
            retained_new = len({r.key for r in ribs} - before_keys)
            filtered_improvement=(compliance_before-current.compliance)/compliance_before
            self.log.append(
                f"post-addition filtering round {iteration+1}: retained_new={retained_new}, "
                f"filter_rounds={deletion_rounds}, C={current.compliance:.7g}, "
                f"decrease={100*filtered_improvement:.3f}%"
            )
            self.active_history.append(
                Stage(
                    f"post_addition_filtering_round_{iteration+1}",list(ribs),thicknesses.copy(),
                    current.compliance,self.analysis_count,
                    f"added={len(chosen)}, replaced={len(replaced_indices)}, "
                    f"retained={retained_new}, filter rounds={deletion_rounds}, "
                    f"decrease={100*filtered_improvement:.3f}%",
                )
            )
            if retained_new == 0:
                self.log.append(
                    "active set converged: no newly added rib retained after "
                    "filtering; entering geometry optimization"
                )
                break
            if filtered_improvement < minimum_cycle_improvement:
                self.log.append(
                    f"active set converged: full addition/filtering cycle "
                    f"decrease {100*filtered_improvement:.3f}% < "
                    f"{100*minimum_cycle_improvement:.3f}%; entering geometry optimization"
                )
                break
        else:
            self.log.append(f"active set stopped at configured limit {max_iterations}")
        self._report_progress("adaptive optimization", ribs, current)
        return ribs, thicknesses, current

    def optimize_geometry(
        self,
        ribs: list[Rib],
        thicknesses: np.ndarray,
        current: AnalysisResult | None,
        coordinate_bounds: np.ndarray | None = None,
        max_iterations_override: int | None = None,
        initial_move_step: float | None = None,
        iteration_history: list[dict] | None = None,
    ) -> tuple[list[Rib], np.ndarray, AnalysisResult]:
        final = self._optimize_geometry(
            ribs,
            thicknesses,
            current,
            coordinate_bounds,
            max_iterations_override,
            initial_move_step,
            iteration_history,
        )
        self._report_progress("geometry optimization", final[0], final[2])
        return final

    def _optimize_geometry(
        self,
        ribs: list[Rib],
        thicknesses: np.ndarray,
        current: AnalysisResult | None,
        coordinate_bounds: np.ndarray | None = None,
        max_iterations_override: int | None = None,
        initial_move_step: float | None = None,
        iteration_history: list[dict] | None = None,
    ) -> tuple[list[Rib], np.ndarray, AnalysisResult]:
        """Sequential convex geometry/thickness optimization.

        If ``current`` is omitted, Eq. (7) starts by analyzing the supplied
        design so that its first convex approximation uses the correct
        reduced-topology displacement field. Each outer iteration solves the
        approximation without FEA; its output is analyzed as the next outer
        iterate. A configurable true-response check contracts the move limit
        and resolves the same approximation when a trial is materially worse;
        the best feasible true-FEA incumbent is always returned.
        """
        settings = self.cfg["algorithm"]
        self.geometry_termination_reason = None
        max_iterations = int(settings["geometry_max_iterations"] if max_iterations_override is None else max_iterations_override)
        if current is None:
            current = self.analyze(ribs, thicknesses)
        if max_iterations <= 0:
            self.geometry_termination_reason = "disabled"
            return ribs, thicknesses, current
        n = len(ribs)
        variable_map = build_mirror_variable_map(
            ribs, self.mirror_axes, self.symmetry_width, self.symmetry_height
        )
        nt = variable_map.thickness_count
        coordinates = np.array([[*r.p0, *r.p1] for r in ribs], float)
        x = np.r_[
            variable_map.reduce_thicknesses(thicknesses),
            variable_map.reduce_coordinates(coordinates.ravel()),
        ]
        fd_step = float(settings["geometry_fd_fraction"]) * float(self.cfg["initial_rib_cell_size"])
        proximal = float(settings["geometry_sca_proximal"])
        objective_tolerance = float(settings["sca_objective_tolerance"])
        design_tolerance = float(settings.get("sca_design_tolerance", 0.001))
        design_guard_tolerance = float(
            settings.get("sca_design_guard_tolerance", 0.010)
        )
        constraint_tolerance = float(settings["sca_constraint_tolerance"])
        consecutive_required = int(
            settings.get("sca_consecutive_convergence_steps", 2)
        )
        (
            rollback_threshold,
            rollback_max_retries,
            rollback_minimum_move,
        ) = self._outer_objective_rollback_settings()

        def unpack(x: np.ndarray) -> tuple[np.ndarray, list[Rib]]:
            t = variable_map.expand_thicknesses(x[:nt])
            p = variable_map.expand_coordinates(x[nt:]).reshape(n, 4)
            moved = [Rib(tuple(q[:2]),tuple(q[2:]),r.height,r.name,r.segments) for q,r in zip(p,ribs)]
            return t,moved

        def volume_data(x:np.ndarray)->tuple[float,np.ndarray]:
            t,moved=unpack(x)
            thickness_jac=np.zeros(n)
            coordinate_jac=np.zeros(4*n)
            volume=0.0
            for i,(te,r) in enumerate(zip(t,moved)):
                p0=np.asarray(r.p0);p1=np.asarray(r.p1);d=p1-p0;L=max(np.linalg.norm(d),1e-12)
                volume+=L*r.height*te; thickness_jac[i]=L*r.height
                g=r.height*te*d/L
                coordinate_jac[4*i:4*i+2]=-g
                coordinate_jac[4*i+2:4*i+4]=g
            return float(volume),np.r_[
                variable_map.reduce_thickness_gradient(thickness_jac),
                variable_map.reduce_coordinate_gradient(coordinate_jac),
            ]
        if coordinate_bounds is None:
            coordinate_bounds = np.array([
                [
                    [0.0,self.model.width],
                    [0.0,self.model.height],
                    [0.0,self.model.width],
                    [0.0,self.model.height],
                ] for q in coordinates
            ],float)

        reduced_coordinate_bounds = variable_map.reduce_coordinate_bounds(
            coordinate_bounds
        )
        global_lower = np.r_[
            np.full(nt, self.t_lower), reduced_coordinate_bounds[:, 0]
        ]
        global_upper = np.r_[
            np.full(nt, self.t_upper), reduced_coordinate_bounds[:, 1]
        ]
        coordinate_step_scale = variable_map.reduce_coordinate_scale(
            self._coordinate_move_step_scale(n)
        )
        move_limit = self._new_move_limit(
            global_lower, global_upper,
            np.r_[np.full(nt, np.nan), coordinate_step_scale],
            initial_global_step=initial_move_step,
        )
        initial_t, initial_ribs = unpack(x)
        projected_coordinates = np.array([
            [*rib.p0, *rib.p1] for rib in initial_ribs
        ])
        if (
            not np.allclose(initial_t, thicknesses, rtol=0.0, atol=1.0e-12)
            or not np.allclose(projected_coordinates, coordinates, rtol=0.0, atol=1.0e-10)
        ):
            current = self.analyze(initial_ribs, initial_t)
        initial_violation = max(self.volume(initial_ribs, initial_t)/self.volume_bound-1.0, 0.0)
        move_lower, move_upper = move_limit.update(
            x, current.compliance, initial_violation, constraint_tolerance
        )
        initial_feasible = initial_violation <= constraint_tolerance
        best_x: np.ndarray | None = x.copy() if initial_feasible else None
        best_result: AnalysisResult | None = current if initial_feasible else None
        best_outer: int | None = 0 if initial_feasible else None
        consecutive_converged = 0
        for outer in range(1, max_iterations + 1):
            t, moved = unpack(x)
            gt_full = self.model.compliance_gradient(moved, t, current)
            gp_full = self.model.geometry_gradient(moved, t, current, fd_step).ravel()
            a = variable_map.reduce_thickness_gradient(
                np.maximum(-gt_full, 0.0)*t**2
            )
            gp = variable_map.reduce_coordinate_gradient(gp_full)
            volume_k, volume_gradient = volume_data(x)
            scale_p = variable_map.reduce_coordinate_quadratic_scale(np.tile(
                [self.model.width,self.model.height,self.model.width,self.model.height], n
            ))
            scale_c = max(current.compliance, 1.0e-16)

            def approximation(y:np.ndarray)->float:
                dy=y-x; safe=np.maximum(y[:nt],self.t_lower)
                reciprocal=np.sum(a*(1.0/safe-1.0/x[:nt]))
                dp=dy[nt:]
                return float(reciprocal+gp@dp+0.5*proximal*scale_c*np.sum((dp/scale_p)**2))
            def approximation_jac(y:np.ndarray)->np.ndarray:
                out=np.zeros_like(y);safe=np.maximum(y[:nt],self.t_lower)
                out[:nt]=-a/safe**2;dp=y[nt:]
                out[nt:]=gp+proximal*scale_c*dp/scale_p**2
                return out

            # Solve the Eq. (7) convex approximation. If a coordinate move
            # makes one or more ribs geometrically invalid, freeze only those
            # rib positions at the current outer design and resolve the same
            # FEA-free approximation. Other rib coordinates and every
            # thickness variable remain free. Local geometry freezing does not
            # contract the global move limit; contraction occurs only when the
            # generic severe-objective safeguard rejects the true FEA response.
            response_trials: list[dict] = []
            accepted = False
            response_retry = 0
            severe_retries_used = 0
            rollback_failure_reason = "unknown"
            while True:
                if response_retry:
                    move_lower, move_upper = move_limit.current_bounds(x)
                inner_lower=move_lower.copy()
                inner_upper=move_upper.copy()
                frozen_geometry_indices: set[int] = set()
                frozen_geometry_reasons: dict[int, set[str]] = {}
                invalid = False
                for inner in range(1, n+2):
                    candidate = solve_geometry_convex_subproblem(
                        x,
                        a,
                        gp,
                        volume_gradient,
                        volume_k,
                        self.volume_bound,
                        proximal,
                        scale_c,
                        scale_p,
                        inner_lower,
                        inner_upper,
                    )
                    candidate_t,candidate_ribs=unpack(candidate)
                    candidate_t=self._feasible_start(candidate_ribs,candidate_t)
                    candidate[:nt]=variable_map.reduce_thicknesses(candidate_t)
                    candidate_t,candidate_ribs=unpack(candidate)
                    freeze_reasons = geometry_move_freeze_reasons(
                        moved,
                        candidate_ribs,
                        0.25*float(self.cfg["initial_rib_cell_size"]),
                    )
                    if not freeze_reasons:
                        invalid = False
                        break
                    invalid = True
                    new_indices = set(freeze_reasons)-frozen_geometry_indices
                    if not new_indices:
                        # This can only occur for a pre-existing invalid baseline
                        # or a bound-tolerance artifact. Preserve all current rib
                        # positions, retain the optimized thicknesses, and restore
                        # exact volume feasibility without contracting Gstep.
                        candidate[nt:]=x[nt:]
                        candidate_t,candidate_ribs=unpack(candidate)
                        candidate_t=self._feasible_start(candidate_ribs,candidate_t)
                        candidate[:nt]=variable_map.reduce_thicknesses(candidate_t)
                        candidate_t,candidate_ribs=unpack(candidate)
                        invalid=False
                        self.log.append(
                            "geometry convex inner local freeze fallback: "
                            "all rib positions retained at current outer design"
                        )
                        break
                    for index in sorted(new_indices):
                        variables = np.unique(
                            variable_map.coordinate_variable[4*index:4*index+4]
                        )
                        variables = variables[variables >= 0]
                        coordinate_indices = nt+variables
                        inner_lower[coordinate_indices]=x[coordinate_indices]
                        inner_upper[coordinate_indices]=x[coordinate_indices]
                        frozen_geometry_reasons.setdefault(index,set()).update(
                            freeze_reasons[index]
                        )
                    frozen_geometry_indices.update(new_indices)
                    frozen_names=[ribs[index].name for index in sorted(new_indices)]
                    self.log.append(
                        f"geometry convex inner local freeze: inner={inner}, "
                        f"ribs={frozen_names}, move_global unchanged="
                        f"{move_limit.global_step:.5g}"
                    )
                if invalid:
                    # Defensive final fallback; the current geometry is the valid
                    # baseline, while thicknesses still use the inner optimum.
                    candidate[nt:]=x[nt:]
                    candidate_t,candidate_ribs=unpack(candidate)
                    candidate_t=self._feasible_start(candidate_ribs,candidate_t)
                    candidate[:nt]=variable_map.reduce_thicknesses(candidate_t)
                    candidate_t,candidate_ribs=unpack(candidate)

                move_global_trial = float(move_limit.global_step)
                candidate_volume=self.volume(candidate_ribs,candidate_t)
                candidate_design_finite = bool(
                    self._candidate_design_is_finite(
                        candidate, candidate_ribs, candidate_t
                    )
                    and np.isfinite(candidate_volume)
                )
                if not candidate_design_finite:
                    severe_retries_used += 1
                    response_trials.append({
                        "retry": int(response_retry),
                        "compliance": None,
                        "objective_relative_change_signed": None,
                        "objective_finite": False,
                        "design_finite": False,
                        "fea_performed": False,
                        "move_global": move_global_trial,
                        "accepted": False,
                        "rejection_reasons": ["nonfinite_candidate_design"],
                        "severe_retries_used": int(severe_retries_used),
                    })
                    if severe_retries_used > rollback_max_retries:
                        rollback_failure_reason = (
                            "severe_retry_budget_exhausted"
                        )
                        break
                    if not self._contract_outer_move_for_retry(
                        move_limit, rollback_minimum_move
                    ):
                        rollback_failure_reason = "minimum_move_reached"
                        break
                    response_retry += 1
                    continue
                trial=self.analyze(candidate_ribs,candidate_t)
                (
                    signed_trial_change,
                    trial_objective_finite,
                    severe_worsening,
                ) = self._severe_objective_response(
                    current.compliance, trial.compliance, rollback_threshold
                )
                trial_violation=max(
                    candidate_volume/self.volume_bound-1.0,0.0
                )
                rejected = bool(severe_worsening)
                rejection_reasons = (
                    ["severe_objective"] if severe_worsening else []
                )
                if severe_worsening:
                    severe_retries_used += 1
                response_trials.append({
                    "retry": int(response_retry),
                    "compliance": (
                        float(trial.compliance) if trial_objective_finite else None
                    ),
                    "objective_relative_change_signed": (
                        float(signed_trial_change) if trial_objective_finite else None
                    ),
                    "objective_finite": trial_objective_finite,
                    "design_finite": True,
                    "fea_performed": True,
                    "move_global": move_global_trial,
                    "accepted": not rejected,
                    "rejection_reasons": rejection_reasons,
                    "severe_retries_used": int(severe_retries_used),
                })
                if not rejected:
                    accepted = True
                    break
                self.log.append(
                    "geometry true-response trial rejected: "
                    f"outer={outer}, retry={response_retry}, "
                    f"dC={100*signed_trial_change:.4f}%, "
                    f"move_global={move_global_trial:.5g}, "
                    f"reasons={rejection_reasons}"
                )
                if (
                    severe_worsening
                    and severe_retries_used > rollback_max_retries
                ):
                    rollback_failure_reason = "severe_retry_budget_exhausted"
                    break
                if not self._contract_outer_move_for_retry(
                    move_limit, rollback_minimum_move
                ):
                    rollback_failure_reason = "minimum_move_reached"
                    break
                response_retry += 1

            if not accepted:
                self.geometry_termination_reason = (
                    "objective_rollback_failed_"
                    f"{rollback_failure_reason}"
                )
                self.log.append(
                    "geometry SCA stopped: no acceptable true-response step "
                    f"after {len(response_trials)} trials at outer={outer}; "
                    f"reason={rollback_failure_reason}"
                )
                if iteration_history is not None:
                    iteration_history.append({
                        "outer": int(outer),
                        "objective": float(current.compliance),
                        "compliance": float(current.compliance),
                        "volume": float(volume_k),
                        "volume_ratio": float(volume_k/self.volume_bound),
                        "constraint_violation": float(initial_violation),
                        "feasible": bool(initial_feasible),
                        "accepted": False,
                        "termination_reason": (
                            "objective_rollback_failed_"
                            f"{rollback_failure_reason}"
                        ),
                        "rollback_failure_reason": rollback_failure_reason,
                        "response_trials": response_trials,
                        "response_retry_count": int(len(response_trials)-1),
                        "severe_retry_count": int(severe_retries_used),
                        "best_feasible_outer": best_outer,
                        "best_feasible_compliance": (
                            None if best_result is None
                            else float(best_result.compliance)
                        ),
                        "rib_names": [rib.name for rib in moved],
                        "thicknesses": [float(value) for value in t],
                        "coordinates": [
                            [float(value) for value in (*rib.p0, *rib.p1)]
                            for rib in moved
                        ],
                    })
                if best_x is None or best_result is None:
                    self.log.append(
                        "geometry SCA warning: no feasible true-FEA incumbent; "
                        "returning the current iterate"
                    )
                    return moved,t,current
                best_t,best_ribs=unpack(best_x)
                return best_ribs,best_t,best_result

            relative_change=abs(trial.compliance-current.compliance)/max(abs(current.compliance),1e-16)
            signed_relative_change=(
                (trial.compliance-current.compliance)
                / max(abs(current.compliance),1e-16)
            )
            design_change=maximum_normalized_design_change(
                x,candidate,global_lower,global_upper
            )
            thickness_design_change=maximum_normalized_design_change(
                x[:nt],candidate[:nt],global_lower[:nt],global_upper[:nt]
            )
            coordinate_design_change=(
                maximum_normalized_design_change(
                    x[nt:],candidate[nt:],global_lower[nt:],global_upper[nt:]
                ) if len(x) > nt else 0.0
            )
            maximum_absolute_thickness_change=float(
                np.max(np.abs(candidate_t-t))
            )
            old_coordinates=np.array([[*rib.p0,*rib.p1] for rib in moved]).ravel()
            new_coordinates=np.array([[*rib.p0,*rib.p1] for rib in candidate_ribs]).ravel()
            maximum_absolute_coordinate_change=float(np.max(np.abs(
                new_coordinates-old_coordinates
            )))
            violation=max(candidate_volume/self.volume_bound-1.0,0.0)
            feasible=candidate_volume<=self.volume_bound*(1+constraint_tolerance)
            predicted_change=approximation(candidate)
            actual_change=trial.compliance-current.compliance
            ratio=actual_change/predicted_change if abs(predicted_change)>1.0e-16 else np.nan
            self.log.append(
                f"geometry SCA outer completed: outer={outer}, "
                f"predicted dC={predicted_change:.7g}, true dC={actual_change:.7g}, "
                f"ratio={ratio:.5g}, dx={100*design_change:.4f}%, "
                f"move_global={move_limit.global_step:.5g}"
            )
            move_global_used=float(move_limit.global_step)
            x,current=candidate,trial
            initial_violation = violation
            if feasible and (
                best_result is None or current.compliance < best_result.compliance
            ):
                best_x = x.copy()
                best_result = current
                best_outer = outer
            move_lower,move_upper=move_limit.update(
                x,current.compliance,violation,constraint_tolerance
            )
            step_converged=sca_step_converged(
                feasible,
                relative_change,
                design_change,
                objective_tolerance,
                design_tolerance,
                design_guard_tolerance,
            )
            consecutive_converged=(
                consecutive_converged+1 if step_converged else 0
            )
            if iteration_history is not None:
                iteration_history.append({
                    "outer": int(outer),
                    "objective": float(trial.compliance),
                    "compliance": float(trial.compliance),
                    "volume": float(candidate_volume),
                    "volume_ratio": float(candidate_volume/self.volume_bound),
                    "constraint_violation": float(violation),
                    "feasible": bool(feasible),
                    "accepted": True,
                    "response_trials": response_trials,
                    "response_retry_count": int(len(response_trials)-1),
                    "best_feasible_outer": best_outer,
                    "best_feasible_compliance": (
                        None if best_result is None
                        else float(best_result.compliance)
                    ),
                    "is_best_feasible": bool(best_outer == outer),
                    "objective_relative_change": float(relative_change),
                    "objective_relative_change_signed": float(signed_relative_change),
                    "design_change": float(design_change),
                    "thickness_design_change": float(thickness_design_change),
                    "coordinate_design_change": float(coordinate_design_change),
                    "maximum_absolute_thickness_change": maximum_absolute_thickness_change,
                    "maximum_absolute_coordinate_change": maximum_absolute_coordinate_change,
                    "predicted_objective_change": float(predicted_change),
                    "true_objective_change": float(actual_change),
                    "approximation_ratio": float(ratio),
                    "inner_iterations": int(inner),
                    "frozen_geometry_count": int(len(frozen_geometry_indices)),
                    "frozen_geometry_indices": [
                        int(index) for index in sorted(frozen_geometry_indices)
                    ],
                    "frozen_geometry_names": [
                        ribs[index].name for index in sorted(frozen_geometry_indices)
                    ],
                    "frozen_geometry_reasons": {
                        ribs[index].name: sorted(frozen_geometry_reasons[index])
                        for index in sorted(frozen_geometry_reasons)
                    },
                    "move_global_used": move_global_used,
                    "move_global_next": float(move_limit.global_step),
                    "step_converged": bool(step_converged),
                    "consecutive_converged": int(consecutive_converged),
                    "rib_names": [rib.name for rib in candidate_ribs],
                    "thicknesses": [float(value) for value in candidate_t],
                    "coordinates": [
                        [float(value) for value in (*rib.p0, *rib.p1)]
                        for rib in candidate_ribs
                    ],
                })
            if consecutive_converged>=consecutive_required:
                self.geometry_termination_reason = "converged"
                if iteration_history is not None:
                    iteration_history[-1]["termination_reason"] = "converged"
                self.log.append(
                    f"geometry SCA converged: outer={outer}, "
                    f"dC={100*relative_change:.4f}%, dx={100*design_change:.4f}%"
                )
                if best_x is None or best_result is None:
                    self.log.append(
                        "geometry SCA warning: no feasible true-FEA incumbent; "
                        "returning the current iterate"
                    )
                    return candidate_ribs,candidate_t,current
                best_t,best_ribs=unpack(best_x)
                if best_outer is not None and best_outer != outer:
                    self.log.append(
                        "geometry SCA returned best feasible true-FEA incumbent: "
                        f"outer={best_outer}, C={best_result.compliance:.7g}"
                    )
                return best_ribs,best_t,best_result
        self.log.append(f"geometry SCA warning: outer iteration limit {max_iterations} reached")
        self.geometry_termination_reason = "iteration_limit"
        if best_x is None or best_result is None:
            final_t,final_ribs=unpack(x)
            self.log.append(
                "geometry SCA warning: no feasible true-FEA incumbent; "
                "returning the last iterate"
            )
            return final_ribs,final_t,current
        best_t,best_ribs=unpack(best_x)
        if best_outer is not None and best_outer != max_iterations:
            self.log.append(
                "geometry SCA returned best feasible true-FEA incumbent: "
                f"outer={best_outer}, C={best_result.compliance:.7g}"
            )
        return best_ribs,best_t,best_result

    def _solve_rationalization_eq18(
        self,
        ribs: list[Rib],
        thicknesses: np.ndarray,
        current_result: AnalysisResult,
        cref: float,
        tref: float,
        coordinate_bounds: np.ndarray | None = None,
    ) -> tuple[list[Rib], np.ndarray, AnalysisResult, np.ndarray]:
        """Solve the smooth Eq. (18) problem for the current active rib set."""
        settings = self.cfg["algorithm"]
        self.rationalization_termination_reason = None
        (
            rollback_threshold,
            rollback_max_retries,
            rollback_minimum_move,
        ) = self._outer_objective_rollback_settings()
        n = len(ribs)
        variable_map = build_mirror_variable_map(
            ribs, self.mirror_axes, self.symmetry_width, self.symmetry_height
        )
        nt = variable_map.thickness_count
        coordinates = np.array([[*r.p0, *r.p1] for r in ribs], float)
        projected = np.r_[
            variable_map.reduce_thicknesses(thicknesses),
            variable_map.reduce_coordinates(coordinates.ravel()),
        ]
        if coordinate_bounds is None:
            coordinate_bounds = np.array([
                [
                    [0.0,self.model.width],
                    [0.0,self.model.height],
                    [0.0,self.model.width],
                    [0.0,self.model.height],
                ] for q in coordinates
            ], float)
        fd_step = float(settings["geometry_fd_fraction"]) * float(self.cfg["initial_rib_cell_size"])
        min_length = 0.25 * float(self.cfg["initial_rib_cell_size"])
        objective_tolerance = float(settings["sca_objective_tolerance"])
        design_tolerance = float(settings.get("sca_design_tolerance", 0.001))
        design_guard_tolerance = float(
            settings.get("sca_design_guard_tolerance", 0.010)
        )
        constraint_tolerance = float(settings["sca_constraint_tolerance"])
        consecutive_required = int(
            settings.get("sca_consecutive_convergence_steps", 2)
        )
        compliance_tolerance = float(
            settings.get("rationalization_compliance_tolerance", 0.001)
        )
        if compliance_tolerance != 0.001:
            raise ValueError("rationalization_compliance_tolerance must be 0.001")
        compliance_limit = cref*(1.0+compliance_tolerance)
        rationalization_move_step = self._positive_move_limit_setting(
            "rationalization_move_limit_initial", 0.50
        )
        proximal = float(settings["rationalization_sca_proximal"])
        dual_tolerance = float(
            settings.get("rationalization_dual_tolerance", 1.0e-9)
        )
        if dual_tolerance <= 0.0:
            raise ValueError("rationalization_dual_tolerance must be positive")
        max_outer = int(settings["rationalization_max_iterations"])
        minimum_outer = int(settings.get("rationalization_min_iterations", 5))
        if minimum_outer < 1:
            raise ValueError("rationalization_min_iterations must be positive")
        full_coordinate_scale = np.tile([
            self.model.width, self.model.height,
            self.model.width, self.model.height,
        ], n)
        scale_p = variable_map.reduce_coordinate_quadratic_scale(
            full_coordinate_scale
        )
        reduced_coordinate_bounds = variable_map.reduce_coordinate_bounds(
            coordinate_bounds
        )
        global_lower = np.r_[
            np.full(nt, self.t_lower), reduced_coordinate_bounds[:, 0]
        ]
        global_upper = np.r_[
            np.full(nt, self.t_upper), reduced_coordinate_bounds[:, 1]
        ]

        def unpack(x: np.ndarray) -> tuple[np.ndarray, list[Rib]]:
            t = variable_map.expand_thicknesses(x[:nt])
            p = variable_map.expand_coordinates(x[nt:]).reshape(n, 4)
            moved = [Rib(tuple(q[:2]), tuple(q[2:]), r.height, r.name, r.segments) for q, r in zip(p, ribs)]
            return t, moved

        def volume_data(x: np.ndarray) -> tuple[float, np.ndarray]:
            t, moved = unpack(x)
            thickness_jac = np.zeros(n)
            coordinate_jac = np.zeros(4*n)
            volume = 0.0
            for i, (te, rib) in enumerate(zip(t, moved)):
                p0, p1 = np.asarray(rib.p0), np.asarray(rib.p1)
                delta = p1 - p0
                length = max(np.linalg.norm(delta), 1.0e-12)
                volume += length * rib.height * te
                thickness_jac[i] = length * rib.height
                coordinate_gradient = rib.height * te * delta / length
                coordinate_jac[4*i:4*i+2] = -coordinate_gradient
                coordinate_jac[4*i+2:4*i+4] = coordinate_gradient
            return float(volume), np.r_[
                variable_map.reduce_thickness_gradient(thickness_jac),
                variable_map.reduce_coordinate_gradient(coordinate_jac),
            ]

        def projected_count(x: np.ndarray, beta: float, threshold: float) -> float:
            return smooth_member_count(
                variable_map.expand_thicknesses(x[:nt]), threshold, beta
            )

        def projected_count_gradient(
            x: np.ndarray,
            beta: float,
            threshold: float,
        ) -> np.ndarray:
            gradient = np.zeros_like(x)
            full_gradient = smooth_member_count_gradient(
                variable_map.expand_thicknesses(x[:nt]), threshold, beta
            )
            gradient[:nt] = variable_map.reduce_thickness_gradient(
                full_gradient
            )
            return gradient

        beta_max = float(settings["rationalization_beta"])
        beta_initial = float(settings.get("rationalization_beta_initial", 1.0))
        beta_increment = float(settings.get("rationalization_beta_increment", 1.0))
        if beta_initial <= 0.0 or beta_increment <= 0.0:
            raise ValueError("rationalization beta initial value and increment must be positive")
        if beta_initial > beta_max:
            raise ValueError("rationalization beta initial value exceeds its maximum")
        # The reference thickness is fixed for the complete Eq. (18) solve.
        # The former 50*tref -> tref homotopy created nineteen independent SCA
        # stages and dominated the rationalization FEA count.
        for projection_threshold in [tref]:
            outer_fea = 0
            coordinate_step_scale = variable_map.reduce_coordinate_scale(
                self._coordinate_move_step_scale(n)
            )
            move_limit = self._new_move_limit(
                global_lower, global_upper,
                np.r_[np.full(nt, np.nan), coordinate_step_scale],
                initial_global_step=rationalization_move_step,
            )
            initial_t, initial_ribs = unpack(projected)
            projected_coordinates = np.array([
                [*rib.p0, *rib.p1] for rib in initial_ribs
            ])
            if (
                not np.allclose(initial_t, thicknesses, rtol=0.0, atol=1.0e-12)
                or not np.allclose(projected_coordinates, coordinates, rtol=0.0, atol=1.0e-10)
            ):
                current_result = self.analyze(initial_ribs, initial_t)
            initial_violation = max(
                current_result.compliance/cref-1.0,
                self.volume(initial_ribs, initial_t)/self.volume_bound-1.0,
                0.0,
            )
            move_lower, move_upper = move_limit.update(
                projected,
                projected_count(projected, beta_initial, projection_threshold),
                initial_violation,
                constraint_tolerance,
            )
            consecutive_converged = 0
            converged = False
            rollback_failed = False
            for outer in range(1, max_outer + 1):
                beta = min(
                    beta_initial+(outer-1)*beta_increment,
                    beta_max,
                )
                current_x = projected.copy()
                current_t, current_ribs = unpack(current_x)
                current_compliance_value = float(current_result.compliance)

                # The current design has already been analyzed (either before
                # entering rationalization or as the previous trial).  Use
                # that FEA to obtain sensitivities and construct convex local
                # models. No finite-element call occurs inside the dual solve.
                gt_full = self.model.compliance_gradient(current_ribs, current_t, current_result)
                gp_full = self.model.geometry_gradient(current_ribs, current_t, current_result, fd_step).ravel()
                reciprocal_coeff = variable_map.reduce_thickness_gradient(
                    np.maximum(-gt_full, 0.0)*current_t**2
                )
                gp = variable_map.reduce_coordinate_gradient(gp_full)
                count_k = projected_count(current_x, beta, projection_threshold)
                count_gradient = projected_count_gradient(
                    current_x, beta, projection_threshold
                )
                if outer == 1 and float(np.max(np.abs(count_gradient[:nt]))) <= 1.0e-14:
                    self.log.append(
                        f"rationalization projection saturated at fixed "
                        f"tref={projection_threshold:.7g}: member-count "
                        f"gradient is numerically zero"
                    )
                volume_k, volume_gradient = volume_data(current_x)
                compliance_scale = max(abs(current_result.compliance), 1.0e-16)
                group_sizes = np.asarray([
                    len(group) for group in variable_map.rib_groups
                ], float)
                thickness_scale = (
                    np.maximum(current_x[:nt], tref)/np.sqrt(group_sizes)
                )

                def approximate_count(y: np.ndarray) -> float:
                    dy = y - current_x
                    dt = dy[:nt]
                    dp = dy[nt:]
                    return float(
                        count_k + count_gradient @ dy
                        + 0.5 * proximal * np.sum((dt / thickness_scale)**2)
                        + 0.5 * proximal * np.sum((dp / scale_p)**2)
                    )

                def approximate_compliance(y: np.ndarray) -> float:
                    safe = np.maximum(y[:nt], self.t_lower)
                    dp = y[nt:] - current_x[nt:]
                    return float(
                        current_result.compliance
                        + np.sum(reciprocal_coeff * (1.0 / safe - 1.0 / current_x[:nt]))
                        + gp @ dp
                        + 0.5 * proximal * compliance_scale * np.sum((dp / scale_p)**2)
                    )

                # Resolve the same beta/current-state approximation after a
                # severe true member-count increase.  Rejected trials never
                # replace ``projected`` or ``current_result``.
                response_trials: list[dict] = []
                accepted = False
                rollback_failure_reason = "unknown"
                for response_retry in range(rollback_max_retries+1):
                    if response_retry:
                        move_lower, move_upper = move_limit.current_bounds(current_x)
                    invalid = False
                    inner_lower=move_lower.copy()
                    inner_upper=move_upper.copy()
                    frozen_geometry_indices: set[int] = set()
                    frozen_geometry_reasons: dict[int, set[str]] = {}
                    for inner in range(1, n+2):
                        dual_result = solve_rationalization_convex_subproblem(
                            current=current_x,
                            count_gradient=count_gradient,
                            reciprocal_coefficients=reciprocal_coeff,
                            geometry_gradient=gp,
                            volume_gradient=volume_gradient,
                            compliance_at_current=current_result.compliance,
                            compliance_bound=cref,
                            volume_at_current=volume_k,
                            volume_bound=self.volume_bound,
                            proximal=proximal,
                            compliance_scale=compliance_scale,
                            thickness_scale=thickness_scale,
                            coordinate_scale=scale_p,
                            lower=inner_lower,
                            upper=inner_upper,
                            constraint_tolerance=dual_tolerance,
                        )
                        candidate = np.asarray(dual_result.x, float)
                        candidate_t, candidate_ribs = unpack(candidate)
                        candidate_t = self._feasible_start(candidate_ribs, candidate_t)
                        candidate[:nt] = variable_map.reduce_thicknesses(candidate_t)
                        candidate_t, candidate_ribs = unpack(candidate)
                        freeze_reasons = geometry_move_freeze_reasons(
                            current_ribs,candidate_ribs,min_length
                        )
                        if freeze_reasons:
                            invalid = True
                            new_indices=set(freeze_reasons)-frozen_geometry_indices
                            if not new_indices:
                                candidate[nt:]=current_x[nt:]
                                candidate_t,candidate_ribs=unpack(candidate)
                                candidate_t=self._feasible_start(candidate_ribs,candidate_t)
                                candidate[:nt]=variable_map.reduce_thicknesses(candidate_t)
                                candidate_t,candidate_ribs=unpack(candidate)
                                invalid=False
                                self.log.append(
                                    "rationalization convex inner local freeze fallback: "
                                    "all rib positions retained at current outer design"
                                )
                                break
                            for index in sorted(new_indices):
                                variables = np.unique(
                                    variable_map.coordinate_variable[4*index:4*index+4]
                                )
                                variables = variables[variables >= 0]
                                coordinate_indices = nt+variables
                                inner_lower[coordinate_indices]=current_x[coordinate_indices]
                                inner_upper[coordinate_indices]=current_x[coordinate_indices]
                                frozen_geometry_reasons.setdefault(index,set()).update(
                                    freeze_reasons[index]
                                )
                            frozen_geometry_indices.update(new_indices)
                            frozen_names=[
                                current_ribs[index].name for index in sorted(new_indices)
                            ]
                            self.log.append(
                                "rationalization convex inner local freeze: "
                                f"inner={inner}, ribs={frozen_names}, "
                                f"move_global unchanged={move_limit.global_step:.5g}"
                            )
                            continue
                        invalid=False
                        break
                    if invalid:
                        candidate[nt:]=current_x[nt:]
                        candidate_t, candidate_ribs = unpack(candidate)
                        candidate_t=self._feasible_start(candidate_ribs,candidate_t)
                        candidate[:nt]=variable_map.reduce_thicknesses(candidate_t)
                        candidate_t,candidate_ribs=unpack(candidate)

                    true_count = projected_count(
                        candidate, beta, projection_threshold
                    )
                    candidate_volume = self.volume(candidate_ribs, candidate_t)
                    candidate_variables_finite = self._candidate_design_is_finite(
                        candidate, candidate_ribs, candidate_t
                    )
                    candidate_count_finite = bool(np.isfinite(true_count))
                    candidate_volume_finite = bool(np.isfinite(candidate_volume))
                    candidate_design_finite = bool(
                        candidate_variables_finite
                        and candidate_count_finite
                        and candidate_volume_finite
                    )
                    move_global_trial = float(move_limit.global_step)
                    if not candidate_design_finite:
                        rejection_reasons = []
                        if not candidate_variables_finite:
                            rejection_reasons.append("nonfinite_candidate_design")
                        if not candidate_count_finite:
                            rejection_reasons.append("nonfinite_projected_count")
                        if not candidate_volume_finite:
                            rejection_reasons.append("nonfinite_volume_response")
                        response_trials.append({
                            "retry": int(response_retry),
                            "objective": (
                                float(true_count) if np.isfinite(true_count) else None
                            ),
                            "objective_relative_change_signed": None,
                            "objective_finite": bool(np.isfinite(true_count)),
                            "design_finite": False,
                            "constraint_response_finite": False,
                            "compliance": None,
                            "fea_performed": False,
                            "move_global": move_global_trial,
                            "accepted": False,
                            "rejection_reasons": rejection_reasons,
                        })
                        self.log.append(
                            "rationalization invalid candidate rejected before FEA: "
                            f"beta={beta:g}, outer={outer}, retry={response_retry}, "
                            f"move_global={move_global_trial:.5g}"
                        )
                        if response_retry >= rollback_max_retries:
                            rollback_failure_reason = "retry_budget_exhausted"
                            break
                        if not self._contract_outer_move_for_retry(
                            move_limit, rollback_minimum_move
                        ):
                            rollback_failure_reason = "minimum_move_reached"
                            break
                        continue

                    inner_predicted_objective = float(approximate_count(candidate))
                    inner_predicted_compliance = float(approximate_compliance(candidate))
                    inner_linearized_volume = float(
                        volume_k + volume_gradient @ (candidate-current_x)
                    )
                    predicted_compliance_residual = inner_predicted_compliance-cref
                    predicted_volume_residual = (
                        inner_linearized_volume-self.volume_bound
                    )
                    inner_success = bool(
                        dual_result.success
                        and predicted_compliance_residual
                        <= dual_tolerance*max(abs(cref), 1.0)
                        and predicted_volume_residual
                        <= dual_tolerance*max(abs(self.volume_bound), 1.0)
                    )
                    inner_status = int(dual_result.status)
                    inner_iterations = int(dual_result.iterations)
                    inner_message = str(dual_result.message)

                    outer_fea += 1
                    trial = self.analyze(candidate_ribs, candidate_t)
                    (
                        signed_count_change,
                        trial_objective_finite,
                        severe_worsening,
                    ) = self._severe_objective_response(
                        count_k, true_count, rollback_threshold
                    )
                    constraint_response_finite = bool(
                        np.isfinite(float(trial.compliance))
                        and np.isfinite(candidate_volume)
                    )
                    rejected = bool(
                        severe_worsening or not constraint_response_finite
                    )
                    rejection_reasons = []
                    if severe_worsening:
                        rejection_reasons.append("severe_objective")
                    if not constraint_response_finite:
                        rejection_reasons.append("nonfinite_constraint_response")
                    response_trials.append({
                        "retry": int(response_retry),
                        "objective": float(true_count),
                        "objective_relative_change_signed": float(
                            signed_count_change
                        ),
                        "objective_finite": trial_objective_finite,
                        "design_finite": True,
                        "constraint_response_finite": constraint_response_finite,
                        "compliance": (
                            float(trial.compliance)
                            if np.isfinite(float(trial.compliance)) else None
                        ),
                        "fea_performed": True,
                        "move_global": move_global_trial,
                        "accepted": not rejected,
                        "rejection_reasons": rejection_reasons,
                    })
                    if not rejected:
                        accepted = True
                        break
                    self.log.append(
                        "rationalization true-response trial rejected: "
                        f"beta={beta:g}, outer={outer}, retry={response_retry}, "
                        f"dN={100*signed_count_change:.4f}%, "
                        f"move_global={move_global_trial:.5g}, "
                        f"reasons={rejection_reasons}"
                    )
                    if response_retry >= rollback_max_retries:
                        rollback_failure_reason = "retry_budget_exhausted"
                        break
                    if not self._contract_outer_move_for_retry(
                        move_limit, rollback_minimum_move
                    ):
                        rollback_failure_reason = "minimum_move_reached"
                        break

                if not accepted:
                    rollback_failed = True
                    self.rationalization_termination_reason = (
                        "objective_rollback_failed"
                    )
                    self.rationalization_history.append({
                        "event": "eq18_outer_iteration",
                        "outer": int(outer),
                        "beta": float(beta),
                        "current_objective": float(count_k),
                        "objective": float(count_k),
                        "current_compliance": current_compliance_value,
                        "compliance": current_compliance_value,
                        "volume": float(volume_k),
                        "accepted": False,
                        "termination_reason": "objective_rollback_failed",
                        "rollback_failure_reason": rollback_failure_reason,
                        "response_trials": response_trials,
                        "response_retry_count": int(len(response_trials)-1),
                        "rib_names": [rib.name for rib in current_ribs],
                        "thicknesses": [float(value) for value in current_t],
                    })
                    self.log.append(
                        "rationalization SCA stopped: no acceptable severe-"
                        f"objective step after {len(response_trials)} trials "
                        f"at beta={beta:g}, outer={outer}; the accepted outer "
                        "start was restored"
                    )
                    break

                count_change = abs(true_count-count_k)/max(abs(count_k),1.0e-16)
                design_change = maximum_normalized_design_change(
                    current_x,candidate,global_lower,global_upper
                )
                compliance_feasible = trial.compliance <= compliance_limit
                volume_feasible = (
                    self.volume(candidate_ribs,candidate_t)
                    <= self.volume_bound*(1.0+constraint_tolerance)
                )
                projected = candidate
                current_result = trial
                violation = max(
                    trial.compliance/cref-1.0,
                    self.volume(candidate_ribs,candidate_t)/self.volume_bound-1.0,
                    0.0,
                )
                # With beta continuation, compare the candidate and current
                # design using the same beta when updating the move limit.
                # Otherwise a change of projection sharpness could be mistaken
                # for an unsuccessful design step.
                move_limit.previous_objective = float(count_k)
                move_lower,move_upper=move_limit.update(
                    projected,true_count,violation,constraint_tolerance
                )
                self.log.append(
                    f"rationalization SCA outer completed: beta={beta:g}, "
                    f"threshold={projection_threshold:.7g}, outer={outer}, "
                    f"dN={100*count_change:.4f}%, dx={100*design_change:.4f}%, "
                    f"C={trial.compliance:.7g}, V={self.volume(candidate_ribs,candidate_t):.7g}"
                )
                self.rationalization_history.append({
                    "event": "eq18_outer_iteration",
                    "outer": int(outer),
                    "beta": float(beta),
                    "current_objective": float(count_k),
                    "objective": float(true_count),
                    "inner_predicted_objective": inner_predicted_objective,
                    "current_compliance": current_compliance_value,
                    "compliance": float(trial.compliance),
                    "inner_predicted_compliance": inner_predicted_compliance,
                    "inner_predicted_compliance_margin": float(
                        cref-inner_predicted_compliance
                    ),
                    "true_compliance_change": float(
                        trial.compliance-current_compliance_value
                    ),
                    "inner_linearized_volume": inner_linearized_volume,
                    "inner_success": inner_success,
                    "inner_status": inner_status,
                    "inner_iterations": inner_iterations,
                    "inner_message": inner_message,
                    "geometry_freeze_inner_solves": int(inner),
                    "frozen_geometry_indices": [
                        int(index) for index in sorted(frozen_geometry_indices)
                    ],
                    "frozen_geometry_names": [
                        current_ribs[index].name
                        for index in sorted(frozen_geometry_indices)
                    ],
                    "frozen_geometry_reasons": {
                        current_ribs[index].name: sorted(
                            frozen_geometry_reasons[index]
                        )
                        for index in sorted(frozen_geometry_reasons)
                    },
                    "volume": float(self.volume(candidate_ribs, candidate_t)),
                    "objective_relative_change": float(count_change),
                    "design_change": float(design_change),
                    "accepted": True,
                    "response_trials": response_trials,
                    "response_retry_count": int(len(response_trials)-1),
                    "compliance_feasible": bool(compliance_feasible),
                    "volume_feasible": bool(volume_feasible),
                    "rib_names": [rib.name for rib in candidate_ribs],
                    "thicknesses": [float(value) for value in candidate_t],
                })
                beta_at_maximum = bool(
                    beta >= beta_max-1.0e-12*max(abs(beta_max), 1.0)
                )
                step_converged = sca_step_converged(
                    beta_at_maximum and compliance_feasible and volume_feasible,
                    count_change,
                    design_change,
                    objective_tolerance,
                    design_tolerance,
                    design_guard_tolerance,
                )
                consecutive_converged = (
                    consecutive_converged+1 if step_converged else 0
                )
                if (
                    outer >= minimum_outer
                    and consecutive_converged >= consecutive_required
                ):
                    converged = True
                if converged:
                    self.rationalization_termination_reason = "converged"
                    self.log.append(
                        f"rationalization SCA converged: beta={beta:g}, "
                        f"threshold={projection_threshold:.7g}, outer={outer}, "
                        f"dN={100*count_change:.4f}%, dx={100*design_change:.4f}%, "
                        f"projected count={true_count:.3f}, FEA={outer_fea}"
                    )
                    break
            if rollback_failed:
                self.log.append(
                    f"rationalization SCA beta={beta:g}/{beta_max:g}, "
                    f"threshold={projection_threshold:.7g} stopped after "
                    f"objective rollback failure; FEA={outer_fea}"
                )
            elif not converged:
                self.rationalization_termination_reason = "iteration_limit"
                self.log.append(
                    f"rationalization SCA beta={beta:g}/{beta_max:g}, "
                    f"threshold={projection_threshold:.7g} reached outer limit {max_outer}; "
                    f"projected count="
                    f"{projected_count(projected,beta,projection_threshold):.3f}, "
                    f"FEA={outer_fea}"
                )

        projected_t, moved = unpack(projected)
        return moved, projected_t, current_result, coordinate_bounds

    def rationalize(
        self,
        ribs: list[Rib],
        thicknesses: np.ndarray,
        geometry_result: AnalysisResult,
        relaxation: float,
    ) -> tuple[list[Rib], np.ndarray, AnalysisResult]:
        final = self._rationalize(
            ribs, thicknesses, geometry_result, relaxation
        )
        self._report_progress("rationalization", final[0], final[2])
        return final

    def _rationalize(self, ribs: list[Rib], thicknesses: np.ndarray, geometry_result: AnalysisResult, relaxation: float) -> tuple[list[Rib], np.ndarray, AnalysisResult]:
        self.rationalization_history.clear()
        self.rationalization_termination_reason = None
        if relaxation <= 0 or len(ribs) <= 3:
            return ribs, thicknesses, geometry_result
        settings = self.cfg["algorithm"]
        cref = (1.0 + float(relaxation)) * geometry_result.compliance
        acceptance_limit = cref
        base_ribs = list(ribs)
        base_t = np.asarray(thicknesses, float).copy()
        n_rib_before_rationalization = len(base_ribs)
        reference_quantile = self.rationalization_reference_quantile(
            relaxation, n_rib_before_rationalization
        )
        recovery_move_step = self._positive_move_limit_setting(
            "rationalization_move_limit_initial", 0.50
        )
        recovery_iterations = int(settings["rationalization_geometry_iterations"])
        tref = float(np.quantile(base_t, reference_quantile))
        phase_start = self.analysis_count
        reference_record = {
            "event": "reference",
            "tref": tref,
            "reference_quantile": reference_quantile,
            "reference_quantile_formula": "1/n_rib + rho",
            "n_rib_before_rationalization": n_rib_before_rationalization,
            "relaxation": float(relaxation),
            "recovery_strategy": "fixed_initial_deleted_seed_count_thickest_first",
            "recovery_seed_target_formula": "max(1, ceil(n_rrib/3))",
            "recovery_seed_target_basis": "initial_deleted_rib_count",
            "recovery_seed_ranking": (
                "saved_eq18_thickness_descending_then_original_index"
            ),
            "recovery_mirror_completion": bool(self.mirror_axes),
            "recovery_mirror_axes": list(self.mirror_axes),
            "cref": cref,
            "final_acceptance_limit": acceptance_limit,
            "eq18_compliance_limit": 1.001*cref,
            "geometry_compliance": float(geometry_result.compliance),
        }
        self.rationalization_history.append(reference_record)
        self.log.append(
            f"rationalization quantile-threshold solve: tref={tref:.7g}, "
            f"{100*reference_quantile:.7g}th percentile of "
            f"{len(base_t)} geometry-stage thicknesses "
            f"(q=1/{n_rib_before_rationalization} + "
            f"{float(relaxation):.7g}), "
            f"Cref={cref:.7g}, Eq.18 compliance limit={1.001*cref:.7g}"
        )
        continuous_ribs, continuous_t, continuous_result, bounds = (
            self._solve_rationalization_eq18(
                list(base_ribs), base_t.copy(), geometry_result, cref, tref
            )
        )
        eq18_fea = self.analysis_count-phase_start
        # Preserve the one Eq. (18) solution. Any discrete deletion attempt
        # below starts independently from this saved continuous design.
        saved_ribs = list(continuous_ribs)
        saved_t = np.asarray(continuous_t, float).copy()
        saved_result = continuous_result
        saved_bounds = np.asarray(bounds, float).copy()
        short_length_threshold = self.short_rib_length_threshold()
        short_thickness_factor = self.short_rib_thickness_factor()

        eq18_rib_data = []
        self.log.append(
            f"rationalization Eq.18 thicknesses: tref={tref:.9g}, "
            f"ribs={len(saved_ribs)}, C={saved_result.compliance:.9g}"
        )
        for index, (rib, thickness) in enumerate(zip(saved_ribs, saved_t), start=1):
            below = bool(thickness < tref)
            short = bool(rib.length < short_length_threshold)
            below_short_cap = bool(thickness < short_thickness_factor*tref)
            record = {
                "index": index,
                "name": rib.name,
                "p0": [float(value) for value in rib.p0],
                "p1": [float(value) for value in rib.p1],
                "length": float(rib.length),
                "thickness": float(thickness),
                "thickness_over_tref": float(thickness/tref),
                "below_tref": below,
                "short": short,
                "below_short_thickness_cap": below_short_cap,
                "filter_candidate_at_tref": bool(
                    below or (short and below_short_cap)
                ),
            }
            eq18_rib_data.append(record)
            self.log.append(
                f"rationalization Eq.18 rib {index:02d}: name={rib.name}, "
                f"p0=({rib.p0[0]:.7g},{rib.p0[1]:.7g}), "
                f"p1=({rib.p1[0]:.7g},{rib.p1[1]:.7g}), "
                f"L={rib.length:.9g}, t={thickness:.9g}, "
                f"t/tref={thickness/tref:.6g}, "
                f"short={short}, t<{short_thickness_factor:g}*tref={below_short_cap}, "
                f"filter={'remove' if below or (short and below_short_cap) else 'keep'}"
            )
        self.rationalization_history.append({
            "event": "eq18_solution",
            "tref": tref,
            "compliance": float(saved_result.compliance),
            "volume": self.volume(saved_ribs, saved_t),
            "fea": eq18_fea,
            "ribs": eq18_rib_data,
        })

        saved_lengths = np.asarray([rib.length for rib in saved_ribs], float)
        initial_thin = saved_t < tref
        initial_short = saved_lengths < short_length_threshold
        initial_short_and_light = initial_short & (
            saved_t < short_thickness_factor*tref
        )
        initial_remove = self._mirror_group_removal_mask(
            saved_ribs, initial_thin | initial_short_and_light
        )
        initial_removed_indices = list(map(int, np.flatnonzero(initial_remove)))
        if not initial_removed_indices:
            self.log.append(
                "rationalization stopped: Eq. (18) produced no thin or short/light "
                f"rib at tref={tref:.7g}; restored pre-rationalization geometry "
                f"design (Eq.18 FEA={eq18_fea})"
            )
            return base_ribs, base_t, geometry_result

        n_rrib = len(initial_removed_indices)
        recovery_seed_target = max(1, (n_rrib+2)//3)
        reference_record.update({
            "n_rrib": n_rrib,
            "recovery_seed_target": recovery_seed_target,
        })
        self.log.append(
            "rationalization recovery configuration: "
            "strategy=fixed_initial_deleted_seed_count_thickest_first, "
            f"n_rrib={n_rrib}, seed_target={recovery_seed_target}, "
            "seed_target_formula=max(1,ceil(n_rrib/3)), "
            "ranking=saved_eq18_thickness_descending_then_original_index, "
            f"mirror_completion={bool(self.mirror_axes)}, "
            f"mirror_axes={list(self.mirror_axes)}"
        )

        attempt_costs = []
        # Keep tref, Cref, n_rrib, and the seed target fixed. After every failed
        # reduced-topology solve, rank the ribs that remain deleted by saved
        # Eq. (18) thickness (original index breaks ties), restore up to the
        # fixed number of seeds, and complete their mirror groups only when
        # symmetry was explicitly configured. Re-solve while at least one rib
        # remains deleted.
        removal_indices = set(initial_removed_indices)
        maximum_attempts = len(initial_removed_indices)
        recovery_round = 0
        for attempt in range(1, maximum_attempts + 1):
            remove_mask = np.array(
                [index in removal_indices for index in range(len(saved_ribs))],
                dtype=bool,
            )
            keep = ~remove_mask
            removed = int(np.count_nonzero(~keep))
            removed_indices = [int(index+1) for index in np.flatnonzero(~keep)]
            kept_indices = [int(index+1) for index in np.flatnonzero(keep)]
            removed_names = [saved_ribs[index-1].name for index in removed_indices]
            kept_names = [saved_ribs[index-1].name for index in kept_indices]
            restored_zero_indices = sorted(
                set(initial_removed_indices)-removal_indices
            )
            attempt_record = {
                "event": "filtering_attempt",
                "attempt": attempt,
                "attempts": maximum_attempts,
                "threshold": float(tref),
                "removed_indices": removed_indices,
                "removed_names": removed_names,
                "kept_indices": kept_indices,
                "kept_names": kept_names,
                "restored_indices": [index+1 for index in restored_zero_indices],
                "restored_names": [
                    saved_ribs[index].name for index in restored_zero_indices
                ],
                "thin_removed_indices": [
                    int(index+1)
                    for index in np.flatnonzero(initial_thin & remove_mask)
                ],
                "short_light_removed_indices": [
                    int(index+1)
                    for index in np.flatnonzero(
                        initial_short_and_light & ~initial_thin & remove_mask
                    )
                ],
                "short_length_threshold": float(short_length_threshold),
                "short_thickness_factor": float(short_thickness_factor),
            }
            self.rationalization_history.append(attempt_record)
            if not keep.any():
                self.log.append(
                    f"rationalization deletion attempt {attempt}/{maximum_attempts} rejected: "
                    f"tref={tref:.7g} would delete every rib"
                )
                attempt_costs.append((attempt, 0, 0))
                reduced_result = None
            else:
                reduced_ribs = [
                    rib for rib, flag in zip(saved_ribs, keep) if flag
                ]
                reduced_t = saved_t[keep].copy()
                reduced_bounds = saved_bounds[keep].copy()
                self.log.append(
                    f"rationalization deletion attempt {attempt}/{maximum_attempts}: "
                    f"{len(saved_ribs)} -> {len(reduced_ribs)}, removed={removed}, "
                    f"tref={tref:.7g}, removed_indices={removed_indices}, "
                    f"removed_names={removed_names}, kept_indices={kept_indices}, "
                    f"restored_names={attempt_record['restored_names']}"
                )

                try:
                    reduced_t = self._feasible_start(reduced_ribs, reduced_t)
                    geometry_start = self.analysis_count
                    geometry_iteration_history: list[dict] = []
                    # Enter Eq. (7) directly. Its initialization FEA supplies the
                    # first valid reduced-topology displacement/sensitivity field.
                    reduced_ribs, reduced_t, reduced_result = self.optimize_geometry(
                        reduced_ribs,
                        reduced_t,
                        None,
                        reduced_bounds,
                        recovery_iterations,
                        initial_move_step=recovery_move_step,
                        iteration_history=geometry_iteration_history,
                    )
                    geometry_fea = self.analysis_count-geometry_start
                    deletion_fea = 0
                except np.linalg.LinAlgError:
                    deletion_fea = 0
                    geometry_fea = self.analysis_count-geometry_start
                    reduced_result = None
                    self.log.append(
                        f"rationalization deletion attempt {attempt}/{maximum_attempts} "
                        "rejected: singular batch-reduced model"
                    )
                for record in geometry_iteration_history:
                    self.rationalization_history.append({
                        "event": "post_filter_geometry_iteration",
                        "attempt": int(attempt),
                        **record,
                    })
                attempt_costs.append((attempt, deletion_fea, geometry_fea))

            accepted = bool(
                reduced_result is not None
                and reduced_result.compliance <= acceptance_limit
            )
            final_rib_data = []
            for index, (rib, thickness) in enumerate(
                zip(reduced_ribs, reduced_t), start=1
            ) if reduced_result is not None else []:
                final_rib_data.append({
                    "index": index,
                    "name": rib.name,
                    "p0": [float(value) for value in rib.p0],
                    "p1": [float(value) for value in rib.p1],
                    "length": float(rib.length),
                    "thickness": float(thickness),
                })
                self.log.append(
                    f"rationalization post-filter Eq.7 attempt {attempt} rib {index:02d}: "
                    f"name={rib.name}, L={rib.length:.9g}, t={thickness:.9g}"
                )
            self.rationalization_history.append({
                "event": "post_filter_geometry",
                "attempt": attempt,
                "threshold": float(tref),
                "compliance": (
                    None if reduced_result is None else float(reduced_result.compliance)
                ),
                "acceptance_limit": float(acceptance_limit),
                "accepted": accepted,
                "fea": 0 if reduced_result is None else geometry_fea,
                "ribs": final_rib_data,
            })
            if accepted:
                self.log.append(
                    f"rationalization accepted on deletion attempt {attempt}/{maximum_attempts}: "
                    f"tref={tref:.7g}, ribs={len(reduced_ribs)}, "
                    f"C={reduced_result.compliance:.7g} <= "
                    f"Cref={acceptance_limit:.7g}; Eq.18 FEA={eq18_fea}, "
                    f"deletion FEA={deletion_fea}, Eq.7 FEA={geometry_fea}, "
                    f"total={self.analysis_count-phase_start}"
                )
                return reduced_ribs, reduced_t, reduced_result

            if reduced_result is not None:
                self.log.append(
                    f"rationalization deletion attempt {attempt}/{maximum_attempts} rejected: "
                    f"C={reduced_result.compliance:.7g} > Cref={acceptance_limit:.7g}"
                )

            recovery_round += 1
            ranked_indices = sorted(
                removal_indices,
                key=lambda index: (-float(saved_t[index]), int(index)),
            )
            seed_indices = ranked_indices[:recovery_seed_target]
            seed_mask = np.zeros(len(saved_ribs), dtype=bool)
            seed_mask[seed_indices] = True
            restore_mask = seed_mask.copy()
            if self.mirror_axes:
                restore_mask = self._mirror_expand_mask(
                    saved_ribs, restore_mask
                )
            restore_indices = sorted(
                index for index in removal_indices
                if restore_mask[index]
            )
            for restore_index in restore_indices:
                removal_indices.remove(restore_index)
            seed_names = [saved_ribs[index].name for index in seed_indices]
            restored_names = [saved_ribs[index].name for index in restore_indices]
            self.rationalization_history.append({
                "event": "restoration_after_failed_validation",
                "attempt": attempt,
                "round": recovery_round,
                "strategy": "fixed_initial_deleted_seed_count_thickest_first",
                "n_rrib": n_rrib,
                "seed_target": recovery_seed_target,
                "seed_target_formula": "max(1, ceil(n_rrib/3))",
                "seed_target_basis": "initial_deleted_rib_count",
                "ranking": "saved_eq18_thickness_descending_then_original_index",
                "mirror_completion": bool(self.mirror_axes),
                "mirror_axes": list(self.mirror_axes),
                "ranked_currently_deleted_indices": [
                    index+1 for index in ranked_indices
                ],
                "ranked_currently_deleted_names": [
                    saved_ribs[index].name for index in ranked_indices
                ],
                "ranked_currently_deleted_thicknesses": [
                    float(saved_t[index]) for index in ranked_indices
                ],
                "seed_indices": [index+1 for index in seed_indices],
                "seed_names": seed_names,
                "seed_thicknesses": [
                    float(saved_t[index]) for index in seed_indices
                ],
                "restored_indices": [index+1 for index in restore_indices],
                "restored_names": restored_names,
                "actual_restored_count": len(restore_indices),
                "remaining_removed_indices": [
                    index+1 for index in sorted(removal_indices)
                ],
                "remaining_removed_names": [
                    saved_ribs[index].name for index in sorted(removal_indices)
                ],
            })
            self.log.append(
                f"rationalization restoration after failed attempt {attempt}: "
                f"round={recovery_round}, "
                "strategy=fixed_initial_deleted_seed_count_thickest_first, "
                f"n_rrib={n_rrib}, seed_target={recovery_seed_target}, "
                f"mirror_completion={bool(self.mirror_axes)}, "
                f"mirror_axes={list(self.mirror_axes)}, "
                f"seed_indices={[index+1 for index in seed_indices]}, "
                f"seed_names={seed_names}, "
                f"seed_thicknesses={[float(saved_t[index]) for index in seed_indices]}, "
                f"restored_indices={[index+1 for index in restore_indices]}, "
                f"restored_names={restored_names}, "
                f"actual_restored_count={len(restore_indices)}"
            )
            if not removal_indices:
                self.log.append(
                    "rationalization restoration left no rib deleted; "
                    "no further Eq.7 recovery solve is required"
                )
                break
            self.log.append(
                f"rationalization retrying Eq.7 after recovery round "
                f"{recovery_round} fixed-seed restoration"
            )

        cost_text = ", ".join(
            f"attempt {attempt}: deletion={deletion_fea}, Eq.7={geometry_fea}"
            for attempt, deletion_fea, geometry_fea in attempt_costs
        )
        self.log.append(
            "rationalization failed to accept a rib-deleted design; restored "
            f"pre-rationalization geometry result (Eq.18 FEA={eq18_fea}; "
            f"{cost_text}; total={self.analysis_count-phase_start})"
        )
        return base_ribs, base_t, geometry_result

    def run(self, initial_ribs: list[Rib], candidates: Sequence[Rib], extra_relaxation: float | None = None) -> OptimizationRun:
        geometry_move_step = self._positive_move_limit_setting(
            "geometry_move_limit_initial", 0.50
        )
        self._positive_move_limit_setting(
            "rationalization_move_limit_initial", 0.50
        )
        run = OptimizationRun(log=self.log)
        start_count = self.analysis_count
        thicknesses, result = self.size(initial_ribs)
        ribs = list(initial_ribs)
        run.stages.append(Stage("initial_sizing", list(ribs), thicknesses.copy(), result.compliance, self.analysis_count - start_count))

        phase_start = self.analysis_count
        ribs, thicknesses, result = self.adapt(ribs, thicknesses, result, candidates)
        run.active_history = list(self.active_history)
        run.stages.append(Stage("adaptive", list(ribs), thicknesses.copy(), result.compliance, self.analysis_count - phase_start))

        phase_start = self.analysis_count
        ribs, thicknesses, result = self.optimize_geometry(
            ribs, thicknesses, result,
            initial_move_step=geometry_move_step,
        )
        run.stages.append(Stage("geometry", list(ribs), thicknesses.copy(), result.compliance, self.analysis_count - phase_start))

        phase_start = self.analysis_count
        relaxation = float(self.cfg.get("rationalization_relaxation", 0.0) if extra_relaxation is None else extra_relaxation)
        ribs, thicknesses, result = self.rationalize(ribs, thicknesses, result, relaxation)
        run.rationalization_histories["rationalized"] = [
            dict(event) for event in self.rationalization_history
        ]
        run.stages.append(Stage(
            "rationalized", list(ribs), thicknesses.copy(), result.compliance,
            self.analysis_count - phase_start,
            f"rho={relaxation:g}; input=geometry",
        ))

        further_relaxation = self.cfg.get("further_rationalization_relaxation")
        if further_relaxation is not None:
            phase_start = self.analysis_count
            further_relaxation = float(further_relaxation)
            # Deliberately pass the first rationalized design and its analysis
            # result into the next pass; do not restart from the geometry stage.
            ribs, thicknesses, result = self.rationalize(
                ribs, thicknesses, result, further_relaxation
            )
            run.rationalization_histories["further_rationalized"] = [
                dict(event) for event in self.rationalization_history
            ]
            run.stages.append(Stage(
                "further_rationalized", list(ribs), thicknesses.copy(),
                result.compliance, self.analysis_count - phase_start,
                f"rho={further_relaxation:g}; input=rationalized",
            ))

        # Preserve the legacy flat history while making pass provenance
        # explicit.  Per-pass histories remain available on OptimizationRun.
        self.rationalization_history = [
            {"pass_name": pass_name, **event}
            for pass_name, history in run.rationalization_histories.items()
            for event in history
        ]
        return run
