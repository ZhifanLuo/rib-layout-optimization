"""Internal enhanced-MMA adaptive move limits for SCA outer iterations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EnhancedMMAMoveLimit:
    """Reproduce the move-limit part of ``OrdMoveLmt`` from enhanced MMA.

    ``update`` is called at every FEA-evaluated outer iterate before building
    the next convex approximation. ``contract`` is used only inside an
    FEA-free convex inner solve when independent bounds would otherwise create
    invalid geometry.
    """

    lower: np.ndarray
    upper: np.ndarray
    step_scale: np.ndarray | None = None
    initial_global_step: float = 0.5
    same_direction_increase: float = 1.2
    oscillation_decrease: float = 0.7
    unsuccessful_decrease: float = 0.75
    maximum_global_step: float = 10.0
    direction_zero_tolerance: float = 1.0e-6
    global_step: float = field(init=False)
    local_step: np.ndarray = field(init=False)
    iteration: int = field(default=0, init=False)
    previous_x: np.ndarray | None = field(default=None, init=False)
    previous_previous_x: np.ndarray | None = field(default=None, init=False)
    previous_objective: float | None = field(default=None, init=False)
    previous_violation: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.lower = np.asarray(self.lower, float).copy()
        self.upper = np.asarray(self.upper, float).copy()
        if self.lower.shape != self.upper.shape or np.any(self.upper < self.lower):
            raise ValueError("invalid global move-limit bounds")
        if self.step_scale is not None:
            self.step_scale = np.asarray(self.step_scale, float).copy()
            if self.step_scale.shape != self.lower.shape:
                raise ValueError("move-limit step scale does not match bounds")
        if self.direction_zero_tolerance < 0.0:
            raise ValueError("move-limit direction zero tolerance must be nonnegative")
        self.global_step = float(self.initial_global_step)
        self.local_step = np.ones_like(self.lower)

    @staticmethod
    def _is_better(
        objective: float,
        violation: float,
        previous_objective: float,
        previous_violation: float,
        feasible_tolerance: float,
    ) -> bool:
        if violation > feasible_tolerance:
            return violation < previous_violation or (
                violation == previous_violation and objective < previous_objective
            )
        if previous_violation > feasible_tolerance:
            return True
        return objective < previous_objective

    def update(
        self,
        x: np.ndarray,
        objective: float,
        violation: float = 0.0,
        feasible_tolerance: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update the move-limit history at the latest FEA-evaluated design."""
        x = np.asarray(x, float)
        if x.shape != self.lower.shape:
            raise ValueError("move-limit design shape does not match bounds")
        self.iteration += 1
        if self.iteration == 1:
            if self.global_step < 1.0e-3 or self.global_step > self.maximum_global_step:
                self.global_step = 0.5
            self.local_step.fill(1.0)
        else:
            assert self.previous_objective is not None and self.previous_violation is not None
            better = self._is_better(
                float(objective), float(violation), self.previous_objective,
                self.previous_violation, float(feasible_tolerance),
            )
            if better:
                self.global_step = min(self.global_step, self.maximum_global_step)
            else:
                self.global_step *= self.unsuccessful_decrease
            if self.iteration > 2 and self.previous_x is not None and self.previous_previous_x is not None:
                # Compare dimensionless moves so that thickness and endpoint
                # coordinates use the same numerical-zero criterion.  If
                # either of two consecutive moves is effectively zero, its
                # sign is dominated by floating-point noise and must not
                # trigger the enhanced-MMA 1.2/0.7 direction update.
                normalization = np.maximum(self.upper-self.lower, 1.0e-30)
                current_move = (x-self.previous_x)/normalization
                previous_move = (
                    self.previous_x-self.previous_previous_x
                )/normalization
                significant = (
                    np.abs(current_move) > self.direction_zero_tolerance
                ) & (
                    np.abs(previous_move) > self.direction_zero_tolerance
                )
                direction_product = current_move*previous_move
                self.local_step[
                    significant & (direction_product < 0.0)
                ] *= self.oscillation_decrease
                self.local_step[
                    significant & (direction_product > 0.0)
                ] *= self.same_direction_increase

        self.previous_previous_x = None if self.previous_x is None else self.previous_x.copy()
        self.previous_x = x.copy()
        self.previous_objective = float(objective)
        self.previous_violation = float(violation)
        return self.current_bounds(x)

    def contract(self) -> None:
        """Contract inner move bounds before resolving an approximation."""
        self.global_step *= self.unsuccessful_decrease

    def current_bounds(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, float)
        variable_range = self.upper-self.lower
        scale = np.abs(x)
        small = scale < 1.0
        scale[small] = np.where(variable_range[small] > 1.0, 1.0, variable_range[small])
        if self.step_scale is not None:
            prescribed = np.isfinite(self.step_scale)
            scale[prescribed] = self.step_scale[prescribed]
        step = scale*self.global_step*self.local_step
        return np.maximum(x-step, self.lower), np.minimum(x+step, self.upper)

    def maximum_normalized_half_width(self, x: np.ndarray) -> float:
        lower, upper = self.current_bounds(x)
        variable_range = np.maximum(self.upper-self.lower, 1.0e-30)
        return float(np.max(np.maximum(np.asarray(x)-lower, upper-np.asarray(x))/variable_range))
