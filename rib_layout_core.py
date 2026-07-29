"""Common finite-element and rib-layout optimization workflow.

The four ``exampleN.py`` files contain only case-specific data.  This module
provides their shared configuration, model construction, rib generation,
optimization sequence, command-line handling, and public numerical API.

The tested element-level implementations remain in the private ``ribopt``
package.  Keeping those focused modules separate avoids turning this public
facade into an unmaintainable several-thousand-line source file.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import importlib
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from rib_layout_env import (
    DEFAULT_OUTPUT_ROOT,
    LINEAR_SOLVER,
    LINEAR_SOLVER_THREADS,
    SENSITIVITY_WORKERS,
    configure_runtime,
)

configure_runtime()

from rib_layout_algorithms.model import AnalysisResult, Rib, StiffenedPlateModel
from rib_layout_algorithms.model_shell import ShellStiffenedPlateModel
from rib_layout_algorithms.symmetry import mirror_axes, missing_mirror_partners
from rib_layout_algorithms.optimization import (
    OptimizationRun, RibLayoutOptimizer, Stage,
)


COMMON_CONFIG: dict[str, Any] = {
    "analysis_model": "shell",
    "interface_subdivisions_per_cell": 4,
    "rib_cache_max_entries": 128,
    "candidate_operator_cache_max_entries": 512,
    "rib_basis_cache_max_entries": 128,
    "linear_solver": LINEAR_SOLVER,
    "linear_solver_threads": LINEAR_SOLVER_THREADS,
    "rib": {
        "segments": 10,
        "lower": 0.000001,
    },
    "algorithm": {
        "sensitivity_workers": SENSITIVITY_WORKERS,
        "filter_tolerance": 0.01,
        "filter_max_decade": 5,
        "filter_threshold_ratios": [0.10, 0.01, 0.001, 0.0001, 0.00001],
        "short_rib_shell_cells": 3.0,
        "short_rib_cell_fraction": 0.25,
        "short_rib_thickness_factor": 5.0,
        "sizing_max_iterations": 100,
        "sca_objective_tolerance": 0.005,
        "sca_constraint_tolerance": 0.001,
        "sca_design_tolerance": 0.001,
        "sca_design_guard_tolerance": 0.010,
        "sca_consecutive_convergence_steps": 2,
        "sca_min_iterations": 2,
        "move_limit_initial": 0.50,
        "move_limit_direction_increase": 1.20,
        "move_limit_direction_decrease": 0.70,
        "move_limit_direction_zero_tolerance": 1.0e-6,
        "move_limit_unsuccessful_decrease": 0.75,
        "move_limit_maximum_global": 10.0,
        "geometry_sca_proximal": 0.20,
        "rationalization_sca_proximal": 0.20,
        "rationalization_dual_tolerance": 1.0e-9,
        "active_set_max_iterations": 12,
        "additions_per_iteration": 2,
        "addition_factor_min_ratio": 0.70,
        "addition_sizing_improvement_min": 0.01,
        "active_cycle_improvement_min": 0.01,
        "candidate_max_span_cells": 3,
        "candidate_directions": [
            [1, 0], [0, 1], [1, 1], [1, -1],
            [2, 1], [2, -1], [1, 2], [1, -2],
        ],
        "geometry_max_iterations": 100,
        "geometry_fd_fraction": 0.0002,
        "rationalization_beta": 10.0,
        "rationalization_beta_initial": 1.0,
        "rationalization_beta_increment": 1.0,
        "rationalization_compliance_tolerance": 0.001,
        "rationalization_move_limit_initial": 0.50,
        "rationalization_min_iterations": 11,
        "rationalization_max_iterations": 100,
        "rationalization_geometry_iterations": 100,
    },
}


def merge_config(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """Recursively merge configuration mappings without mutating either."""
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def make_case_config(
    case_config: Mapping[str, Any],
    *,
    quick: bool = False,
    geometry_sweeps: int | None = None,
) -> dict:
    """Create one validated run configuration from common and case data."""
    cfg = merge_config(COMMON_CONFIG, case_config)
    if quick and "quick_mesh" in cfg:
        cfg["mesh"] = deepcopy(cfg["quick_mesh"])
    if quick and "quick_rib_segments" in cfg:
        cfg["rib"]["segments"] = int(cfg["quick_rib_segments"])
    if geometry_sweeps is not None:
        cfg["algorithm"]["geometry_max_iterations"] = int(geometry_sweeps)
    required = (
        "number", "name", "domain", "mesh", "wall_thickness", "material",
        "initial_rib_cell_size", "load_cases", "supports", "rib",
        "volume_bound", "rationalization_relaxation",
    )
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"case configuration is missing: {missing}")
    if "further_rationalization_relaxation" in cfg:
        further_relaxation = float(cfg["further_rationalization_relaxation"])
        if further_relaxation < 0.0:
            raise ValueError(
                "further_rationalization_relaxation must be nonnegative"
            )
    return cfg


def load_case(number: int, quick: bool = False, config_path=None) -> dict:
    """Load one case from its dedicated module.

    ``config_path`` is accepted only to give legacy callers a clear migration
    error; case configuration now belongs in ``example1.py`` through
    ``example4.py`` rather than a combined YAML file.
    """
    if config_path is not None:
        raise ValueError(
            "external combined config files are no longer supported; edit "
            f"ribopt/example{int(number)}.py instead"
        )
    if int(number) not in range(1, 5):
        raise KeyError(f"unknown example {number}")
    module = importlib.import_module(f"example{int(number)}")
    return make_case_config(module.CASE_CONFIG, quick=quick)


def build_model(cfg: dict) -> StiffenedPlateModel:
    """Construct the configured shell FE model."""
    width, height = cfg["domain"]
    nx, ny = cfg["mesh"]
    model_class = (
        ShellStiffenedPlateModel
        if cfg.get("analysis_model", "shell") == "shell"
        else StiffenedPlateModel
    )
    model_arguments = dict(
        width=width,
        height=height,
        nx=nx,
        ny=ny,
        wall_thickness=cfg["wall_thickness"],
        E=cfg["material"]["E"],
        nu=cfg["material"]["nu"],
        loads=cfg["load_cases"],
        supports=cfg["supports"],
    )
    if model_class is StiffenedPlateModel:
        return model_class(**model_arguments)
    return model_class(
        **model_arguments,
        interface_subdivisions_per_cell=cfg.get(
            "interface_subdivisions_per_cell", 2
        ),
        rib_cache_max_entries=cfg.get("rib_cache_max_entries", 128),
        candidate_operator_cache_max_entries=cfg.get(
            "candidate_operator_cache_max_entries", 512
        ),
        rib_basis_cache_max_entries=cfg.get("rib_basis_cache_max_entries", 128),
        sensitivity_workers=cfg["algorithm"].get("sensitivity_workers", 1),
        linear_solver=cfg.get("linear_solver", "auto"),
        linear_solver_threads=cfg.get("linear_solver_threads", 1),
    )


def initial_ribs(cfg: dict) -> list[Rib]:
    """Generate both diagonals of every initial square lattice cell."""
    width, height = map(float, cfg["domain"])
    cell = float(cfg["initial_rib_cell_size"])
    nx_float, ny_float = width/cell, height/cell
    nx, ny = int(round(nx_float)), int(round(ny_float))
    if not np.isclose(nx_float, nx) or not np.isclose(ny_float, ny):
        raise ValueError(
            "domain dimensions must be integer multiples of "
            "initial_rib_cell_size"
        )
    rib_height = cfg["rib"]["height"]
    segments = cfg["rib"]["segments"]
    ribs: list[Rib] = []
    counter = 0
    for iy in range(ny):
        y0, y1 = iy*cell, (iy+1)*cell
        for ix in range(nx):
            x0, x1 = ix*cell, (ix+1)*cell
            counter += 1
            ribs.append(Rib(
                (x0, y0), (x1, y1), rib_height, f"D{counter}+", segments
            ))
            counter += 1
            ribs.append(Rib(
                (x0, y1), (x1, y0), rib_height, f"D{counter}-", segments
            ))
    missing = missing_mirror_partners(
        ribs, mirror_axes(cfg), width, height
    )
    if missing:
        raise RuntimeError(f"initial rib structure is not mirror-closed: {missing}")
    return ribs


def candidate_ribs(cfg: dict) -> list[Rib]:
    """Generate the shared finite lattice candidate ground structure."""
    width, height = map(float, cfg["domain"])
    cell = float(cfg["initial_rib_cell_size"])
    nx, ny = int(round(width/cell)), int(round(height/cell))
    rib_height = cfg["rib"]["height"]
    segments = cfg["rib"]["segments"]
    directions = [
        tuple(map(int, direction))
        for direction in cfg["algorithm"]["candidate_directions"]
    ]
    max_span = int(cfg["algorithm"]["candidate_max_span_cells"])
    candidates: list[Rib] = []
    seen: set[tuple] = set()
    counter = 0
    for iy in range(ny+1):
        for ix in range(nx+1):
            for dx, dy in directions:
                for span in range(1, max_span+1):
                    jx, jy = ix+span*dx, iy+span*dy
                    if not (0 <= jx <= nx and 0 <= jy <= ny):
                        continue
                    p0 = (ix*cell, iy*cell)
                    p1 = (jx*cell, jy*cell)
                    key = tuple(sorted((p0, p1)))
                    if key in seen:
                        continue
                    seen.add(key)
                    counter += 1
                    candidates.append(Rib(
                        p0, p1, rib_height, f"C{counter}", segments
                    ))
    missing = missing_mirror_partners(
        candidates, mirror_axes(cfg), width, height
    )
    if missing:
        raise RuntimeError(f"candidate rib structure is not mirror-closed: {missing}")
    return candidates


@dataclass
class RunArtifacts:
    """Numerical objects and timing returned by the shared workflow."""

    config: dict
    optimizer: RibLayoutOptimizer
    run: OptimizationRun
    elapsed_seconds: float


def run_optimization(config: dict) -> RunArtifacts:
    """Build the FE model and execute the common four-stage optimization."""
    model = build_model(config)
    optimizer = RibLayoutOptimizer(model, config)
    started = time.perf_counter()
    run = optimizer.run(initial_ribs(config), candidate_ribs(config))
    elapsed = time.perf_counter() - started
    return RunArtifacts(config, optimizer, run, elapsed)


def run_example_cli(
    case_config: Mapping[str, Any],
    output_options: Mapping[str, Any],
    argv: Sequence[str] | None = None,
) -> int:
    """Command-line entry shared by ``example1.py`` through ``example4.py``."""
    number = int(case_config["number"])
    parser = argparse.ArgumentParser(
        description=f"Run rib-layout optimization Example {number}"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use the case's reduced mesh when one is defined",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="root directory for optimization results",
    )
    parser.add_argument(
        "--geometry-sweeps",
        type=int,
        default=None,
        help="override simultaneous geometry-optimization iterations",
    )
    args = parser.parse_args(argv)

    config = make_case_config(
        case_config,
        quick=args.quick,
        geometry_sweeps=args.geometry_sweeps,
    )
    artifacts = run_optimization(config)

    from rib_layout_output import save_optimization_results

    output_dir = save_optimization_results(
        artifacts, Path(args.output), output_options, quick=args.quick
    )
    print(
        f"Example {number} completed: mesh={config['mesh']}, "
        f"FEA={artifacts.optimizer.analysis_count}, "
        f"elapsed={artifacts.elapsed_seconds:.2f} s"
    )
    print(f"Results: {output_dir}")
    return 0


__all__ = [
    "AnalysisResult", "COMMON_CONFIG", "OptimizationRun", "Rib",
    "RibLayoutOptimizer", "RunArtifacts", "Stage", "build_model",
    "candidate_ribs", "initial_ribs", "load_case", "make_case_config", "merge_config",
    "run_example_cli", "run_optimization",
]
