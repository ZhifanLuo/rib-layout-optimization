"""Optimize the complete finite candidate pool as an internal baseline.

Sizing-only remains the legacy default.  Optional geometry and rationalization
stages use the same production routines, while phase termination and process
failure are recorded separately.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rib_layout_core import build_model, candidate_ribs, load_case
from rib_layout_algorithms.optimization import RibLayoutOptimizer
from tools.run_robustness_study import source_provenance
from rib_layout_serialization import strict_json_dumps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", type=int, nargs="+", required=True, choices=(1, 2, 3, 4)
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument(
        "--post-sizing-policy",
        choices=("sizing-only", "geometry", "rationalization"),
        default="sizing-only",
    )
    parser.add_argument("--geometry-max-iterations", type=int, default=None)
    parser.add_argument("--geometry-move-step", type=float, default=None)
    parser.add_argument("--rationalization-relaxation", type=float, default=None)
    return parser


def expanded_system_size(model, ribs) -> int:
    if not hasattr(model, "rib_bottom_points"):
        return int(model.ndof)
    return int(model.ndof + sum(
        6*len(model.rib_bottom_points(rib)) for rib in ribs
    ))


def stage_metrics(
    name, ribs, thicknesses, result, optimizer, analyses, elapsed_seconds,
    geometry_termination_reason=None, phase_termination_reason=None,
    phase_converged=None, note="",
) -> dict:
    return {
        "name": name,
        "rib_count": len(ribs),
        "compliance": float(result.compliance),
        "volume": optimizer.volume(ribs, thicknesses),
        "analyses": int(analyses),
        "elapsed_seconds": float(elapsed_seconds),
        "phase_process_status": "complete",
        "phase_termination_reason": phase_termination_reason,
        "phase_converged": phase_converged,
        "geometry_termination_reason": geometry_termination_reason,
        "geometry_converged": (
            None if geometry_termination_reason is None
            else bool(geometry_termination_reason == "converged")
        ),
        "note": note,
        "ribs": [
            {
                "name": rib.name,
                "p0": list(rib.p0),
                "p1": list(rib.p1),
                "height": rib.height,
                "thickness": float(thickness),
            }
            for rib, thickness in zip(ribs, thicknesses)
        ],
    }


def sizing_termination_reason(optimizer: RibLayoutOptimizer) -> str:
    """Recover the sizing phase's explicit production log disposition."""
    if any(line.startswith("sizing SCA converged:") for line in optimizer.log):
        return "converged"
    if any(
        line.startswith("sizing SCA warning: outer iteration limit")
        for line in optimizer.log
    ):
        return "iteration_limit"
    return "completed_reason_unavailable"


def rationalization_termination_reason(
    history: list[dict], relaxation: float, input_rib_count: int,
    output_rib_count: int,
) -> str:
    """Classify rationalization outcome separately from inner Eq. (7)."""
    if float(relaxation) <= 0.0:
        return "disabled_nonpositive_relaxation"
    if int(input_rib_count) <= 3:
        return "omitted_already_simple"
    if any(
        event.get("event") == "post_filter_geometry"
        and bool(event.get("accepted"))
        for event in history
    ):
        return "accepted_rib_deleted_design"
    if not any(event.get("event") == "filtering_attempt" for event in history):
        return "no_deletion_candidate"
    if int(output_rib_count) == int(input_rib_count):
        return "restored_input_no_accepted_deletion"
    return "completed_unclassified"


def failed_case_record(
    case_number: int,
    started_utc: str,
    started: float,
    exc: Exception,
    cfg: dict | None = None,
    provenance: dict | None = None,
) -> dict:
    return {
        "case": int(case_number),
        "process_status": "failed",
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter()-started,
        "config": cfg,
        "executable_provenance": provenance,
        "process_status_definition": (
            "failed means a Python exception interrupted this case; any "
            "optimizer convergence information is therefore unavailable"
        ),
        "sizing_termination_reason": None,
        "main_geometry_termination_reason": None,
        "rationalization_termination_reason": None,
        "post_rationalization_geometry_termination_reason": None,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
    }


def run_case(case_number: int, args: argparse.Namespace, provenance: dict) -> dict:
    started_utc = datetime.now(timezone.utc).isoformat()
    run_started = time.perf_counter()
    cfg = load_case(case_number, quick=args.quick)
    ribs = candidate_ribs(cfg)
    candidate_count = len(ribs)
    model = build_model(cfg)
    optimizer = RibLayoutOptimizer(model, cfg)
    system_dofs = expanded_system_size(model, ribs)

    stage_started = time.perf_counter()
    count_before = optimizer.analysis_count
    thicknesses, result = optimizer.size(ribs, maxiter=args.max_iterations)
    sizing_reason = sizing_termination_reason(optimizer)
    stages = [stage_metrics(
        "sizing", ribs, thicknesses, result, optimizer,
        optimizer.analysis_count-count_before,
        time.perf_counter()-stage_started,
        phase_termination_reason=sizing_reason,
        phase_converged=(
            True if sizing_reason == "converged"
            else False if sizing_reason == "iteration_limit" else None
        ),
    )]
    main_reason = None
    rationalization_reason = None
    post_rationalization_reason = None
    if args.post_sizing_policy in {"geometry", "rationalization"}:
        stage_started = time.perf_counter()
        count_before = optimizer.analysis_count
        move_step = (
            float(args.geometry_move_step)
            if args.geometry_move_step is not None
            else float(cfg["algorithm"].get("geometry_move_limit_initial", 0.50))
        )
        ribs, thicknesses, result = optimizer.optimize_geometry(
            list(ribs), thicknesses, result,
            max_iterations_override=args.geometry_max_iterations,
            initial_move_step=move_step,
        )
        main_reason = optimizer.geometry_termination_reason
        stages.append(stage_metrics(
            "geometry", ribs, thicknesses, result, optimizer,
            optimizer.analysis_count-count_before,
            time.perf_counter()-stage_started,
            geometry_termination_reason=main_reason,
            phase_termination_reason=main_reason,
            phase_converged=bool(main_reason == "converged"),
        ))
    if args.post_sizing_policy == "rationalization":
        stage_started = time.perf_counter()
        count_before = optimizer.analysis_count
        relaxation = (
            float(args.rationalization_relaxation)
            if args.rationalization_relaxation is not None
            else float(cfg.get("rationalization_relaxation", 0.0))
        )
        # Do not let the main-geometry reason leak into this phase when
        # rationalization performs no post-deletion geometry solve.
        optimizer.geometry_termination_reason = None
        rationalization_input_count = len(ribs)
        ribs, thicknesses, result = optimizer.rationalize(
            list(ribs), thicknesses, result, relaxation
        )
        post_rationalization_reason = optimizer.geometry_termination_reason
        rationalization_reason = rationalization_termination_reason(
            optimizer.rationalization_history, relaxation,
            rationalization_input_count, len(ribs),
        )
        stages.append(stage_metrics(
            "rationalized", ribs, thicknesses, result, optimizer,
            optimizer.analysis_count-count_before,
            time.perf_counter()-stage_started,
            geometry_termination_reason=post_rationalization_reason,
            phase_termination_reason=rationalization_reason,
            phase_converged=None,
            note=f"rho={relaxation:g}; input=geometry",
        ))
    final_stage = stages[-1]
    constraint_tolerance = float(cfg["algorithm"]["sca_constraint_tolerance"])
    record = {
        "case": case_number,
        "process_status": "complete",
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "executable_provenance": provenance,
        "config": cfg,
        "quick": bool(args.quick),
        "post_sizing_policy": args.post_sizing_policy,
        "process_status_definition": (
            "complete means the requested phases returned without a Python "
            "exception; it does not imply optimizer convergence"
        ),
        "mesh_nx": int(cfg["mesh"][0]),
        "mesh_ny": int(cfg["mesh"][1]),
        "candidate_count": candidate_count,
        "ground_system_dofs": int(model.ndof),
        "expanded_system_dofs": system_dofs,
        "compliance": final_stage["compliance"],
        "volume": final_stage["volume"],
        "rib_count": final_stage["rib_count"],
        "volume_bound": float(cfg["volume_bound"]),
        "final_feasible": bool(
            final_stage["volume"] <= float(cfg["volume_bound"])*(1+constraint_tolerance)
        ),
        "analysis_count": optimizer.analysis_count,
        "elapsed_seconds": time.perf_counter()-run_started,
        "best_feasible_outer": int(
            optimizer.sizing_history[-1]["best_feasible_outer"]
            if optimizer.sizing_history else 0
        ),
        "sizing_termination_reason": sizing_reason,
        "sizing_converged": (
            True if sizing_reason == "converged"
            else False if sizing_reason == "iteration_limit" else None
        ),
        "main_geometry_termination_reason": main_reason,
        "main_geometry_converged": (
            None if main_reason is None else bool(main_reason == "converged")
        ),
        "rationalization_termination_reason": rationalization_reason,
        "post_rationalization_geometry_termination_reason": (
            post_rationalization_reason
        ),
        "post_rationalization_geometry_converged": (
            None if post_rationalization_reason is None
            else bool(post_rationalization_reason == "converged")
        ),
        "stages": stages,
        "ribs": final_stage["ribs"],
        "sizing_history": optimizer.sizing_history,
        "rationalization_history": optimizer.rationalization_history,
        "comparison_scope": (
            "internal full-enumerated-pool baseline; search paths and "
            "computational work remain unequal to the active-set route"
        ),
        "log": optimizer.log,
    }
    print(
        f"Case {case_number}: candidates={candidate_count}, dofs={system_dofs}, "
        f"policy={args.post_sizing_policy}, C={result.compliance:.9g}, "
        f"ribs={len(ribs)}, FEA={optimizer.analysis_count}"
    )
    return record


def write_summary(output: Path, records: list[dict]) -> None:
    fields = [
        "case", "process_status", "quick", "post_sizing_policy", "mesh_nx",
        "mesh_ny", "candidate_count", "ground_system_dofs",
        "expanded_system_dofs", "compliance", "volume", "rib_count",
        "volume_bound", "final_feasible", "analysis_count", "elapsed_seconds",
        "best_feasible_outer", "main_geometry_termination_reason",
        "sizing_termination_reason", "rationalization_termination_reason",
        "post_rationalization_geometry_termination_reason", "error",
    ]
    with (output/"full_pool_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: record.get(key) for key in fields} for record in records)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    aggregate_started_utc = datetime.now(timezone.utc).isoformat()
    if args.geometry_max_iterations is not None and args.geometry_max_iterations < 0:
        raise ValueError("geometry-max-iterations must be nonnegative")
    if args.geometry_move_step is not None and args.geometry_move_step <= 0.0:
        raise ValueError("geometry-move-step must be positive")
    if (
        args.rationalization_relaxation is not None
        and args.rationalization_relaxation < 0.0
    ):
        raise ValueError("rationalization relaxation must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = source_provenance(
        Path(__file__).resolve().parents[1], Path(__file__)
    )
    records: list[dict] = []
    failed = False
    for case_number in args.case:
        started_utc = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        cfg = None
        try:
            cfg = load_case(case_number, quick=args.quick)
            record = run_case(case_number, args, provenance)
        except Exception as exc:
            failed = True
            record = failed_case_record(
                case_number, started_utc, started, exc, cfg=cfg,
                provenance=provenance,
            )
        records.append(record)
        (args.output/f"full_pool_case_{case_number}.json").write_text(
            strict_json_dumps(record, indent=2), encoding="utf-8"
        )
        write_summary(args.output, records)
    aggregate = {
        "schema_version": 2,
        "process_status": "complete_with_failures" if failed else "complete",
        "started_utc": aggregate_started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "process_status_definition": (
            "complete means every requested case returned without a Python "
            "exception; phase convergence is reported separately"
        ),
        "executable_provenance": provenance,
        "cases": records,
    }
    (args.output/"full_pool_results.json").write_text(
        strict_json_dumps(aggregate, indent=2), encoding="utf-8"
    )
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
