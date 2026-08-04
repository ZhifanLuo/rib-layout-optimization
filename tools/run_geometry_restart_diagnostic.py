"""Repeat Eq. (7) from a saved stage, with deterministic multistarts."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rib_layout_core import build_model, load_case
from rib_layout_algorithms.model import Rib
from rib_layout_algorithms.optimization import RibLayoutOptimizer
from rib_layout_algorithms.symmetry import build_mirror_variable_map, mirror_axes
from tools.run_robustness_study import source_provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, required=True, choices=(1, 2, 3, 4))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("adaptive", "geometry", "rationalized", "further_rationalized"),
        default="geometry",
        help="Saved stage used as the initial design (default: geometry).",
    )
    parser.add_argument(
        "--restarts", type=int, default=1,
        help="Number of unperturbed repeat runs (default: 1).",
    )
    parser.add_argument(
        "--multistarts", type=int, default=0,
        help="Number of symmetry-preserving perturbed starts.",
    )
    parser.add_argument(
        "--thickness-perturbation", type=float, default=0.10,
        help="Log-normal standard deviation for multistart thicknesses.",
    )
    parser.add_argument(
        "--endpoint-perturbation", type=float, default=0.0,
        help=(
            "Normal standard deviation of endpoint coordinates as a fraction "
            "of one local shell-cell dimension (default: 0, disabled)."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument(
        "--initial-move-step", type=float, default=None,
        help="Optional initial global geometry move step.",
    )
    parser.add_argument(
        "--source-compliance-tolerance", type=float, default=1.0e-6,
        help="Maximum relative mismatch after saved-stage reanalysis.",
    )
    return parser


def stage_design(source: dict, stage_name: str, segments: int) -> tuple[list[Rib], np.ndarray, dict]:
    stage = next(
        (item for item in source.get("stages", ()) if item.get("name") == stage_name),
        None,
    )
    if stage is None:
        available = [item.get("name") for item in source.get("stages", ())]
        raise ValueError(
            f"saved stage {stage_name!r} is absent; available stages: {available}"
        )
    ribs = [
        Rib(
            tuple(item["p0"]), tuple(item["p1"]), float(item["height"]),
            item["name"], segments,
        )
        for item in stage["ribs"]
    ]
    thicknesses = np.asarray(
        [item["thickness"] for item in stage["ribs"]], float
    )
    return ribs, thicknesses, stage


def validate_source_metadata(source: dict, case_number: int, cfg: dict) -> None:
    if int(source.get("example", -1)) != int(case_number):
        raise ValueError("saved result case mismatch")
    if list(map(int, source.get("mesh_used", ()))) != list(map(int, cfg["mesh"])):
        raise ValueError("saved result mesh mismatch")
    if bool(source.get("quick", False)):
        raise ValueError("restart diagnostics require a non-quick formal source")


def validate_saved_compliance(
    stage_name: str,
    saved: float,
    reanalyzed: float,
    tolerance: float,
) -> float:
    mismatch = abs(float(reanalyzed)/max(abs(float(saved)), 1.0e-30)-1.0)
    if mismatch > float(tolerance):
        raise RuntimeError(
            f"saved stage {stage_name!r} is incompatible with the current "
            f"executable: relative compliance mismatch={mismatch:.6g}"
        )
    return mismatch


def restart_initial_volume(
    optimizer: RibLayoutOptimizer,
    initial_ribs: list[Rib],
    initial_t: np.ndarray,
) -> float:
    """Use the actual perturbed geometry for restart-volume accounting."""
    return optimizer.volume(initial_ribs, initial_t)


def failed_restart_record(
    run_index: int,
    kind: str,
    started_utc: str,
    started: float,
    exc: Exception,
) -> dict:
    return {
        "run": int(run_index),
        "kind": kind,
        "process_status": "failed",
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "optimizer_converged": False,
        "elapsed_seconds": time.perf_counter()-started,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
    }


def rib_records(ribs: list[Rib], thicknesses: np.ndarray) -> list[dict]:
    return [
        {
            "name": rib.name,
            "p0": list(rib.p0),
            "p1": list(rib.p1),
            "height": rib.height,
            "thickness": float(thickness),
        }
        for rib, thickness in zip(ribs, thicknesses)
    ]


def perturbed_thicknesses(
    optimizer: RibLayoutOptimizer,
    ribs: list[Rib],
    thicknesses: np.ndarray,
    rng: np.random.Generator,
    sigma: float,
) -> np.ndarray:
    """Perturb independent thickness variables while preserving symmetry."""
    variable_map = build_mirror_variable_map(
        ribs, mirror_axes(optimizer.cfg), *map(float, optimizer.cfg["domain"])
    )
    reduced = variable_map.reduce_thicknesses(thicknesses)
    reduced *= np.exp(float(sigma)*rng.standard_normal(len(reduced)))
    reduced = np.clip(reduced, optimizer.t_lower, optimizer.t_upper)
    return optimizer._feasible_start(
        ribs, variable_map.expand_thicknesses(reduced)
    )


def perturbed_ribs(
    optimizer: RibLayoutOptimizer,
    ribs: list[Rib],
    rng: np.random.Generator,
    sigma: float,
) -> list[Rib]:
    """Perturb independent endpoint variables within bounds and symmetry.

    ``sigma`` is nondimensional: one unit is one ground-shell cell in the
    corresponding coordinate direction.  The affine mirror-variable map is
    used before clipping, so every returned layout satisfies the configured
    reflection relations to roundoff.  A deterministic backoff avoids a
    numerically degenerate rib if a large requested perturbation collapses an
    endpoint pair.
    """
    if float(sigma) == 0.0 or not ribs:
        return list(ribs)
    width, height = map(float, optimizer.cfg["domain"])
    variable_map = build_mirror_variable_map(
        ribs, mirror_axes(optimizer.cfg), width, height
    )
    coordinates = np.asarray([[*rib.p0, *rib.p1] for rib in ribs], float).ravel()
    reduced = variable_map.reduce_coordinates(coordinates)
    full_bounds = np.tile(
        np.asarray([[0.0, width], [0.0, height]], float), (2*len(ribs), 1)
    )
    reduced_bounds = variable_map.reduce_coordinate_bounds(full_bounds)
    full_scale = np.tile(
        np.asarray([optimizer.model.dx, optimizer.model.dy], float), 2*len(ribs)
    )
    reduced_scale = variable_map.reduce_coordinate_scale(full_scale)
    direction = rng.standard_normal(len(reduced))
    minimum_length = 1.0e-6*min(float(optimizer.model.dx), float(optimizer.model.dy))
    factor = float(sigma)
    for _ in range(24):
        trial = np.clip(
            reduced+factor*reduced_scale*direction,
            reduced_bounds[:, 0],
            reduced_bounds[:, 1],
        )
        expanded = variable_map.expand_coordinates(trial).reshape(len(ribs), 4)
        moved = [
            Rib(tuple(point[:2]), tuple(point[2:]), rib.height, rib.name, rib.segments)
            for point, rib in zip(expanded, ribs)
        ]
        if all(rib.length > minimum_length for rib in moved):
            return moved
        factor *= 0.5
    raise RuntimeError("endpoint perturbation could not produce nondegenerate ribs")


def compliance_distribution(records: list[dict]) -> dict:
    """Return publication-oriented best/median/worst run summaries."""
    if not records:
        return {
            "best": None, "median": None, "median_final_compliance": None,
            "worst": None, "feasible_count": 0, "run_count": 0,
            "best_feasible": None,
        }
    ordered = sorted(records, key=lambda item: float(item["final_compliance"]))
    median_value = float(np.median([
        float(record["final_compliance"]) for record in records
    ]))
    median = min(
        records,
        key=lambda item: (
            abs(float(item["final_compliance"])-median_value), int(item["run"])
        ),
    )
    feasible = [record for record in records if bool(record.get("final_feasible", True))]

    def compact(record: dict) -> dict:
        return {
            "run": int(record["run"]),
            "kind": record["kind"],
            "final_compliance": float(record["final_compliance"]),
            "analysis_count": int(record["analysis_count"]),
            "termination_reason": record.get("termination_reason"),
            "optimizer_converged": bool(record.get("optimizer_converged", False)),
            "final_feasible": bool(record.get("final_feasible", True)),
        }

    return {
        "best": compact(ordered[0]),
        "median": compact(median),
        "median_final_compliance": median_value,
        "worst": compact(ordered[-1]),
        "feasible_count": len(feasible),
        "run_count": len(records),
        "best_feasible": compact(min(
            feasible, key=lambda item: float(item["final_compliance"])
        )) if feasible else None,
    }


def write_history(path: Path, history: list[dict]) -> None:
    scalar_keys = sorted({
        key for record in history for key, value in record.items()
        if not isinstance(value, (list, dict))
    })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[*scalar_keys, "response_trials"]
        )
        writer.writeheader()
        for record in history:
            row = {key: record.get(key) for key in scalar_keys}
            row["response_trials"] = json.dumps(
                record.get("response_trials", []), separators=(",", ":")
            )
            writer.writerow(row)


LEGACY_HISTORY_FIELDS = [
    "outer", "objective", "compliance", "volume", "volume_ratio",
    "constraint_violation", "feasible", "objective_relative_change",
    "objective_relative_change_signed", "design_change",
    "thickness_design_change", "coordinate_design_change",
    "maximum_absolute_thickness_change",
    "maximum_absolute_coordinate_change", "predicted_objective_change",
    "true_objective_change", "approximation_ratio", "inner_iterations",
    "move_global_used", "move_global_next", "frozen_geometry_count",
    "frozen_geometry_names", "frozen_geometry_reasons", "step_converged",
    "consecutive_converged",
]


def write_legacy_history(
    path: Path,
    history: list[dict],
    ribs: list[Rib],
    initial_t: np.ndarray,
    initial_compliance: float,
    initial_volume: float,
    volume_bound: float,
    constraint_tolerance: float,
) -> None:
    """Write the exact pre-multistart CSV columns, including the initial row."""
    names = [rib.name for rib in ribs]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([*LEGACY_HISTORY_FIELDS, *[f"t_{name}" for name in names]])
        initial = {
            "outer": 0,
            "objective": float(initial_compliance),
            "compliance": float(initial_compliance),
            "volume": float(initial_volume),
            "volume_ratio": float(initial_volume/volume_bound),
            "constraint_violation": float(max(initial_volume/volume_bound-1.0, 0.0)),
            "feasible": bool(
                initial_volume <= volume_bound*(1.0+constraint_tolerance)
            ),
        }
        writer.writerow(
            [initial.get(field) for field in LEGACY_HISTORY_FIELDS]
            + [float(value) for value in initial_t]
        )
        for record in history:
            thickness_by_name = dict(zip(record["rib_names"], record["thicknesses"]))
            values = []
            for field in LEGACY_HISTORY_FIELDS:
                value = record.get(field)
                values.append(
                    json.dumps(value, separators=(",", ":"))
                    if isinstance(value, (list, dict)) else value
                )
            writer.writerow(values+[thickness_by_name[name] for name in names])


def legacy_single_restart_payload(
    *, case: int, source: Path, stage_name: str, mesh: list[int],
    rib_count: int, saved_compliance: float, record: dict,
) -> dict:
    """Return the legacy top-level JSON schema plus additive new fields."""
    return {
        "case": case,
        "source": str(source.resolve()),
        "source_stage": stage_name,
        "mesh": mesh,
        "rib_count": rib_count,
        "saved_source_compliance": float(saved_compliance),
        "restart_compliance": float(record["initial_compliance"]),
        "restart_initialization_fea": 1,
        "geometry_outer_fea": int(record["geometry_analysis_count"]),
        "outer_iterations": int(record["outer_records"]),
        **record,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    created_utc = datetime.now(timezone.utc).isoformat()
    if args.restarts < 0 or args.multistarts < 0:
        raise ValueError("restart and multistart counts must be nonnegative")
    if args.restarts+args.multistarts < 1:
        raise ValueError("at least one restart or multistart is required")
    if args.thickness_perturbation < 0.0:
        raise ValueError("thickness perturbation must be nonnegative")
    if args.endpoint_perturbation < 0.0:
        raise ValueError("endpoint perturbation must be nonnegative")
    if args.source_compliance_tolerance < 0.0:
        raise ValueError("source compliance tolerance must be nonnegative")

    source = json.loads(args.source.read_text(encoding="utf-8"))
    cfg = load_case(args.case, quick=False)
    validate_source_metadata(source, args.case, cfg)
    ribs, saved_t, stage = stage_design(
        source, args.stage, int(cfg["rib"]["segments"])
    )
    source_optimizer = RibLayoutOptimizer(build_model(cfg), cfg)
    reanalyzed_source = source_optimizer.analyze(ribs, saved_t)
    source_mismatch = validate_saved_compliance(
        args.stage, float(stage["compliance"]),
        float(reanalyzed_source.compliance), args.source_compliance_tolerance,
    )
    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = source_provenance(
        Path(__file__).resolve().parents[1], Path(__file__)
    )
    source_sha256 = hashlib.sha256(args.source.read_bytes()).hexdigest()
    records: list[dict] = []
    saved_coordinates = np.asarray([[*rib.p0, *rib.p1] for rib in ribs], float)

    run_kinds = ["restart"]*args.restarts + ["multistart"]*args.multistarts
    for run_index, kind in enumerate(run_kinds, 1):
        run_started_utc = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        history: list[dict] = []
        try:
            optimizer = RibLayoutOptimizer(build_model(cfg), cfg)
            initial_ribs = list(ribs)
            initial_t = saved_t.copy()
            if kind == "multistart":
                initial_ribs = perturbed_ribs(
                    optimizer, initial_ribs, rng, args.endpoint_perturbation
                )
                initial_t = perturbed_thicknesses(
                    optimizer, initial_ribs, initial_t, rng,
                    args.thickness_perturbation,
                )
            initial_coordinates = np.asarray(
                [[*rib.p0, *rib.p1] for rib in initial_ribs], float
            )
            initial_result = optimizer.analyze(initial_ribs, initial_t)
            start_count = optimizer.analysis_count
            final_ribs, final_t, final_result = optimizer.optimize_geometry(
                initial_ribs, initial_t, initial_result,
                max_iterations_override=args.max_iterations,
                initial_move_step=args.initial_move_step,
                iteration_history=history,
            )
            final_coordinates = np.asarray(
                [[*rib.p0, *rib.p1] for rib in final_ribs], float
            )
            final_volume = optimizer.volume(final_ribs, final_t)
            record = {
                "run": run_index,
                "kind": kind,
                "process_status": "complete",
                "started_utc": run_started_utc,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "optimizer_converged": bool(
                    optimizer.geometry_termination_reason == "converged"
                ),
                "initial_compliance": float(initial_result.compliance),
                "final_compliance": float(final_result.compliance),
                "relative_compliance_change": float(
                    final_result.compliance/initial_result.compliance-1.0
                ),
                "initial_volume": restart_initial_volume(
                    optimizer, initial_ribs, initial_t
                ),
                "final_volume": final_volume,
                "final_feasible": bool(
                    final_volume <= optimizer.volume_bound*(
                        1.0+float(cfg["algorithm"]["sca_constraint_tolerance"])
                    )
                ),
                "analysis_count": optimizer.analysis_count,
                "geometry_analysis_count": optimizer.analysis_count-start_count,
                "outer_records": len(history),
                "termination_reason": optimizer.geometry_termination_reason,
                "rejected_trial_count": sum(
                    sum(
                        not trial["accepted"]
                        for trial in event.get("response_trials", [])
                    )
                    for event in history
                ),
                "elapsed_seconds": time.perf_counter()-started,
                "maximum_initial_endpoint_perturbation_from_saved": float(
                    np.max(np.abs(initial_coordinates-saved_coordinates))
                ),
                "maximum_endpoint_coordinate_change": float(
                    np.max(np.abs(final_coordinates-initial_coordinates))
                ),
                "maximum_thickness_change": float(
                    np.max(np.abs(final_t-initial_t))
                ),
                "initial_ribs": rib_records(initial_ribs, initial_t),
                "final_ribs": rib_records(final_ribs, final_t),
                "iteration_history": history,
                "log": optimizer.log,
            }
            write_history(
                args.output/f"geometry_{kind}_{run_index:02d}_history.csv", history
            )
        except Exception as exc:
            record = failed_restart_record(
                run_index, kind, run_started_utc, started, exc
            )
        record.update({
            "executable_provenance": provenance,
            "config": cfg,
            "source": str(args.source.resolve()),
            "source_sha256": source_sha256,
            "source_stage": args.stage,
        })
        records.append(record)
        (args.output/f"geometry_{kind}_{run_index:02d}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )

    successful = [
        record for record in records if record["process_status"] == "complete"
    ]
    failed = len(successful) != len(records)
    best = min(successful, key=lambda item: item["final_compliance"]) if successful else None
    summary_fields = [
        "run", "kind", "process_status", "optimizer_converged",
        "initial_compliance", "final_compliance",
        "relative_compliance_change", "initial_volume", "final_volume",
        "analysis_count", "geometry_analysis_count", "outer_records",
        "termination_reason", "rejected_trial_count", "elapsed_seconds",
        "maximum_initial_endpoint_perturbation_from_saved",
        "maximum_endpoint_coordinate_change", "maximum_thickness_change", "error",
    ]
    with (args.output/"geometry_restart_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(
            {key: record.get(key) for key in summary_fields} for record in records
        )
    payload = {
        "case": args.case,
        "created_utc": created_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "process_status": "complete_with_failures" if failed else "complete",
        "executable_provenance": provenance,
        "config": cfg,
        "source": str(args.source.resolve()),
        "source_sha256": source_sha256,
        "source_example": source.get("example"),
        "source_mesh": source.get("mesh_used"),
        "source_stage": args.stage,
        "saved_source_compliance": float(stage["compliance"]),
        "reanalyzed_source_compliance": float(reanalyzed_source.compliance),
        "source_relative_compliance_mismatch": source_mismatch,
        "source_reanalysis_count": source_optimizer.analysis_count,
        "source_compliance_tolerance": args.source_compliance_tolerance,
        "mesh": cfg["mesh"],
        "seed": args.seed,
        "restart_count": args.restarts,
        "multistart_count": args.multistarts,
        "thickness_perturbation": args.thickness_perturbation,
        "endpoint_perturbation_cell_fraction": args.endpoint_perturbation,
        "requested_max_iterations": args.max_iterations,
        "requested_initial_move_step": args.initial_move_step,
        "domain": cfg["domain"],
        "mirror_symmetry": list(mirror_axes(cfg)),
        "geometry_algorithm_settings": {
            key: cfg["algorithm"].get(key)
            for key in (
                "geometry_max_iterations", "geometry_fd_fraction",
                "geometry_sca_proximal", "sca_objective_tolerance",
                "sca_constraint_tolerance", "sca_design_tolerance",
            )
        },
        "best_run": best["run"] if best else None,
        "best_final_compliance": best["final_compliance"] if best else None,
        "successful_run_count": len(successful),
        "failed_run_count": len(records)-len(successful),
        "distribution": compliance_distribution(successful),
        "runs": records,
    }
    (args.output/"geometry_restart_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    # Preserve the original single-restart filenames and top-level result
    # fields for existing diagnostic scripts.
    if args.restarts == 1 and args.multistarts == 0 and successful:
        legacy = legacy_single_restart_payload(
            case=args.case,
            source=args.source,
            stage_name=args.stage,
            mesh=cfg["mesh"],
            rib_count=len(ribs),
            saved_compliance=float(stage["compliance"]),
            record=records[0],
        )
        (args.output/f"geometry_from_{args.stage}_results.json").write_text(
            json.dumps(legacy, indent=2), encoding="utf-8"
        )
        write_legacy_history(
            args.output/f"geometry_from_{args.stage}_iteration_history.csv",
            records[0]["iteration_history"],
            ribs,
            np.asarray([item["thickness"] for item in records[0]["initial_ribs"]]),
            float(records[0]["initial_compliance"]),
            float(records[0]["initial_volume"]),
            float(cfg["volume_bound"]),
            float(cfg["algorithm"]["sca_constraint_tolerance"]),
        )
        (args.output/f"geometry_from_{args.stage}_log.txt").write_text(
            "\n".join(records[0]["log"]), encoding="utf-8"
        )
    print(
        f"Case {args.case}: runs={len(records)}, "
        + (
            f"best C={best['final_compliance']:.9g} (run {best['run']})"
            if best else "no successful restart"
        )
    )
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
