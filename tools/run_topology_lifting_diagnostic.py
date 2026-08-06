"""Reinsert deleted ribs at the lower bound and re-optimize the larger topology."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rib_layout_core import build_model, load_case
from rib_layout_serialization import portable_artifact_path, strict_json_dumps
from rib_layout_algorithms.model import Rib
from rib_layout_algorithms.optimization import RibLayoutOptimizer
from tools.run_geometry_restart_diagnostic import rib_records, stage_design


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, required=True, choices=(1, 2, 3, 4))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-stage", default="geometry")
    parser.add_argument("--reduced-stage", default="rationalized")
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument(
        "--source-compliance-tolerance", type=float, default=1.0e-6,
        help=(
            "Maximum saved-versus-reanalyzed relative compliance difference "
            "for each source stage (default: 1e-6)."
        ),
    )
    return parser


def validate_source_metadata(source: dict, case: int, cfg: dict) -> None:
    """Require source identity and discretization metadata to match the case."""
    source_case = source.get("example", source.get("case"))
    if source_case is None or int(source_case) != int(case):
        raise ValueError(
            f"source case mismatch: expected {case}, found {source_case!r}"
        )
    if "quick" not in source or not isinstance(source["quick"], bool):
        raise ValueError("source must contain Boolean quick metadata")
    if "mesh_used" not in source:
        raise ValueError("source must contain mesh_used metadata")
    source_mesh = list(map(int, source["mesh_used"]))
    configured_mesh = list(map(int, cfg["mesh"]))
    if source_mesh != configured_mesh:
        raise ValueError(
            "source mesh mismatch: "
            f"saved={source_mesh}, configured={configured_mesh}, "
            f"quick={source['quick']}"
        )


def compliance_relative_error(saved: float, reanalyzed: float) -> float:
    return float(abs(float(reanalyzed)-float(saved))/max(abs(float(saved)), 1.0e-16))


def validate_saved_compliance(
    stage_name: str,
    saved: float,
    reanalyzed: float,
    tolerance: float,
) -> float:
    error = compliance_relative_error(saved, reanalyzed)
    if error > tolerance:
        raise RuntimeError(
            f"source stage {stage_name!r} is incompatible with the current "
            f"executable/configuration: saved C={saved:.12g}, "
            f"reanalyzed C={reanalyzed:.12g}, relative difference={error:.3g} "
            f"> tolerance={tolerance:.3g}"
        )
    return error


def lifted_design(
    full_ribs: list[Rib],
    full_t: np.ndarray,
    reduced_ribs: list[Rib],
    reduced_t: np.ndarray,
    lower: float,
) -> tuple[list[Rib], np.ndarray, list[str]]:
    """Use reduced-stage states where present and lower-bound deleted members."""
    del full_t  # The larger topology supplies geometry and ordering, not sizing.
    retained = {
        rib.name: (rib, float(thickness))
        for rib, thickness in zip(reduced_ribs, reduced_t)
    }
    ribs: list[Rib] = []
    thicknesses: list[float] = []
    lifted_names: list[str] = []
    for rib in full_ribs:
        if rib.name in retained:
            kept_rib, kept_t = retained[rib.name]
            ribs.append(kept_rib)
            thicknesses.append(kept_t)
        else:
            ribs.append(rib)
            thicknesses.append(float(lower))
            lifted_names.append(rib.name)
    return ribs, np.asarray(thicknesses, float), lifted_names


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.source_compliance_tolerance < 0.0:
        raise ValueError("source compliance tolerance must be nonnegative")
    source = json.loads(args.source.read_text(encoding="utf-8"))
    cfg = load_case(args.case, quick=bool(source.get("quick", False)))
    validate_source_metadata(source, args.case, cfg)
    segments = int(cfg["rib"]["segments"])
    full_ribs, full_t, full_stage = stage_design(source, args.full_stage, segments)
    reduced_ribs, reduced_t, reduced_stage = stage_design(
        source, args.reduced_stage, segments
    )
    ribs, thicknesses, lifted_names = lifted_design(
        full_ribs, full_t, reduced_ribs, reduced_t, float(cfg["rib"]["lower"])
    )
    if not lifted_names:
        raise ValueError("the selected reduced stage contains no deleted ribs to lift")

    optimizer = RibLayoutOptimizer(build_model(cfg), cfg)
    validation_start = optimizer.analysis_count
    reanalyzed_full = optimizer.analyze(full_ribs, full_t)
    full_error = validate_saved_compliance(
        args.full_stage,
        float(full_stage["compliance"]),
        float(reanalyzed_full.compliance),
        args.source_compliance_tolerance,
    )
    reanalyzed_reduced = optimizer.analyze(reduced_ribs, reduced_t)
    reduced_error = validate_saved_compliance(
        args.reduced_stage,
        float(reduced_stage["compliance"]),
        float(reanalyzed_reduced.compliance),
        args.source_compliance_tolerance,
    )
    source_validation_analysis_count = optimizer.analysis_count-validation_start
    thicknesses = optimizer._feasible_start(ribs, thicknesses)
    initialization_start = optimizer.analysis_count
    initial_result = optimizer.analyze(ribs, thicknesses)
    lifted_initialization_analysis_count = (
        optimizer.analysis_count-initialization_start
    )
    history: list[dict] = []
    start_count = optimizer.analysis_count
    started = time.perf_counter()
    final_ribs, final_t, final_result = optimizer.optimize_geometry(
        ribs, thicknesses, initial_result,
        max_iterations_override=args.max_iterations,
        iteration_history=history,
    )
    elapsed = time.perf_counter()-started
    payload = {
        "case": args.case,
        "source": portable_artifact_path(args.source),
        "full_stage": args.full_stage,
        "reduced_stage": args.reduced_stage,
        "full_rib_count": len(full_ribs),
        "reduced_rib_count": len(reduced_ribs),
        "lifted_rib_count": len(lifted_names),
        "lifted_rib_names": lifted_names,
        "saved_full_compliance": float(full_stage["compliance"]),
        "saved_full_stage_analysis_count": full_stage.get("analyses"),
        "reanalyzed_full_compliance": float(reanalyzed_full.compliance),
        "full_compliance_relative_error": full_error,
        "reanalyzed_full_analysis_count": 1,
        "saved_reduced_compliance": float(reduced_stage["compliance"]),
        "saved_reduced_stage_analysis_count": reduced_stage.get("analyses"),
        "reanalyzed_reduced_compliance": float(reanalyzed_reduced.compliance),
        "reduced_compliance_relative_error": reduced_error,
        "reanalyzed_reduced_analysis_count": 1,
        "source_compliance_tolerance": args.source_compliance_tolerance,
        "source_validation_analysis_count": source_validation_analysis_count,
        "lifted_initial_compliance": float(initial_result.compliance),
        "lifted_initialization_analysis_count": (
            lifted_initialization_analysis_count
        ),
        "lifted_final_compliance": float(final_result.compliance),
        "relative_to_reduced": float(
            final_result.compliance/float(reanalyzed_reduced.compliance)-1.0
        ),
        "geometry_analysis_count": optimizer.analysis_count-start_count,
        "total_analysis_count": optimizer.analysis_count,
        "geometry_termination_reason": optimizer.geometry_termination_reason,
        "elapsed_seconds": elapsed,
        "final_ribs": rib_records(final_ribs, final_t),
        "iteration_history": history,
        "log": optimizer.log,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"topology_lifting_results.json").write_text(
        strict_json_dumps(payload, indent=2), encoding="utf-8"
    )
    with (args.output/"topology_lifting_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "case", "full_stage", "reduced_stage", "full_rib_count",
            "reduced_rib_count", "lifted_rib_count", "saved_full_compliance",
            "reanalyzed_full_compliance", "full_compliance_relative_error",
            "saved_reduced_compliance", "reanalyzed_reduced_compliance",
            "reduced_compliance_relative_error", "source_validation_analysis_count",
            "lifted_initial_compliance", "lifted_initialization_analysis_count",
            "lifted_final_compliance", "relative_to_reduced",
            "geometry_analysis_count", "total_analysis_count",
            "geometry_termination_reason", "elapsed_seconds",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({key: payload[key] for key in fields})
    print(
        f"Case {args.case}: lifted {len(lifted_names)} ribs, "
        f"C={final_result.compliance:.9g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
