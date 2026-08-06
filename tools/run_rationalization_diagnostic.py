"""Resume rationalization from any saved stage and export outer histories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rib_layout_core import build_model, load_case
from rib_layout_serialization import portable_artifact_path, strict_json_dumps
from rib_layout_algorithms.model import Rib
from rib_layout_algorithms.optimization import RibLayoutOptimizer


def saved_design(stage: dict, segments: int) -> tuple[list[Rib], np.ndarray]:
    ribs = [
        Rib(
            tuple(record["p0"]), tuple(record["p1"]),
            float(record["height"]), record["name"], segments,
        )
        for record in stage["ribs"]
    ]
    thicknesses = np.asarray(
        [float(record["thickness"]) for record in stage["ribs"]], float
    )
    return ribs, thicknesses


def write_iteration_csv(
    path: Path,
    events: list[dict],
    all_rib_names: list[str],
) -> None:
    fixed = [
        "event", "attempt", "outer", "beta", "objective", "compliance",
        "volume", "objective_relative_change", "design_change",
        "compliance_feasible", "volume_feasible",
        "predicted_objective_change", "true_objective_change",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([*fixed, *[f"t_{name}" for name in all_rib_names]])
        for event in events:
            thickness_by_name = dict(zip(event["rib_names"], event["thicknesses"]))
            writer.writerow([
                event.get(column) for column in fixed
            ] + [thickness_by_name.get(name) for name in all_rib_names])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, required=True, choices=(1, 2, 3, 4))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relaxation", type=float, required=True)
    parser.add_argument(
        "--stage", default="geometry",
        help="Saved stage used as the rationalization input (default: geometry).",
    )
    args = parser.parse_args()

    source = json.loads(args.source.read_text(encoding="utf-8"))
    cfg = load_case(args.case, quick=bool(source.get("quick", False)))
    source_stage = next(
        stage for stage in source["stages"] if stage["name"] == args.stage
    )
    segments = int(cfg["rib"]["segments"])
    source_ribs, source_t = saved_design(source_stage, segments)
    optimizer = RibLayoutOptimizer(build_model(cfg), cfg)

    # One restart FEA reconstructs the displacement field at the saved design.
    source_result = optimizer.analyze(source_ribs, source_t)
    restart_compliance = float(source_result.compliance)
    rationalization_start = optimizer.analysis_count
    final_ribs, final_t, final_result = optimizer.rationalize(
        source_ribs, source_t, source_result, args.relaxation
    )

    args.output.mkdir(parents=True, exist_ok=True)
    history = optimizer.rationalization_history
    eq18_iterations = [
        event for event in history if event.get("event") == "eq18_outer_iteration"
    ]
    eq7_iterations = [
        event for event in history
        if event.get("event") == "post_filter_geometry_iteration"
    ]
    all_names = [rib.name for rib in source_ribs]
    write_iteration_csv(args.output / "eq18_iteration_history.csv", eq18_iterations, all_names)
    write_iteration_csv(args.output / "eq7_iteration_history.csv", eq7_iterations, all_names)

    payload = {
        "case": args.case,
        "source": portable_artifact_path(args.source),
        "source_stage": args.stage,
        "relaxation": float(args.relaxation),
        "mesh": cfg["mesh"],
        "restart_fea": 1,
        "restart_compliance": restart_compliance,
        "saved_source_compliance": float(source_stage["compliance"]),
        "initial_rib_count": len(source_ribs),
        "initial_volume": optimizer.volume(source_ribs, source_t),
        "compliance_limit": float((1.0+args.relaxation)*restart_compliance),
        "rationalization_fea": optimizer.analysis_count-rationalization_start,
        "final_rib_count": len(final_ribs),
        "final_compliance": float(final_result.compliance),
        "final_relative_compliance_change": float(
            final_result.compliance/restart_compliance-1.0
        ),
        "final_volume": optimizer.volume(final_ribs, final_t),
        "final_ribs": [
            {
                "name": rib.name,
                "p0": list(rib.p0), "p1": list(rib.p1),
                "height": rib.height, "thickness": float(thickness),
            }
            for rib, thickness in zip(final_ribs, final_t)
        ],
        "rationalization_history": history,
        "log": optimizer.log,
    }
    (args.output / "diagnostic_results.json").write_text(
        strict_json_dumps(payload, indent=2), encoding="utf-8"
    )
    (args.output / "diagnostic_log.txt").write_text(
        "\n".join(optimizer.log), encoding="utf-8"
    )
    print(
        f"Case {args.case}, stage={args.stage}: restart C={restart_compliance:.9g}, "
        f"Eq.18 steps={len(eq18_iterations)}, Eq.7 steps={len(eq7_iterations)}, "
        f"final ribs={len(final_ribs)}, C={final_result.compliance:.9g}, "
        f"rationalization FEA={optimizer.analysis_count-rationalization_start}"
    )


if __name__ == "__main__":
    main()
