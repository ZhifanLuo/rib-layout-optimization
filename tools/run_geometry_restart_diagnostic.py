"""Run Eq. (7) geometry optimization from a saved optimization stage."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rib_layout_core import build_model, load_case
from rib_layout_algorithms.model import Rib
from rib_layout_algorithms.optimization import RibLayoutOptimizer


def column_records(ribs: list[Rib], thicknesses: np.ndarray) -> list[dict]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, required=True, choices=(1, 2, 3, 4))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("adaptive", "geometry"), default="geometry",
        help="Saved stage used as the initial design (default: geometry).",
    )
    args = parser.parse_args()

    source = json.loads(args.source.read_text(encoding="utf-8"))
    cfg = load_case(args.case, quick=bool(source.get("quick", False)))
    stage = next(item for item in source["stages"] if item["name"] == args.stage)
    segments = int(cfg["rib"]["segments"])
    ribs = [
        Rib(
            tuple(item["p0"]), tuple(item["p1"]), float(item["height"]),
            item["name"], segments,
        )
        for item in stage["ribs"]
    ]
    thicknesses = np.asarray([item["thickness"] for item in stage["ribs"]], float)
    initial_coordinates = np.asarray([[*rib.p0, *rib.p1] for rib in ribs], float)
    initial_t = thicknesses.copy()

    optimizer = RibLayoutOptimizer(build_model(cfg), cfg)
    restart_result = optimizer.analyze(ribs, thicknesses)
    iteration_history: list[dict] = []
    start_count = optimizer.analysis_count
    final_ribs, final_t, final_result = optimizer.optimize_geometry(
        ribs,
        thicknesses,
        restart_result,
        iteration_history=iteration_history,
    )
    final_coordinates = np.asarray(
        [[*rib.p0, *rib.p1] for rib in final_ribs], float
    )

    args.output.mkdir(parents=True, exist_ok=True)
    rib_names = [rib.name for rib in ribs]
    fixed_headers = [
        "outer", "objective", "compliance", "volume", "volume_ratio",
        "constraint_violation", "feasible", "objective_relative_change",
        "objective_relative_change_signed", "design_change",
        "thickness_design_change", "coordinate_design_change",
        "maximum_absolute_thickness_change",
        "maximum_absolute_coordinate_change",
        "predicted_objective_change", "true_objective_change",
        "approximation_ratio", "inner_iterations", "move_global_used",
        "move_global_next", "frozen_geometry_count", "frozen_geometry_names",
        "frozen_geometry_reasons", "step_converged", "consecutive_converged",
    ]
    history_name = f"geometry_from_{args.stage}_iteration_history.csv"
    with (args.output / history_name).open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([*fixed_headers, *[f"t_{name}" for name in rib_names]])
        initial_volume = optimizer.volume(ribs, initial_t)
        initial_values = {
            "outer": 0,
            "objective": float(restart_result.compliance),
            "compliance": float(restart_result.compliance),
            "volume": float(initial_volume),
            "volume_ratio": float(initial_volume/optimizer.volume_bound),
            "constraint_violation": float(
                max(initial_volume/optimizer.volume_bound-1.0, 0.0)
            ),
            "feasible": bool(
                initial_volume <= optimizer.volume_bound
                * (1.0+cfg["algorithm"]["sca_constraint_tolerance"])
            ),
        }
        writer.writerow(
            [initial_values.get(header) for header in fixed_headers]
            + [float(value) for value in initial_t]
        )
        for record in iteration_history:
            thickness_by_name = dict(zip(record["rib_names"], record["thicknesses"]))
            fixed_values = []
            for header in fixed_headers:
                value = record.get(header)
                fixed_values.append(
                    json.dumps(value, separators=(",", ":"))
                    if isinstance(value, (list, dict)) else value
                )
            writer.writerow(
                fixed_values
                + [thickness_by_name[name] for name in rib_names]
            )

    payload = {
        "case": args.case,
        "source": str(args.source.resolve()),
        "source_stage": args.stage,
        "mesh": cfg["mesh"],
        "rib_count": len(ribs),
        "saved_source_compliance": float(stage["compliance"]),
        "restart_compliance": float(restart_result.compliance),
        "restart_initialization_fea": 1,
        "geometry_outer_fea": optimizer.analysis_count-start_count,
        "outer_iterations": len(iteration_history),
        "initial_volume": optimizer.volume(ribs, initial_t),
        "final_volume": optimizer.volume(final_ribs, final_t),
        "final_compliance": float(final_result.compliance),
        "relative_compliance_change": float(
            final_result.compliance/restart_result.compliance-1.0
        ),
        "maximum_endpoint_coordinate_change": float(
            np.max(np.abs(final_coordinates-initial_coordinates))
        ),
        "maximum_thickness_change": float(np.max(np.abs(final_t-initial_t))),
        "initial_ribs": column_records(ribs, initial_t),
        "final_ribs": column_records(final_ribs, final_t),
        "iteration_history": iteration_history,
        "log": optimizer.log,
    }
    results_name = f"geometry_from_{args.stage}_results.json"
    (args.output / results_name).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (args.output / f"geometry_from_{args.stage}_log.txt").write_text(
        "\n".join(optimizer.log), encoding="utf-8"
    )
    print(
        f"Case {args.case}: C0={restart_result.compliance:.9g}, "
        f"Cfinal={final_result.compliance:.9g}, "
        f"change={100*payload['relative_compliance_change']:.4f}%, "
        f"outer={len(iteration_history)}, "
        f"FEA={optimizer.analysis_count-start_count}"
    )


if __name__ == "__main__":
    main()
