"""Size every member in the finite candidate pool without active-set screening."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rib_layout_core import build_model, candidate_ribs, load_case
from rib_layout_algorithms.optimization import RibLayoutOptimizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", type=int, nargs="+", required=True,
        choices=(1, 2, 3, 4),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=None)
    return parser


def expanded_system_size(model, ribs) -> int:
    """Return the exact DOF count used by the expanded shell analysis."""
    if not hasattr(model, "rib_bottom_points"):
        return int(model.ndof)
    return int(model.ndof + sum(
        6*len(model.rib_bottom_points(rib)) for rib in ribs
    ))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for case_number in args.case:
        cfg = load_case(case_number, quick=args.quick)
        ribs = candidate_ribs(cfg)
        model = build_model(cfg)
        optimizer = RibLayoutOptimizer(model, cfg)
        system_dofs = expanded_system_size(model, ribs)
        started = time.perf_counter()
        thicknesses, result = optimizer.size(
            ribs, maxiter=args.max_iterations
        )
        elapsed = time.perf_counter()-started
        record = {
            "case": case_number,
            "quick": bool(args.quick),
            "mesh_nx": int(cfg["mesh"][0]),
            "mesh_ny": int(cfg["mesh"][1]),
            "candidate_count": len(ribs),
            "ground_system_dofs": int(model.ndof),
            "expanded_system_dofs": system_dofs,
            "compliance": float(result.compliance),
            "volume": optimizer.volume(ribs, thicknesses),
            "volume_bound": float(cfg["volume_bound"]),
            "analysis_count": optimizer.analysis_count,
            "elapsed_seconds": elapsed,
            "best_feasible_outer": int(
                optimizer.sizing_history[-1]["best_feasible_outer"]
                if optimizer.sizing_history else 0
            ),
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
            "sizing_history": optimizer.sizing_history,
            "log": optimizer.log,
        }
        records.append(record)
        (args.output/f"full_pool_case_{case_number}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        print(
            f"Case {case_number}: candidates={len(ribs)}, dofs={system_dofs}, "
            f"C={result.compliance:.9g}, FEA={optimizer.analysis_count}"
        )

    fields = [
        "case", "quick", "mesh_nx", "mesh_ny", "candidate_count",
        "ground_system_dofs", "expanded_system_dofs", "compliance", "volume",
        "volume_bound", "analysis_count", "elapsed_seconds", "best_feasible_outer",
    ]
    with (args.output/"full_pool_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: record[key] for key in fields} for record in records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
