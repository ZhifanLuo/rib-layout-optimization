#!/usr/bin/env python
"""Combined command-line runner using the same output schema for every case."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path

from rib_layout_core import make_case_config, run_optimization
from rib_layout_env import DEFAULT_OUTPUT_ROOT
from rib_layout_output import save_optimization_results
from rib_layout_serialization import strict_json_dumps


def rationalization_paths(number: int, cfg: dict) -> tuple[tuple[float, str], ...]:
    """Return the ordered rationalization passes configured for an example."""
    if int(number) not in range(1, 5):
        raise ValueError(f"unknown example {number}")
    paths = [(float(cfg["rationalization_relaxation"]), "5pct")]
    if "further_rationalization_relaxation" in cfg:
        paths.append((
            float(cfg["further_rationalization_relaxation"]),
            "further_5pct",
        ))
    return tuple(paths)


def run_example(
    number: int,
    quick: bool,
    output_root: Path,
    geometry_sweeps: int | None,
) -> dict:
    """Run one example through the shared core and Example-IV output schema."""
    module = importlib.import_module(f"example{int(number)}")
    config = make_case_config(
        module.CASE_CONFIG,
        quick=quick,
        geometry_sweeps=geometry_sweeps,
    )
    artifacts = run_optimization(config)
    output = save_optimization_results(
        artifacts,
        Path(output_root),
        module.OUTPUT_OPTIONS,
        quick=quick,
    )
    payload = json.loads((output/"results.json").read_text(encoding="utf-8"))
    print(
        f"Example {number}: mesh={config['mesh']}, "
        f"{artifacts.optimizer.analysis_count} analyses, "
        f"{artifacts.elapsed_seconds:.2f} s"
    )
    for stage in artifacts.run.stages:
        volume = artifacts.optimizer.volume(stage.ribs, stage.thicknesses)
        print(
            f"  {stage.name:15s} ribs={len(stage.ribs):3d} "
            f"C={stage.compliance:.7g} V={volume:.6g}"
        )
    return payload


def write_aggregate_summaries(output_root: Path) -> None:
    """Write schema-neutral summaries for every completed example."""
    payloads = []
    for number in range(1, 5):
        path = Path(output_root)/f"example_{number}"/"results.json"
        if path.exists():
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
    payloads.sort(key=lambda payload: int(payload["example"]))
    if not payloads:
        return
    (Path(output_root)/"all_results.json").write_text(
        strict_json_dumps(payloads, indent=2), encoding="utf-8"
    )
    with (Path(output_root)/"all_stage_statistics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "example", "stage", "rib_count", "compliance", "volume",
            "analyses", "elapsed_seconds_total",
        ])
        for payload in payloads:
            for stage in payload["stages"]:
                writer.writerow([
                    payload["example"], stage["name"], stage["rib_count"],
                    stage["compliance"], stage["volume"], stage["analyses"],
                    payload["elapsed_seconds"],
                ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run rib-layout optimization examples with unified output"
    )
    parser.add_argument("--case", choices=["1", "2", "3", "4", "all"], default="1")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--geometry-sweeps", type=int, default=None)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    numbers = range(1, 5) if args.case == "all" else (int(args.case),)
    for number in numbers:
        run_example(number, args.quick, args.output, args.geometry_sweeps)
    write_aggregate_summaries(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
