#!/usr/bin/env python
"""Generate manuscript-comparable 3-D stage figures from saved results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rib_layout_core import load_case
from rib_layout_env import PROJECT_ROOT as ROOT
from rib_layout_serialization import portable_artifact_path, strict_json_dumps
from rib_layout_algorithms.model import Rib
from rib_layout_algorithms.optimization import Stage
from rib_layout_algorithms.plotting import (
    example1_detailed_panels,
    make_paper_comparison,
    plot_active_set_history,
    plot_example1_detailed_stages,
    plot_paper_style_stages,
    plot_stage_3d,
)


PAPER_ROOT = ROOT.parent / "Latex" / "Rib_layout_figures"


def _decode_stages(items: list[dict]) -> list[Stage]:
    stages: list[Stage] = []
    for item in items:
        ribs = [Rib(tuple(r["p0"]), tuple(r["p1"]), r["height"], r["name"]) for r in item["ribs"]]
        thicknesses = np.array([r["thickness"] for r in item["ribs"]], float)
        stages.append(Stage(item["name"], ribs, thicknesses, item["compliance"], item["analyses"], item.get("note", "")))
    return stages


def load_stages(results_path: Path) -> tuple[list[Stage], list[Stage]]:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    return _decode_stages(payload["stages"]), _decode_stages(payload.get("active_set_iterations", []))


def generate(number: int, results_root: Path, output_root: Path) -> None:
    cfg = load_case(number, quick=number in (3, 4))
    stages, active_history = load_stages(results_root / f"example_{number}" / "results.json")
    destination = output_root / f"example_{number}"
    destination.mkdir(parents=True, exist_ok=True)
    parameters = {
        "example": number,
        "ground_plane_mm": cfg["domain"],
        "rib_height_mm": cfg["rib"]["height"],
        "initial_cell_size_mm": cfg["initial_rib_cell_size"],
        "initial_rib_length_mm": cfg["initial_rib_cell_size"] * np.sqrt(2.0),
        "initial_rib_count": len(stages[0].ribs),
        "source_results": portable_artifact_path(
            results_root / f"example_{number}" / "results.json"
        ),
    }
    (destination / "corrected_parameters.json").write_text(
        strict_json_dumps(parameters, indent=2), encoding="utf-8"
    )
    plot_stage_3d(None, cfg, destination / "00_domain_loads_supports_3d.png")
    for index, stage in enumerate(stages, start=1):
        plot_stage_3d(stage, cfg, destination / f"{index:02d}_{stage.name}_3d.png")
    composite = destination / f"example_{number}_all_stages_3d.png"
    plot_paper_style_stages(stages, active_history, cfg, composite)
    plot_active_set_history(active_history, cfg, destination / f"example_{number}_active_set_iterations_3d.png")
    if number == 1:
        detailed = example1_detailed_panels(stages, active_history)
        filenames = [
            "workflow_a_domain_3d.png",
            "workflow_b_initial_generation_sizing_3d.png",
            "workflow_c_thin_rib_removal_resizing_3d.png",
            "workflow_d_new_rib_added_3d.png",
            "workflow_e_geometry_optimization_3d.png",
        ]
        for (label, stage), filename in zip(detailed, filenames):
            if stage is None:
                title = label
            else:
                title = f"{label}\nN={len(stage.ribs)}, C={stage.compliance:.6g}"
            plot_stage_3d(stage, cfg, destination / filename, title=title)
        plot_example1_detailed_stages(
            stages,
            active_history,
            cfg,
            destination / "example_1_five_stage_workflow_3d.png",
        )
    paper = PAPER_ROOT / f"fig.{number + 2}.png"
    if paper.exists():
        make_paper_comparison(paper, composite, destination / f"example_{number}_paper_vs_python.png", number)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["1", "2", "3", "4", "all"], default="all")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "stage_figures")
    args = parser.parse_args()
    numbers = range(1, 5) if args.case == "all" else [int(args.case)]
    for number in numbers:
        generate(number, args.results, args.output)
        print(f"Generated 3-D stage and comparison figures for Example {number}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
