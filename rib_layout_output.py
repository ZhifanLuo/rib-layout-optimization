"""Unified result serialization and visualization for all examples."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

from rib_layout_env import PROJECT_ROOT
from rib_layout_serialization import strict_json_dumps
from rib_layout_algorithms.plotting import (
    example1_detailed_panels,
    example2_detailed_timeline,
    plot_active_set_history,
    plot_example1_detailed_stages,
    plot_example2_detailed_timeline,
    plot_example2_detailed_timeline_pages,
    plot_paper_style_stages,
    plot_stage_3d,
)


def stage_to_dict(stage) -> dict[str, Any]:
    return {
        "name": stage.name,
        "rib_count": len(stage.ribs),
        "compliance": float(stage.compliance),
        "analyses": int(stage.analyses),
        "volume": float(sum(
            rib.length*rib.height*thickness
            for rib, thickness in zip(stage.ribs, stage.thicknesses)
        )),
        "note": stage.note,
        "ribs": [
            {
                "name": rib.name,
                "p0": list(rib.p0),
                "p1": list(rib.p1),
                "height": rib.height,
                "thickness": float(thickness),
            }
            for rib, thickness in zip(stage.ribs, stage.thicknesses)
        ],
    }


def _comparison_to_manuscript(number: int, stages: list[dict]) -> dict:
    reference_path = PROJECT_ROOT / "configs" / "reported_results.json"
    if not reference_path.exists():
        return {}
    references = json.loads(reference_path.read_text(encoding="utf-8"))
    reference = references["examples"].get(str(number), {})
    aliases = {
        "initial_sizing": "initial",
        "adaptive": "adaptive",
        "geometry": "geometry",
        "rationalized": "rationalized",
    }
    comparison = {}
    for stage in stages:
        target = reference.get(aliases.get(stage["name"], ""))
        if target is None:
            continue
        comparison[stage["name"]] = {
            "computed_compliance": stage["compliance"],
            "reported_compliance": target["compliance"],
            "relative_difference": (
                stage["compliance"]-target["compliance"]
            )/target["compliance"],
            "computed_ribs": stage["rib_count"],
            "reported_ribs": target["ribs"],
        }
    return comparison


def _write_rationalization_history(
    output: Path,
    history: list[dict],
    *,
    suffix: str = "",
) -> None:
    if not history:
        return
    suffix_text = f"_{suffix}" if suffix else ""
    (output / f"rationalization_history{suffix_text}.json").write_text(
        strict_json_dumps(history, indent=2), encoding="utf-8"
    )
    eq18 = next(
        (event for event in history if event.get("event") == "eq18_solution"),
        None,
    )
    if eq18 is not None:
        with (output / f"rationalization_eq18_thicknesses{suffix_text}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "tref", "index", "name", "p0_x", "p0_y", "p1_x", "p1_y",
                "length", "thickness", "thickness_over_tref", "below_tref",
            ])
            for rib in eq18["ribs"]:
                writer.writerow([
                    eq18["tref"], rib["index"], rib["name"], *rib["p0"],
                    *rib["p1"], rib["length"], rib["thickness"],
                    rib["thickness_over_tref"], rib["below_tref"],
                ])


def _write_rationalization_histories(
    output: Path,
    flat_history: list[dict],
    histories: Mapping[str, list[dict]],
) -> None:
    """Write backward-compatible and unambiguous per-pass histories."""
    for pattern in (
        "rationalization_history*.json",
        "rationalization_eq18_thicknesses*.csv",
    ):
        for stale_path in output.glob(pattern):
            stale_path.unlink()
    _write_rationalization_history(output, flat_history)
    for pass_name, history in histories.items():
        _write_rationalization_history(
            output,
            history,
            suffix=pass_name,
        )


def _write_custom_detail_plots(output: Path, artifacts, profile: str) -> None:
    run, cfg = artifacts.run, artifacts.config
    if profile == "example1":
        plot_example1_detailed_stages(
            run.stages,
            run.active_history,
            cfg,
            output / "example_1_six_stages.png",
        )
        panels = example1_detailed_panels(run.stages, run.active_history)
        with (output / "six_stage_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "panel", "description", "saved_stage", "rib_count",
                "compliance", "volume",
            ])
            for index, (description, stage) in enumerate(panels, start=1):
                if stage is None:
                    writer.writerow([index, description, "domain", 0, "", ""])
                else:
                    writer.writerow([
                        index, description.replace("\n", " "), stage.name,
                        len(stage.ribs), stage.compliance,
                        artifacts.optimizer.volume(stage.ribs, stage.thicknesses),
                    ])
    elif profile == "example2":
        plot_example2_detailed_timeline(
            run.stages,
            run.active_history,
            cfg,
            output / "example_2_detailed_history.png",
        )
        plot_example2_detailed_timeline_pages(
            run.stages,
            run.active_history,
            cfg,
            output / "detailed_history_pages",
        )
        timeline = example2_detailed_timeline(run.stages, run.active_history)
        with (output / "detailed_stage_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "index", "description", "saved_stage", "rib_count",
                "compliance", "volume", "note",
            ])
            for index, (description, stage) in enumerate(timeline, start=1):
                writer.writerow([
                    index, description, stage.name, len(stage.ribs),
                    stage.compliance,
                    artifacts.optimizer.volume(stage.ribs, stage.thicknesses),
                    stage.note,
                ])


def _remove_inactive_detail_outputs(output: Path, active_profile: str) -> None:
    """Remove stale case-specific artifacts not selected for the current run."""
    profile_artifacts = {
        "example1": (
            # Remove both current and legacy names when this profile is inactive.
            "example_1_five_stages.png",
            "five_stage_summary.csv",
            "example_1_six_stages.png",
            "six_stage_summary.csv",
        ),
        "example2": (
            "example_2_detailed_history.png",
            "detailed_stage_summary.csv",
            "detailed_history_pages",
            "detailed_stages",
        ),
    }
    for profile, names in profile_artifacts.items():
        if profile == active_profile:
            continue
        for name in names:
            path = output/name
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()


def save_optimization_results(
    artifacts,
    output_root: Path,
    output_options: Mapping[str, Any],
    *,
    quick: bool,
) -> Path:
    """Save common and example-specific outputs, returning the case folder."""
    cfg, optimizer, run = artifacts.config, artifacts.optimizer, artifacts.run
    number = int(cfg["number"])
    directory_name = str(output_options.get("directory_name", f"example_{number}"))
    output = Path(output_root) / directory_name
    output.mkdir(parents=True, exist_ok=True)

    stages = [stage_to_dict(stage) for stage in run.stages]
    active_history = [stage_to_dict(stage) for stage in run.active_history]
    rationalization_histories = {
        pass_name: list(history)
        for pass_name, history in run.rationalization_histories.items()
    }
    payload = {
        "example": number,
        "name": cfg["name"],
        "title": output_options.get("title", f"Example {number}"),
        "description": output_options.get("description", ""),
        "mesh_used": cfg["mesh"],
        "quick": bool(quick),
        "elapsed_seconds": artifacts.elapsed_seconds,
        "total_analyses": optimizer.analysis_count,
        "numerical_settings": {
            "linear_solver": cfg["linear_solver"],
            "linear_solver_threads": cfg["linear_solver_threads"],
            "sensitivity_workers": cfg["algorithm"]["sensitivity_workers"],
        },
        "stages": stages,
        "active_set_iterations": active_history,
        "rationalization_history": optimizer.rationalization_history,
        "rationalization_histories": rationalization_histories,
        "log": run.log,
    }
    # Save numerical results before any optional presentation work.
    (output / "checkpoint_results.json").write_text(
        strict_json_dumps(payload, indent=2), encoding="utf-8"
    )

    if output_options.get("write_stage_images", True):
        for stage in run.stages:
            plot_stage_3d(stage, cfg, output / f"{stage.name}.png")
        plot_paper_style_stages(
            run.stages,
            run.active_history,
            cfg,
            output / "all_stages.png",
        )
        plot_active_set_history(
            run.active_history, cfg, output / "active_set_iterations.png"
        )
    profile = str(output_options.get("detail_profile", ""))
    _remove_inactive_detail_outputs(output, profile)
    if profile:
        _write_custom_detail_plots(output, artifacts, profile)

    payload["comparison_to_manuscript"] = _comparison_to_manuscript(
        number, stages
    )
    (output / "results.json").write_text(
        strict_json_dumps(payload, indent=2), encoding="utf-8"
    )
    with (output / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "rib_count", "compliance", "volume", "analyses"])
        for stage in stages:
            writer.writerow([
                stage["name"], stage["rib_count"], stage["compliance"],
                stage["volume"], stage["analyses"],
            ])
    with (output / "case_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "example", "description", "stage", "rib_count", "compliance",
            "volume", "analyses",
        ])
        for stage in stages:
            writer.writerow([
                output_options.get("roman", number),
                output_options.get("description", ""),
                stage["name"], stage["rib_count"], stage["compliance"],
                stage["volume"], stage["analyses"],
            ])

    _write_rationalization_histories(
        output,
        optimizer.rationalization_history,
        rationalization_histories,
    )
    return output.resolve()


__all__ = ["save_optimization_results", "stage_to_dict"]
