"""Run reproducible one-factor robustness studies for the rib-layout method."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rib_layout_core import build_model, candidate_ribs, initial_ribs, load_case
from rib_layout_serialization import portable_artifact_path, strict_json_dumps
from rib_layout_algorithms.model import Rib
from rib_layout_algorithms.optimization import RibLayoutOptimizer
from rib_layout_algorithms.symmetry import mirror_axes, mirror_groups, missing_mirror_partners


STUDIES = ("mesh", "pool", "retention", "convergence", "relaxation", "starts")
STARTING_LAYOUTS = ("all_orbits", "even_orbits", "odd_orbits")


def mesh_value(text: str) -> tuple[int, int]:
    normalized = text.lower().replace(",", "x")
    try:
        nx, ny = (int(value) for value in normalized.split("x"))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("mesh must have form NXxNY") from exc
    if nx <= 0 or ny <= 0:
        raise argparse.ArgumentTypeError("mesh dimensions must be positive")
    return nx, ny


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, default=2, choices=(1, 2, 3, 4))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--studies", nargs="+", choices=STUDIES, default=list(STUDIES))
    parser.add_argument("--mesh-values", nargs="+", type=mesh_value, default=((20, 10), (40, 20), (80, 40)))
    parser.add_argument("--pool-spans", nargs="+", type=int, default=(1, 2, 3))
    parser.add_argument("--retention-thresholds", nargs="+", type=float, default=(0.50, 0.70, 0.90))
    parser.add_argument("--convergence-thresholds", nargs="+", type=float, default=(0.005, 0.010, 0.020))
    parser.add_argument("--relaxations", nargs="+", type=float, default=(0.025, 0.050, 0.075))
    parser.add_argument("--starting-layouts", nargs="+", choices=STARTING_LAYOUTS, default=STARTING_LAYOUTS)
    parser.add_argument(
        "--common-reanalysis-mesh", "--common-fine-mesh",
        dest="common_reanalysis_mesh", nargs=2, type=int,
        metavar=("NX", "NY"), default=None,
        help=(
            "Reanalyse every optimized final layout on one common mesh.  "
            "The mesh must be at least as fine componentwise as every "
            "optimization mesh; the legacy --common-fine-mesh spelling is "
            "accepted as an alias."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def starting_layout_variant(cfg: dict, name: str) -> list[Rib]:
    """Return one deterministic, mirror-orbit-closed initial layout."""
    ribs = initial_ribs(cfg)
    if name == "all_orbits":
        selected = ribs
    else:
        width, height = map(float, cfg["domain"])
        groups = mirror_groups(ribs, mirror_axes(cfg), width, height)
        parity = 0 if name == "even_orbits" else 1
        selected_indices = [index for group_index, group in enumerate(groups) if group_index % 2 == parity for index in group]
        selected = [ribs[index] for index in selected_indices]
    if not selected:
        raise ValueError(f"starting layout {name!r} is empty for this case")
    missing = missing_mirror_partners(
        selected, mirror_axes(cfg), *map(float, cfg["domain"])
    )
    if missing:
        raise RuntimeError(f"starting layout {name!r} is not mirror closed: {missing}")
    return selected


def build_study_specs(base_cfg: dict, args: argparse.Namespace) -> list[dict]:
    """Build pre-registered one-factor run specifications."""
    specs: list[dict] = []

    def add(factor: str, value, starting_layout: str = "all_orbits") -> None:
        cfg = deepcopy(base_cfg)
        # Robustness studies compare one rationalization pass.  A case-level
        # optional demonstration pass is deliberately excluded.
        cfg.pop("further_rationalization_relaxation", None)
        if factor == "mesh":
            cfg["mesh"] = list(map(int, value))
        elif factor == "pool":
            cfg["algorithm"]["candidate_max_span_cells"] = int(value)
        elif factor == "retention":
            cfg["algorithm"]["addition_factor_min_ratio"] = float(value)
        elif factor == "convergence":
            cfg["algorithm"]["addition_sizing_improvement_min"] = float(value)
            cfg["algorithm"]["active_cycle_improvement_min"] = float(value)
        elif factor == "relaxation":
            cfg["rationalization_relaxation"] = float(value)
        elif factor != "starts":
            raise ValueError(f"unknown study factor {factor}")
        specs.append({
            "factor": factor,
            "value": list(value) if factor == "mesh" else value,
            "starting_layout": starting_layout,
            "config": cfg,
        })

    selected = set(args.studies)
    if "mesh" in selected:
        for value in args.mesh_values:
            add("mesh", value)
    if "pool" in selected:
        for value in args.pool_spans:
            add("pool", value)
    if "retention" in selected:
        for value in args.retention_thresholds:
            add("retention", value)
    if "convergence" in selected:
        for value in args.convergence_thresholds:
            add("convergence", value)
    if "relaxation" in selected:
        for value in args.relaxations:
            add("relaxation", value)
    if "starts" in selected:
        for value in args.starting_layouts:
            add("starts", value, value)
    for index, spec in enumerate(specs, 1):
        spec["run_id"] = f"{index:03d}_{spec['factor']}_{str(spec['value']).replace(' ', '')}"
    return specs[:args.max_runs] if args.max_runs is not None else specs


def stage_record(stage, optimizer: RibLayoutOptimizer) -> dict:
    return {
        "name": stage.name,
        "rib_count": len(stage.ribs),
        "compliance": float(stage.compliance),
        "volume": optimizer.volume(stage.ribs, stage.thicknesses),
        "analyses": int(stage.analyses),
        "note": stage.note,
        "ribs": [
            {
                "name": rib.name,
                "p0": list(rib.p0),
                "p1": list(rib.p1),
                "height": float(rib.height),
                "thickness": float(thickness),
            }
            for rib, thickness in zip(stage.ribs, stage.thicknesses)
        ],
    }


def _aggregate_hash(paths: list[Path], relative_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        set(paths), key=lambda item: item.relative_to(relative_root).as_posix()
    ):
        relative = path.relative_to(relative_root).as_posix().encode("utf-8")
        digest.update(relative+b"\0"+hashlib.sha256(path.read_bytes()).digest()+b"\0")
    return digest.hexdigest()


def source_provenance(code_root: Path, tool_path: Path | None = None) -> dict:
    """Hash core/config/tests and every Python diagnostic-tool dependency."""
    core_paths: list[Path] = []
    for pattern in ("*.py", "*.bat", "requirements.txt"):
        core_paths.extend(path for path in code_root.glob(pattern) if path.is_file())
    for directory in (code_root/"rib_layout_algorithms", code_root/"tests"):
        core_paths.extend(path for path in directory.rglob("*.py") if path.is_file())
    for pattern in ("*.yaml", "*.yml", "*.json"):
        core_paths.extend(
            path for path in (code_root/"configs").glob(pattern) if path.is_file()
        )
    tool_paths = [
        path for path in (code_root/"tools").rglob("*.py") if path.is_file()
    ]
    all_paths = sorted(set(core_paths+tool_paths))
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=code_root,
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    try:
        status_text = subprocess.run(
            ["git", "status", "--porcelain"], cwd=code_root,
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.rstrip()
    except (OSError, subprocess.SubprocessError):
        status_text = None
    entry = tool_path or Path(__file__)
    return {
        "source_aggregate_sha256": _aggregate_hash(all_paths, code_root.parent),
        "source_file_count": len(all_paths),
        "core_source_aggregate_sha256": _aggregate_hash(
            list(set(core_paths)), code_root.parent
        ),
        "core_source_file_count": len(set(core_paths)),
        "tools_aggregate_sha256": _aggregate_hash(tool_paths, code_root.parent),
        "tool_file_count": len(tool_paths),
        "tool_files": [
            path.relative_to(code_root).as_posix() for path in sorted(tool_paths)
        ],
        "git_commit": commit,
        "git_dirty": None if status_text is None else bool(status_text),
        "git_status_porcelain": status_text,
        "entry_tool": portable_artifact_path(entry, code_root),
        "entry_tool_sha256": hashlib.sha256(entry.read_bytes()).hexdigest(),
    }


def reanalyze_on_mesh(stage: dict, cfg: dict, mesh: tuple[int, int]) -> dict:
    fine_cfg = deepcopy(cfg)
    fine_cfg["mesh"] = list(map(int, mesh))
    model = build_model(fine_cfg)
    ribs = [
        Rib(tuple(item["p0"]), tuple(item["p1"]), item["height"], item["name"], int(fine_cfg["rib"]["segments"]))
        for item in stage["ribs"]
    ]
    thicknesses = np.asarray([item["thickness"] for item in stage["ribs"]], float)
    started = time.perf_counter()
    result = model.analyze(ribs, thicknesses)
    return {
        "process_status": "complete",
        "optimization_held_fixed": True,
        "mesh": list(map(int, mesh)),
        "compliance": float(result.compliance),
        "volume": RibLayoutOptimizer.volume(ribs, thicknesses),
        "elapsed_seconds": time.perf_counter()-started,
        "analysis_count": 1,
    }


def validate_common_reanalysis_mesh(
    common_mesh: tuple[int, int] | list[int] | None,
    specs: list[dict],
) -> None:
    if common_mesh is None:
        return
    if any(value <= 0 for value in common_mesh):
        raise ValueError("common reanalysis mesh dimensions must be positive")
    maximum = [
        max(int(spec["config"]["mesh"][axis]) for spec in specs)
        for axis in range(2)
    ]
    if any(int(common_mesh[axis]) < maximum[axis] for axis in range(2)):
        raise ValueError(
            "common reanalysis mesh must be componentwise at least as fine "
            f"as every optimization mesh; required >= {maximum}"
        )


def fixed_layout_mesh_reanalysis(
    stage: dict,
    cfg: dict,
    meshes: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> dict:
    """Reanalyse one fixed final layout across a response-mesh sequence."""
    started_utc = datetime.now(timezone.utc).isoformat()
    records = [reanalyze_on_mesh(stage, cfg, tuple(mesh)) for mesh in meshes]
    return {
        "process_status": "complete",
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "optimization_held_fixed": True,
        "interpretation": (
            "Finite-element response mesh reanalysis of one fixed rib layout; "
            "no sizing, layout, filtering, or rationalization is repeated."
        ),
        "meshes": records,
    }


def flatten_record(record: dict) -> dict:
    final = record.get("final_stage", {})
    common = record.get("common_reanalysis", {}) or {}
    return {
        "run_id": record["run_id"],
        "factor": record["factor"],
        "value": strict_json_dumps(record["value"], separators=(",", ":")),
        "starting_layout": record["starting_layout"],
        "process_status": record["process_status"],
        "mesh": "x".join(map(str, record["config"]["mesh"])),
        "initial_rib_count": record.get("initial_rib_count"),
        "candidate_count": record.get("candidate_count"),
        "final_rib_count": final.get("rib_count"),
        "final_compliance": final.get("compliance"),
        "final_volume": final.get("volume"),
        "analysis_count": record.get("analysis_count"),
        "elapsed_seconds": record.get("elapsed_seconds"),
        "last_geometry_termination_reason": record.get(
            "last_geometry_termination_reason"
        ),
        "optimizer_converged": record.get("optimizer_converged"),
        "common_reanalysis_compliance": common.get("compliance"),
        "short_rib_length_threshold": record.get("short_rib_length_threshold"),
        "error": record.get("error"),
    }


def write_aggregate(output: Path, payload: dict) -> None:
    (output/"robustness_study.json").write_text(
        strict_json_dumps(payload, indent=2), encoding="utf-8"
    )
    rows = [flatten_record(record) for record in payload["runs"]]
    if rows:
        with (output/"robustness_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    aggregate_started_utc = datetime.now(timezone.utc).isoformat()
    if any(span <= 0 for span in args.pool_spans):
        raise ValueError("candidate-pool spans must be positive")
    if any(not 0.0 <= value <= 1.0 for value in args.retention_thresholds):
        raise ValueError("retention thresholds must lie in [0,1]")
    if any(value < 0.0 for value in (*args.convergence_thresholds, *args.relaxations)):
        raise ValueError("convergence thresholds and relaxations must be nonnegative")
    if args.max_runs is not None and args.max_runs <= 0:
        raise ValueError("max-runs must be positive")
    base_cfg = load_case(args.case)
    specs = build_study_specs(base_cfg, args)
    validate_common_reanalysis_mesh(args.common_reanalysis_mesh, specs)
    if "mesh" in args.studies and list(map(int, base_cfg["mesh"])) not in [
        list(map(int, mesh)) for mesh in args.mesh_values
    ]:
        raise ValueError(
            "mesh study must include the baseline optimization mesh so its "
            "final layout can be held fixed for response reanalysis"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "case": args.case,
        "started_utc": aggregate_started_utc,
        "completed_utc": None,
        "process_status": "running",
        "seed": args.seed,
        "study_protocol": "one_factor_at_a_time",
        "further_rationalization_excluded": True,
        "common_reanalysis_mesh": args.common_reanalysis_mesh,
        "mesh_study_disclosure": (
            "End-to-end mesh variants also change the mesh-dependent physical "
            "short-rib filtering threshold.  They are algorithmic mesh-dependence "
            "runs, not pure FE response convergence.  A separate fixed-layout "
            "response reanalysis is stored for the baseline final design."
        ),
        "source_provenance": source_provenance(
            Path(__file__).resolve().parents[1], Path(__file__)
        ),
        "requested_studies": list(args.studies),
        "planned_runs": specs,
        "runs": [],
    }
    if args.dry_run:
        payload["process_status"] = "planned_not_run"
        payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_aggregate(args.output, payload)
        print(f"Planned {len(specs)} runs; no analyses executed")
        return 0

    failed = False
    for spec in specs:
        cfg = deepcopy(spec["config"])
        record = {
            **spec,
            "process_status": "running",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "completed_utc": None,
            "seed": args.seed,
            "executable_provenance": payload["source_provenance"],
        }
        started = time.perf_counter()
        try:
            initial = starting_layout_variant(cfg, spec["starting_layout"])
            candidates = candidate_ribs(cfg)
            optimizer = RibLayoutOptimizer(build_model(cfg), cfg)
            short_threshold = optimizer.short_rib_length_threshold()
            run = optimizer.run(initial, candidates)
            stages = [stage_record(stage, optimizer) for stage in run.stages]
            optimizer_elapsed_seconds = time.perf_counter()-started
            record.update({
                "process_status": "complete",
                "initial_rib_count": len(initial),
                "candidate_count": len(candidates),
                "analysis_count": optimizer.analysis_count,
                "optimizer_elapsed_seconds": optimizer_elapsed_seconds,
                "last_geometry_termination_reason": (
                    optimizer.geometry_termination_reason
                ),
                "optimizer_converged": bool(
                    optimizer.geometry_termination_reason == "converged"
                ),
                "short_rib_length_threshold": short_threshold,
                "short_rib_threshold_settings": {
                    "short_rib_shell_cells": cfg["algorithm"].get(
                        "short_rib_shell_cells"
                    ),
                    "short_rib_cell_fraction": cfg["algorithm"].get(
                        "short_rib_cell_fraction"
                    ),
                    "shell_cell_size": [optimizer.model.dx, optimizer.model.dy],
                },
                "mesh_sweep_algorithmic_confound": bool(spec["factor"] == "mesh"),
                "stages": stages,
                "final_stage": stages[-1],
                "rationalization_histories": run.rationalization_histories,
                "log": optimizer.log,
            })
            if args.common_reanalysis_mesh is not None:
                record["common_reanalysis"] = reanalyze_on_mesh(
                    stages[-1], cfg, tuple(args.common_reanalysis_mesh)
                )
            # Finalize only after optional common-mesh work so these fields
            # describe the complete per-record process, not just optimization.
            record["elapsed_seconds"] = time.perf_counter()-started
            record["completed_utc"] = datetime.now(timezone.utc).isoformat()
        except Exception as exc:  # retain adverse/failed runs in the evidence set
            failed = True
            record.update({
                "process_status": "failed",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "optimizer_converged": False,
                "elapsed_seconds": time.perf_counter()-started,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })
        payload["runs"].append(record)
        (args.output/f"{spec['run_id']}.json").write_text(
            strict_json_dumps(record, indent=2), encoding="utf-8"
        )
        write_aggregate(args.output, payload)
        print(
            f"{spec['run_id']}: {record['process_status']}, "
            f"C={record.get('final_stage', {}).get('compliance')}"
        )
    if "mesh" in args.studies:
        baseline = next((
            record for record in payload["runs"]
            if record["factor"] == "mesh"
            and list(map(int, record["value"])) == list(map(int, base_cfg["mesh"]))
            and record.get("process_status") == "complete"
        ), None)
        if baseline is None:
            failed = True
            payload["fixed_layout_mesh_reanalysis"] = {
                "process_status": "failed",
                "error": "completed baseline-mesh optimization is unavailable",
            }
        else:
            try:
                fixed = fixed_layout_mesh_reanalysis(
                    baseline["final_stage"], base_cfg,
                    [tuple(mesh) for mesh in args.mesh_values],
                )
                fixed["baseline_run_id"] = baseline["run_id"]
                fixed["baseline_optimization_mesh"] = list(base_cfg["mesh"])
                payload["fixed_layout_mesh_reanalysis"] = fixed
            except Exception as exc:
                failed = True
                payload["fixed_layout_mesh_reanalysis"] = {
                    "process_status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
    payload["process_status"] = "complete_with_failures" if failed else "complete"
    payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_aggregate(args.output, payload)
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
