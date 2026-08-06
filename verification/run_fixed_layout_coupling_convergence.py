"""Verify frozen-layout convergence with the embedded line-beam shell model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_meshes(value: str) -> list[tuple[int, int]]:
    meshes = []
    for item in value.split(","):
        nx, ny = (int(part) for part in item.strip().lower().split("x"))
        if nx < 1 or ny < 1:
            raise ValueError("mesh dimensions must be positive")
        meshes.append((nx, ny))
    return meshes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--meshes", default="20x10,40x20,80x40,160x80,320x160,640x320",
    )
    parser.add_argument("--interface-mesh", default="160x80")
    parser.add_argument("--interface-levels", default="2,4,8,16")
    parser.add_argument(
        "--include-no-rib", action="store_true",
        help="also run the expensive no-rib comparison series",
    )
    args = parser.parse_args()
    started_utc = datetime.now(timezone.utc).isoformat()

    root = args.project_root.resolve()
    code_root = root/"Code"
    sys.path.insert(0, str(code_root))
    from rib_layout_serialization import portable_artifact_path, strict_json_dumps
    from rib_layout_algorithms.model import Rib
    from rib_layout_algorithms.model_shell_beam import (
        EmbeddedLineBeamShellReferenceModel,
    )
    from rib_layout_core import load_case
    from tools.run_robustness_study import source_provenance

    source = args.source.resolve()
    aggregate = json.loads(source.read_text(encoding="utf-8"))
    if "final_stage" in aggregate:
        baseline = aggregate
        stage = aggregate["final_stage"]
        baseline_run_id = "002_mesh_[40,20]"
    else:
        baseline = next(
            record for record in aggregate["runs"]
            if record["run_id"] == "002_mesh_[40,20]"
        )
        stage = baseline["final_stage"]
        baseline_run_id = baseline["run_id"]
    if stage["rib_count"] != 5 or len(stage["ribs"]) != 5:
        raise RuntimeError("expected the frozen Case-II five-rib baseline")
    cfg = load_case(2, quick=False)
    ribs = [
        Rib(
            tuple(item["p0"]), tuple(item["p1"]), float(item["height"]),
            item["name"], int(cfg["rib"]["segments"]),
        )
        for item in stage["ribs"]
    ]
    # The FE thickness is the design variable; rib height is not a thickness.
    thicknesses = [float(item["thickness"]) for item in stage["ribs"]]
    meshes = parse_meshes(args.meshes)
    common = dict(
        wall_thickness=cfg["wall_thickness"], E=cfg["material"]["E"],
        nu=cfg["material"]["nu"], loads=cfg["load_cases"],
        supports=cfg["supports"], linear_solver=cfg.get("linear_solver", "auto"),
        linear_solver_threads=1, sensitivity_workers=1,
    )

    def run_series(with_ribs: bool) -> list[dict]:
        rows = []
        for nx, ny in meshes:
            started = time.perf_counter()
            model = EmbeddedLineBeamShellReferenceModel(
                40.0, 20.0, nx, ny,
                interface_subdivisions_per_cell=cfg[
                    "interface_subdivisions_per_cell"
                ], **common,
            )
            response = model.analyze(
                ribs if with_ribs else [], thicknesses if with_ribs else []
            )
            rows.append({
                "nx": nx, "ny": ny, "ndof": model.ndof,
                "root_integration_points": (
                    sum(len(model.rib_bottom_points(rib)) for rib in ribs)
                    if with_ribs else 0
                ),
                "compliance": response.compliance,
                "load_compliances": response.load_compliances,
                "elapsed_seconds": time.perf_counter()-started,
            })
        for index, row in enumerate(rows):
            row["relative_change_from_previous"] = (
                None if index == 0 else row["compliance"] /
                rows[index-1]["compliance"]-1.0
            )
        return rows

    ribbed = run_series(True)
    no_rib = run_series(False) if args.include_no_rib else []

    interface_mesh = parse_meshes(args.interface_mesh)
    if len(interface_mesh) != 1:
        raise ValueError("interface-mesh must contain one mesh")
    interface_nx, interface_ny = interface_mesh[0]
    interface_levels = [
        int(value.strip()) for value in args.interface_levels.split(",")
    ]
    interface_scan = []
    for level in interface_levels:
        started = time.perf_counter()
        model = EmbeddedLineBeamShellReferenceModel(
            40.0, 20.0, interface_nx, interface_ny,
            interface_subdivisions_per_cell=level, **common,
        )
        response = model.analyze(ribs, thicknesses)
        interface_scan.append({
            "nx": interface_nx, "ny": interface_ny,
            "interface_subdivisions_per_cell": level,
            "root_integration_points": sum(
                len(model.rib_bottom_points(rib)) for rib in ribs
            ),
            "compliance": response.compliance,
            "elapsed_seconds": time.perf_counter()-started,
        })

    finest_change = ribbed[-1]["relative_change_from_previous"]
    payload = {
        "schema_version": 1,
        "process_status": "complete",
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "study_type": "fixed_layout_response_only",
        "coupling": "embedded_consistent_line_beam",
        "case": 2,
        "baseline_run_id": baseline_run_id,
        "source": portable_artifact_path(source, code_root),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "baseline_layout_canonical_sha256": canonical_hash(stage["ribs"]),
        "current_config_canonical_sha256": canonical_hash(cfg),
        "thickness_source": "final_stage.ribs[*].thickness",
        "ribbed_shell": ribbed,
        "no_rib_shell": no_rib,
        "no_rib_included": args.include_no_rib,
        "interface_scan": interface_scan,
        "strict_convergence_tolerance": 0.01,
        "engineering_convergence_tolerance": 0.05,
        "strict_convergence_achieved": finest_change <= 0.01,
        "engineering_convergence_achieved": finest_change <= 0.05,
        "convergence_interpretation": (
            "The selected coupling is accepted for a five-percent engineering "
            "response tolerance; a strict one-percent claim remains unsupported."
        ),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "executable_provenance": source_provenance(
            code_root, Path(__file__).resolve()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(strict_json_dumps(payload, indent=2), encoding="utf-8")
    print(strict_json_dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
