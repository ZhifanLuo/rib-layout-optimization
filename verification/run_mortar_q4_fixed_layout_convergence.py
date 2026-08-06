"""Run fixed-layout convergence with Q4 mortar/L2 rib-root coupling."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time


def parse_meshes(value: str) -> list[tuple[int, int]]:
    result = []
    for item in value.split(","):
        nx, ny = (int(part) for part in item.strip().lower().split("x"))
        if nx < 1 or ny < 1:
            raise ValueError("mesh dimensions must be positive")
        result.append((nx, ny))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--meshes", default="20x10,40x20,80x40,160x80",
    )
    parser.add_argument("--root-segments", type=int, default=40)
    parser.add_argument("--quadrature-order", type=int, default=4)
    parser.add_argument("--root-scan", default="20,40,80")
    args = parser.parse_args()
    started_utc = datetime.now(timezone.utc).isoformat()
    root = args.project_root.resolve()
    code_root = root/"Code"
    sys.path.insert(0, str(code_root))
    from rib_layout_serialization import portable_artifact_path, strict_json_dumps
    from rib_layout_algorithms.model import Rib
    from rib_layout_algorithms.model_shell_mortar import (
        MortarQ4ShellStiffenedPlateModel,
    )
    from rib_layout_core import load_case
    from tools.run_robustness_study import source_provenance

    source = args.source.resolve()
    record = json.loads(source.read_text(encoding="utf-8"))
    stage = record["final_stage"]
    cfg = load_case(2, quick=False)
    loads = [{
        "weight": 1.0,
        "forces": [{
            "point": [20.0, 0.0],
            "value": [0.0, -100.0, 0.0],
        }],
    }]
    supports = {
        "type": "points",
        "points": [[0.0, 0.0], [40.0, 0.0]],
    }
    ribs = [
        Rib(
            tuple(item["p0"]), tuple(item["p1"]), float(item["height"]),
            item["name"], int(cfg["rib"]["segments"]),
        )
        for item in stage["ribs"]
    ]
    thicknesses = [float(item["thickness"]) for item in stage["ribs"]]
    common = dict(
        wall_thickness=cfg["wall_thickness"], E=cfg["material"]["E"],
        nu=cfg["material"]["nu"], loads=loads, supports=supports,
        interface_subdivisions_per_cell=cfg.get(
            "interface_subdivisions_per_cell", 2,
        ), linear_solver=cfg.get("linear_solver", "auto"),
        linear_solver_threads=1, sensitivity_workers=1,
        mortar_quadrature_order=args.quadrature_order,
    )
    meshes = parse_meshes(args.meshes)

    def run_series(root_segments: int, selected_meshes: list[tuple[int, int]]):
        rows = []
        for nx, ny in selected_meshes:
            started = time.perf_counter()
            model = MortarQ4ShellStiffenedPlateModel(
                40.0, 20.0, nx, ny, root_segments=root_segments, **common,
            )
            response = model.analyze(ribs, thicknesses)
            rows.append({
                "root_segments": root_segments, "nx": nx, "ny": ny,
                "nnode": model.nnode, "ndof": model.ndof,
                "root_points": sum(
                    len(model.rib_bottom_points(rib)) for rib in ribs
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

    convergence = run_series(args.root_segments, meshes)
    scan_mesh = parse_meshes("80x40")
    root_scan = [
        run_series(int(value.strip()), scan_mesh)[0]
        for value in args.root_scan.split(",")
    ]
    finest_change = convergence[-1]["relative_change_from_previous"]
    payload = {
        "schema_version": 1,
        "process_status": "complete",
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "study_type": "fixed_layout_response_only",
        "model": "q4_mortar_l2_rib_root_coupling",
        "case": 2,
        "baseline_run_id": "002_mesh_[40,20]",
        "source": portable_artifact_path(source, code_root),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "boundary_semantics": "committed_point_load_and_point_support",
        "root_segments": args.root_segments,
        "mortar_quadrature_order": args.quadrature_order,
        "ribbed_shell": convergence,
        "root_segment_scan_at_80x40": root_scan,
        "strict_convergence_tolerance": 0.01,
        "engineering_convergence_tolerance": 0.05,
        "strict_convergence_achieved": finest_change <= 0.01,
        "engineering_convergence_achieved": finest_change <= 0.05,
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
