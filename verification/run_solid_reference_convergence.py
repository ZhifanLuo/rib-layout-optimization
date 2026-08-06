"""Run a response-only convergence study with an independent 3-D solid model."""

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
    parser.add_argument("--meshes", default="20x10,40x20,80x40")
    parser.add_argument("--footprint-samples", default="8")
    parser.add_argument(
        "--boundary-semantics", choices=("point", "patch"), default="point",
        help="use historical point actions or the current fixed physical patches",
    )
    args = parser.parse_args()
    started_utc = datetime.now(timezone.utc).isoformat()

    root = args.project_root.resolve()
    code_root = root / "Code"
    sys.path.insert(0, str(code_root))
    from rib_layout_serialization import portable_artifact_path, strict_json_dumps
    from rib_layout_algorithms.model import Rib
    from rib_layout_algorithms.solid_reference import StructuredQ1SolidReferenceModel
    from rib_layout_core import load_case
    from tools.run_robustness_study import source_provenance

    source = args.source.resolve()
    aggregate = json.loads(source.read_text(encoding="utf-8"))
    if "final_stage" in aggregate:
        stage = aggregate["final_stage"]
        baseline_run_id = "002_mesh_[40,20]"
    else:
        baseline = next(record for record in aggregate["runs"]
                        if record["run_id"] == "002_mesh_[40,20]")
        stage = baseline["final_stage"]
        baseline_run_id = baseline["run_id"]
    if stage["rib_count"] != 5 or len(stage["ribs"]) != 5:
        raise RuntimeError("expected the frozen Case-II five-rib baseline")
    cfg = load_case(2, quick=False)
    ribs = [
        Rib(tuple(item["p0"]), tuple(item["p1"]), float(item["height"]),
            item["name"], int(cfg["rib"]["segments"]))
        for item in stage["ribs"]
    ]
    thicknesses = [float(item["thickness"]) for item in stage["ribs"]]
    meshes = parse_meshes(args.meshes)
    sample_counts = [int(value.strip()) for value in args.footprint_samples.split(",")]
    if any(value < 2 for value in sample_counts):
        raise ValueError("footprint sample counts must be at least two")
    # The manuscript's original Case-II sequence used a point load and two
    # point supports.  The current working tree also contains later patch
    # experiments, so freeze the historical semantics explicitly unless the
    # caller asks for the fixed physical patch diagnostic.
    point_load_cases = [{
        "weight": float(case.get("weight", 1.0)),
        "forces": [{
            "point": list(force["point"]),
            "value": list(force["value"]),
        } for force in case["forces"]],
    } for case in cfg["load_cases"]]
    point_supports = {
        "type": "points",
        "points": [[0.0, 0.0], [40.0, 0.0]],
        "components": [0, 1, 2],
    }
    if args.boundary_semantics == "patch":
        boundary_loads = cfg["load_cases"]
        boundary_supports = cfg["supports"]
    else:
        boundary_loads = point_load_cases
        boundary_supports = point_supports
    common = dict(
        wall_thickness=cfg["wall_thickness"], E=cfg["material"]["E"],
        nu=cfg["material"]["nu"], loads=boundary_loads,
        supports=boundary_supports, ribs=ribs, rib_thicknesses=thicknesses,
    )

    def run_series() -> list[dict]:
        rows = []
        for nx, ny in meshes:
            started = time.perf_counter()
            model = StructuredQ1SolidReferenceModel(
                40.0, 20.0, nx, ny,
                footprint_samples=sample_counts[-1], **common,
            )
            response = model.analyze()
            rows.append({
                "nx": nx, "ny": ny, "ndof": model.ndof,
                "active_grid_nodes": len(model._active_grid_nodes),
                "footprint_samples": sample_counts[-1],
                "compliance": response.compliance,
                "load_compliances": response.load_compliances,
                "elapsed_seconds": time.perf_counter() - started,
            })
        for index, row in enumerate(rows):
            row["relative_change_from_previous"] = (
                None if index == 0 else row["compliance"] /
                rows[index - 1]["compliance"] - 1.0
            )
        return rows

    solid = run_series()
    sample_mesh = meshes[-1]
    footprint_scan = []
    for samples in sample_counts:
        started = time.perf_counter()
        model = StructuredQ1SolidReferenceModel(
            40.0, 20.0, sample_mesh[0], sample_mesh[1],
            footprint_samples=samples, **common,
        )
        response = model.analyze()
        footprint_scan.append({
            "nx": sample_mesh[0], "ny": sample_mesh[1],
            "footprint_samples": samples, "ndof": model.ndof,
            "compliance": response.compliance,
            "elapsed_seconds": time.perf_counter() - started,
        })
    finest_change = solid[-1]["relative_change_from_previous"]
    payload = {
        "schema_version": 1,
        "process_status": "complete",
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "study_type": "fixed_layout_response_only",
        "model": "structured_q1_3d_solid_with_physical_rib_footprint_fraction",
        "case": 2,
        "boundary_semantics": args.boundary_semantics,
        "baseline_run_id": baseline_run_id,
        "source": portable_artifact_path(source, code_root),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "ribbed_solid": solid,
        "footprint_scan": footprint_scan,
        "strict_convergence_tolerance": 0.01,
        "engineering_convergence_tolerance": 0.05,
        "strict_convergence_achieved": (
            finest_change is not None and finest_change <= 0.01
        ),
        "engineering_convergence_achieved": (
            finest_change is not None and finest_change <= 0.05
        ),
        "interpretation": (
            "The solid model has conforming shared nodes through the base/rib "
            "volume and treats rib intersections as a material union. Its "
            "physical footprint sampling and global mesh trend must both be "
            "reported before using it as an independent reference."
        ),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "executable_provenance": source_provenance(code_root, Path(__file__).resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(strict_json_dumps(payload, indent=2), encoding="utf-8")
    print(strict_json_dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
