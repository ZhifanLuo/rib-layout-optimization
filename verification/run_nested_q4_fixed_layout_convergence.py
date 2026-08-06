"""Verify fixed-layout convergence with nested local Q4 refinement."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def parse_factors(value: str) -> list[int]:
    factors = [int(item.strip()) for item in value.split(",")]
    if any(factor < 1 for factor in factors):
        raise ValueError("refinement factors must be positive")
    return factors


def segment_intersection(p0, p1, q0, q1):
    p0, p1, q0, q1 = (np.asarray(item, float) for item in (p0, p1, q0, q1))
    r, s = p1-p0, q1-q0
    cross = float(r[0]*s[1]-r[1]*s[0])
    if abs(cross) <= 1.0e-12:
        return None
    qp = q0-p0
    t = float((qp[0]*s[1]-qp[1]*s[0])/cross)
    u = float((qp[0]*r[1]-qp[1]*r[0])/cross)
    if 1.0e-10 < t < 1.0-1.0e-10 and 1.0e-10 < u < 1.0-1.0e-10:
        return p0+t*r
    return None


def fixed_refinement_regions(ribs):
    regions = [
        [18.0, 22.0, 0.0, 4.0],
        [0.0, 4.0, 0.0, 4.0],
        [36.0, 40.0, 0.0, 4.0],
    ]
    for rib in ribs:
        p0, p1 = np.asarray(rib.p0, float), np.asarray(rib.p1, float)
        regions.append([
            max(0.0, float(min(p0[0], p1[0])-0.5)),
            min(40.0, float(max(p0[0], p1[0])+0.5)),
            max(0.0, float(min(p0[1], p1[1])-0.5)),
            min(20.0, float(max(p0[1], p1[1])+0.5)),
        ])
    for first, rib_a in enumerate(ribs):
        for rib_b in ribs[first+1:]:
            point = segment_intersection(rib_a.p0, rib_a.p1, rib_b.p0, rib_b.p1)
            if point is not None:
                regions.append([
                    max(0.0, float(point[0]-1.0)),
                    min(40.0, float(point[0]+1.0)),
                    max(0.0, float(point[1]-1.0)),
                    min(20.0, float(point[1]+1.0)),
                ])
    return regions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factors", default="1,2,4,8")
    parser.add_argument(
        "--boundary", choices=("point", "patch", "hard_patch"), default="point",
        help="use committed point semantics or the fixed 4 mm regularized pads",
    )
    args = parser.parse_args()
    started_utc = datetime.now(timezone.utc).isoformat()
    root = args.project_root.resolve()
    code_root = root/"Code"
    sys.path.insert(0, str(code_root))
    from rib_layout_serialization import portable_artifact_path, strict_json_dumps
    from rib_layout_algorithms.model import Rib
    from rib_layout_algorithms.model_shell_local import (
        LocallyRefinedShellStiffenedPlateModel,
    )
    from rib_layout_core import load_case
    from tools.run_robustness_study import source_provenance

    source = args.source.resolve()
    record = json.loads(source.read_text(encoding="utf-8"))
    stage = record["final_stage"]
    cfg = load_case(2, quick=False)
    # Restore the committed GitHub/main boundary semantics explicitly.  The
    # working tree contains later patch-boundary experiments, but this study
    # is intended to diagnose the original Q4 model.
    if args.boundary == "point":
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
    elif args.boundary == "patch":
        loads = [{
            "weight": 1.0,
            "forces": [{
                "point": [20.0, 0.0], "patch_size": [4.0, 4.0],
                "profile": "cosine", "value": [0.0, -100.0, 0.0],
            }],
        }]
        supports = {
            "type": "patch_springs",
            "patches": [
                {"center": [0.0, 0.0], "size": [4.0, 4.0]},
                {"center": [40.0, 0.0], "size": [4.0, 4.0]},
            ],
            "components": [0, 1, 2], "stiffness": 1.0e6,
            "profile": "cosine",
        }
    else:
        loads = [{
            "weight": 1.0,
            "forces": [{
                "point": [20.0, 0.0], "patch_size": [4.0, 4.0],
                "profile": "cosine", "value": [0.0, -100.0, 0.0],
            }],
        }]
        supports = {
            "type": "points",
            "points": [[0.0, 0.0], [40.0, 0.0]],
            "patch_size": [4.0, 4.0],
        }
    ribs = [
        Rib(
            tuple(item["p0"]), tuple(item["p1"]), float(item["height"]),
            item["name"], int(cfg["rib"]["segments"]),
        )
        for item in stage["ribs"]
    ]
    thicknesses = [float(item["thickness"]) for item in stage["ribs"]]
    regions = fixed_refinement_regions(ribs)
    common = dict(
        wall_thickness=cfg["wall_thickness"], E=cfg["material"]["E"],
        nu=cfg["material"]["nu"], loads=loads, supports=supports,
        interface_subdivisions_per_cell=cfg.get(
            "interface_subdivisions_per_cell", 2,
        ),
        linear_solver=cfg.get("linear_solver", "auto"),
        linear_solver_threads=1, sensitivity_workers=1,
    )
    factors = parse_factors(args.factors)
    base_meshes = [(20, 10), (40, 20), (80, 40), (160, 80)]
    rows = []
    for factor in factors:
        for nx, ny in base_meshes:
            started = time.perf_counter()
            model = LocallyRefinedShellStiffenedPlateModel(
                40.0, 20.0, nx, ny, refinement_factor=factor,
                refinement_regions=regions, refinement_margin=0.0,
                **common,
            )
            response = model.analyze(ribs, thicknesses)
            rows.append({
                "factor": factor, "base_nx": nx, "base_ny": ny,
                "effective_nx": model.nx, "effective_ny": model.ny,
                "nnode": model.nnode, "ndof": model.ndof,
                "compliance": response.compliance,
                "load_compliances": response.load_compliances,
                "root_integration_points": sum(
                    len(model.rib_bottom_points(rib)) for rib in ribs
                ),
                "elapsed_seconds": time.perf_counter()-started,
            })
    for factor in factors:
        subset = [row for row in rows if row["factor"] == factor]
        for index, row in enumerate(subset):
            row["relative_change_from_previous"] = (
                None if index == 0 else row["compliance"] /
                subset[index-1]["compliance"]-1.0
            )
    payload = {
        "schema_version": 1,
        "process_status": "complete",
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "study_type": "fixed_layout_response_only",
        "model": "original_q4_shell_nested_local_refinement",
        "case": 2,
        "baseline_run_id": "002_mesh_[40,20]",
        "source": portable_artifact_path(source, code_root),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "boundary_semantics": (
            "committed_point_load_and_point_support"
            if args.boundary == "point" else
            "fixed_4mm_cosine_load_and_support_springs"
            if args.boundary == "patch" else
            "fixed_4mm_cosine_load_and_hard_support_pads"
        ),
        "region_definition": {
            "load_patch": [18.0, 22.0, 0.0, 4.0],
            "support_pads": [[0.0, 4.0, 0.0, 4.0], [36.0, 40.0, 0.0, 4.0]],
            "rib_root_margin": 0.5,
            "rib_intersection_half_width": 1.0,
            "refinement_margin": 0.0,
        },
        "regions": regions,
        "factors": factors,
        "base_meshes": base_meshes,
        "rows": rows,
        "strict_convergence_tolerance": 0.01,
        "engineering_convergence_tolerance": 0.05,
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
