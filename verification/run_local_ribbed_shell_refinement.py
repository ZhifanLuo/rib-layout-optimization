"""Run a fixed-layout local-refinement study for the Case-II shell model."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-nx", type=int, default=20)
    parser.add_argument("--base-ny", type=int, default=10)
    parser.add_argument(
        "--factors", default="1,2,4,8,16",
        help="comma-separated local refinement factors",
    )
    args = parser.parse_args()
    started_utc = datetime.now(timezone.utc).isoformat()

    project_root = args.project_root.resolve()
    code_root = project_root / "Code"
    sys.path.insert(0, str(code_root))
    from rib_layout_serialization import portable_artifact_path, strict_json_dumps
    from rib_layout_algorithms.model import Rib
    from rib_layout_algorithms.model_shell_local import (
        LocallyRefinedShellStiffenedPlateModel,
    )
    from rib_layout_core import load_case
    from tools.run_robustness_study import source_provenance

    source = args.source.resolve()
    aggregate = json.loads(source.read_text(encoding="utf-8"))
    baseline = next(
        record for record in aggregate["runs"]
        if record["run_id"] == "002_mesh_[40,20]"
    )
    stage = baseline["final_stage"]
    if stage["rib_count"] != 5 or len(stage["ribs"]) != 5:
        raise RuntimeError("expected the frozen Case-II five-rib baseline")
    factors = [int(value.strip()) for value in args.factors.split(",") if value.strip()]
    if not factors or any(value < 1 for value in factors):
        raise ValueError("factors must be positive integers")

    cfg = load_case(2, quick=False)
    width, height = (float(value) for value in cfg["domain"])
    ribs = [
        Rib(
            tuple(item["p0"]), tuple(item["p1"]), float(item["height"]),
            item["name"], int(cfg["rib"]["segments"]),
        )
        for item in stage["ribs"]
    ]
    # Thickness, not height, is the design variable used by the FE model.
    thicknesses = [float(item["thickness"]) for item in stage["ribs"]]
    common = dict(
        wall_thickness=cfg["wall_thickness"], E=cfg["material"]["E"],
        nu=cfg["material"]["nu"], loads=cfg["load_cases"],
        supports=cfg["supports"], refinement_ribs=ribs,
        interface_subdivisions_per_cell=cfg["interface_subdivisions_per_cell"],
        linear_solver=cfg.get("linear_solver", "auto"),
        linear_solver_threads=1, sensitivity_workers=1,
    )
    rows = []
    for factor in factors:
        started = time.perf_counter()
        model = LocallyRefinedShellStiffenedPlateModel(
            width, height, args.base_nx, args.base_ny,
            refinement_factor=factor, **common,
        )
        response = model.analyze(ribs, thicknesses)
        rows.append({
            "factor": factor, "nx": model.nx, "ny": model.ny,
            "ndof": model.ndof,
            "root_points": sum(len(model.rib_bottom_points(rib)) for rib in ribs),
            "compliance": response.compliance,
            "load_compliances": response.load_compliances,
            "elapsed_seconds": time.perf_counter() - started,
        })
    rows.sort(key=lambda item: item["factor"])
    for index, row in enumerate(rows):
        row["relative_change_from_previous"] = (
            None if index == 0 else row["compliance"] /
            rows[index - 1]["compliance"] - 1.0
        )

    no_rib = []
    for factor in factors:
        started = time.perf_counter()
        model = LocallyRefinedShellStiffenedPlateModel(
            width, height, args.base_nx, args.base_ny,
            refinement_factor=factor, **{
                key: value for key, value in common.items()
                if key != "refinement_ribs"
            },
        )
        response = model.analyze([], [])
        no_rib.append({
            "factor": factor, "nx": model.nx, "ny": model.ny,
            "ndof": model.ndof, "compliance": response.compliance,
            "elapsed_seconds": time.perf_counter() - started,
        })
    entry_tool = Path(__file__).resolve()
    payload = {
        "schema_version": 1,
        "process_status": "complete",
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "study_type": "fixed_layout_response_only",
        "case": 2,
        "baseline_run_id": baseline["run_id"],
        "base_mesh": [args.base_nx, args.base_ny],
        "factors": factors,
        "thickness_source": "final_stage.ribs[*].thickness",
        "source": portable_artifact_path(source, code_root),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "baseline_layout_canonical_sha256": canonical_hash(stage["ribs"]),
        "current_config_canonical_sha256": canonical_hash(cfg),
        "script_sha256": hashlib.sha256(entry_tool.read_bytes()).hexdigest(),
        "executable_provenance": source_provenance(code_root, entry_tool),
        "ribbed_shell": rows,
        "no_rib_shell": no_rib,
        "convergence_criterion": "successive compliance change below 1 percent",
        "convergence_assessment": (
            "not achieved if the finest reported ribbed-shell change remains "
            "above the stated criterion"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(strict_json_dumps(payload, indent=2), encoding="utf-8")
    print(strict_json_dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
