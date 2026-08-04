"""One-FEA response reanalysis of the frozen Case-II five-rib baseline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    code_root = project_root / "Code"
    sys.path.insert(0, str(code_root))
    from tools.run_robustness_study import reanalyze_on_mesh, source_provenance

    source = args.source.resolve()
    aggregate = json.loads(source.read_text(encoding="utf-8"))
    baseline = next(
        record
        for record in aggregate["runs"]
        if record["run_id"] == "002_mesh_[40,20]"
    )
    stage = baseline["final_stage"]
    config = baseline["config"]
    if stage["rib_count"] != 5 or len(stage["ribs"]) != 5:
        raise RuntimeError("the frozen Case-II baseline is not the expected five-rib layout")
    fixed = aggregate["fixed_layout_mesh_reanalysis"]
    if fixed["baseline_run_id"] != baseline["run_id"]:
        raise RuntimeError("fixed-layout series does not use the selected baseline")

    started_utc = datetime.now(timezone.utc).isoformat()
    response = reanalyze_on_mesh(stage, config, (160, 80))
    if response["analysis_count"] != 1:
        raise RuntimeError("response-only extension did not use exactly one FEA")
    entry_tool = code_root / "tools" / "run_robustness_study.py"
    payload = {
        "schema_version": 1,
        "process_status": "complete",
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": (
            "Response-only finite-element reanalysis of the exact same five-rib "
            "baseline layout; optimization, filtering, and rationalization are held fixed."
        ),
        "convergence_claim": "not_assessed_by_this_single_extension",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "baseline_run_id": baseline["run_id"],
        "baseline_optimization_mesh": baseline["config"]["mesh"],
        "baseline_stage_name": stage["name"],
        "baseline_rib_count": stage["rib_count"],
        "baseline_config_canonical_sha256": canonical_hash(config),
        "baseline_layout_canonical_sha256": canonical_hash(stage["ribs"]),
        "baseline_stage_canonical_sha256": canonical_hash(stage),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "executable_provenance": source_provenance(code_root, entry_tool),
        "response": response,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"mesh=160x80, ribs={stage['rib_count']}, "
        f"C={response['compliance']:.12g}, FEA={response['analysis_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
