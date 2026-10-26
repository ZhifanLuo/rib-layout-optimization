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


def script_identity(code_root: Path, source_provenance, portable_path) -> dict:
    """Bind the script digest and executable provenance to this entry point."""
    entry_tool = Path(__file__).resolve()
    provenance = source_provenance(code_root, entry_tool)
    provenance["entry_tool"] = portable_path(entry_tool, code_root)
    return {
        "script_sha256": hashlib.sha256(entry_tool.read_bytes()).hexdigest(),
        "executable_provenance": provenance,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    code_root = project_root / "Code"
    output = args.output.resolve()
    sys.path.insert(0, str(code_root))
    from rib_layout_serialization import portable_artifact_path, strict_json_dumps
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
    identity = script_identity(
        code_root, source_provenance, portable_artifact_path
    )
    diagnostics_root = output.parents[1]
    source_reference = portable_artifact_path(source, diagnostics_root)
    if Path(source_reference).is_absolute():
        raise RuntimeError("response source must lie inside the diagnostics archive root")
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
        "source": source_reference,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "baseline_run_id": baseline["run_id"],
        "baseline_optimization_mesh": baseline["config"]["mesh"],
        "baseline_stage_name": stage["name"],
        "baseline_rib_count": stage["rib_count"],
        "baseline_config_canonical_sha256": canonical_hash(config),
        "baseline_layout_canonical_sha256": canonical_hash(stage["ribs"]),
        "baseline_stage_canonical_sha256": canonical_hash(stage),
        **identity,
        "response": response,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(strict_json_dumps(payload, indent=2), encoding="utf-8")
    print(
        f"mesh=160x80, ribs={stage['rib_count']}, "
        f"C={response['compliance']:.12g}, FEA={response['analysis_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
