"""One-FEA 160x80 response reanalysis of the current fixed Case-II baseline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


FROZEN_STATE_CHECKER_AGGREGATE = (
    "3b06e4dab1fedffb90af0c7b3ee2c07676fb02cd3a87d20d4d45de3a7d99f6a0"
)


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
    source = args.source.resolve()
    output = args.output.resolve()
    sys.path.insert(0, str(code_root))
    from rib_layout_serialization import portable_artifact_path, strict_json_dumps
    from tools.run_robustness_study import reanalyze_on_mesh, source_provenance

    aggregate = json.loads(source.read_text(encoding="utf-8"))
    if aggregate.get("process_status") != "complete":
        raise RuntimeError("robustness archive is not complete")
    provenance = aggregate.get("source_provenance", {})
    archived_tool_aggregate = provenance.get("source_aggregate_sha256")
    if not archived_tool_aggregate:
        raise RuntimeError("robustness archive lacks executable provenance")

    baselines = [
        record
        for record in aggregate.get("runs", [])
        if record.get("factor") == "mesh"
        and record.get("value") == [40, 20]
        and record.get("process_status") == "complete"
    ]
    if len(baselines) != 1:
        raise RuntimeError(f"expected one completed 40x20 baseline, found {len(baselines)}")
    baseline = baselines[0]
    stage = baseline["final_stage"]
    config = baseline["config"]
    if stage.get("rib_count") != len(stage.get("ribs", [])) or not stage.get("ribs"):
        raise RuntimeError("baseline stage has an invalid rib inventory")

    fixed = aggregate.get("fixed_layout_mesh_reanalysis", {})
    if fixed.get("process_status") != "complete":
        raise RuntimeError("the built-in fixed-layout sequence is not complete")
    if fixed.get("baseline_run_id") != baseline.get("run_id"):
        raise RuntimeError("fixed-layout sequence does not use the selected baseline")
    meshes = fixed.get("meshes", [])
    if [item.get("mesh") for item in meshes] != [[20, 10], [40, 20], [80, 40]]:
        raise RuntimeError("unexpected built-in fixed-layout mesh sequence")
    if any(item.get("analysis_count") != 1 for item in meshes):
        raise RuntimeError("built-in fixed-layout records must each use exactly one FEA")
    if any(item.get("optimization_held_fixed") is not True for item in meshes):
        raise RuntimeError("built-in response sequence did not hold the design fixed")

    started_utc = datetime.now(timezone.utc).isoformat()
    response = reanalyze_on_mesh(stage, config, (160, 80))
    if response.get("analysis_count") != 1 or response.get("optimization_held_fixed") is not True:
        raise RuntimeError("160x80 extension was not exactly one fixed-layout FEA")

    current_identity = source_provenance(code_root, Path(__file__).resolve())
    if current_identity.get("source_aggregate_sha256") != archived_tool_aggregate:
        raise RuntimeError("current executable source changed after the robustness run")
    diagnostics_root = source.parents[1]
    source_reference = portable_artifact_path(source, diagnostics_root)
    if Path(source_reference).is_absolute():
        raise RuntimeError("source must lie inside the diagnostics-current archive")
    payload = {
        "schema_version": 2,
        "process_status": "complete",
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": (
            "Response-only finite-element reanalysis of the exact same baseline "
            "layout; optimization, filtering, and rationalization are held fixed."
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
        "built_in_meshes_canonical_sha256": canonical_hash(meshes),
        "frozen_state_checker_source_aggregate_sha256": FROZEN_STATE_CHECKER_AGGREGATE,
        "robustness_tool_source_aggregate_sha256": archived_tool_aggregate,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "executable_provenance": current_identity,
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
