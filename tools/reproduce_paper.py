"""Run the manuscript's Example V from its explicit archived configuration.

The default prints a plan only. Use --execute and a fresh output directory
outside this source tree to run the complete workflow without quick overrides.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.run_revision_experiments import execute_trial, write_json
from tools.run_robustness_study import source_provenance

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = CODE_ROOT / "configs/paper/example5.json"


def build_spec(config_path: Path = CONFIG) -> dict:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg["number"] != 5 or cfg["mesh"] != [80, 40] or cfg["wall_thickness"] != 1.0:
        raise ValueError("The manuscript Example V requires case 5, an 80x40 mesh and a 1.0 mm base")
    return dict(id="paper_example5", mode="full", config=deepcopy(cfg),
                input_stage=None, source=None, source_sha256=None,
                input_design_sha256=None, reanalysis_mesh=None,
                iteration_cap=None, write_images=True, replay_stages=True)


def output_path(value: Path) -> Path:
    result = value.resolve()
    if result == CODE_ROOT or CODE_ROOT in result.parents:
        raise ValueError("Output must be outside the source tree")
    if result.exists():
        raise ValueError("Output must be a new, nonexisting directory")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, choices=(5,), default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Execute the full printed plan")
    args = parser.parse_args(argv)
    try:
        out = output_path(args.output)
        spec = build_spec()
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    config_hash = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    plan = dict(case=5, config_file=CONFIG.relative_to(CODE_ROOT).as_posix(),
                config_sha256=config_hash, output=str(out), execute=args.execute,
                iteration_cap=None, configuration=spec["config"])
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return 0
    # mkdir's exist_ok=False also prevents a race with another launch.
    out.mkdir(parents=True, exist_ok=False)
    before = source_provenance(CODE_ROOT, Path(__file__))
    record = dict(status="started", started_utc=datetime.now(timezone.utc).isoformat(),
                  plan=plan, source_before=before)
    write_json(out / "reproduction.json", record)
    try:
        result = execute_trial(spec, out)
        after = source_provenance(CODE_ROOT, Path(__file__))
        if after["source_aggregate_sha256"] != before["source_aggregate_sha256"]:
            raise RuntimeError("Source changed during reproduction")
        if hashlib.sha256(CONFIG.read_bytes()).hexdigest() != config_hash:
            raise RuntimeError("Paper configuration changed during reproduction")
        record.update(status="complete", result=result, source_after=after,
                      completed_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error),
                      completed_utc=datetime.now(timezone.utc).isoformat())
        write_json(out / "reproduction.json", record)
        raise
    write_json(out / "reproduction.json", record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
