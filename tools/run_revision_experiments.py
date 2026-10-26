"""Bounded numerical revision trials; default is a read-only plan.

Use --execute and an explicit fresh --output directory to run. Each trial has
started/complete/failed checkpoints and source/input/output hashes. Completed
trials can be verified and skipped with --resume; interrupted trials require a
new output root (they are never silently overwritten).
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
from itertools import product
import json
from pathlib import Path
import sys
import time
import traceback

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rib_layout_core import (
    RunArtifacts, build_model, load_case, make_case_config, run_optimization,
)
from rib_layout_algorithms.model import Rib
from rib_layout_algorithms.optimization import OptimizationRun, RibLayoutOptimizer, Stage
from rib_layout_output import save_optimization_results, stage_to_dict
from rib_layout_serialization import strict_json_dumps
from tools.run_rationalization_diagnostic import saved_design
from tools.run_geometry_restart_diagnostic import validate_saved_compliance
from tools.run_robustness_study import mesh_value, source_provenance

CODE_ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("smooth", "direct_deletion_geometry", "direct_deletion_sizing")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_hash(value) -> str:
    return hashlib.sha256(strict_json_dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def write_json(path: Path, value) -> None:
    # Atomic checkpoint replacement only inside a newly reserved trial directory.
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(strict_json_dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=(
        "full", "no-addition", "compare", "sweep", "longitudinal", "reanalyze",
    ))
    parser.add_argument("--case", type=int, required=True, choices=(2, 3, 5))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, help="Saved standard results.json")
    parser.add_argument("--stage", default="geometry", help="Input stage for mesh replay")
    parser.add_argument("--execute", action="store_true", help="Run the printed plan")
    parser.add_argument("--resume", action="store_true", help="Verify and skip complete trials")
    parser.add_argument("--max-trials", type=int, default=1, help="Hard trial-count guard")
    parser.add_argument("--mesh", type=mesh_value, help="Full/no-addition/longitudinal mesh")
    parser.add_argument("--meshes", type=mesh_value, nargs="+", default=[(40, 20), (80, 40)],
                        help="Fixed-layout reanalysis meshes")
    parser.add_argument("--policies", choices=POLICIES, nargs="+", default=list(POLICIES))
    parser.add_argument("--alpha-multipliers", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--betas", type=float, nargs="+", default=[5.0, 10.0, 20.0])
    parser.add_argument("--beta-initial", type=float, default=1.0)
    parser.add_argument("--beta-increment", type=float, default=1.0)
    parser.add_argument("--longitudinal-count", type=int, default=3)
    parser.add_argument("--volume", type=float, help="Longitudinal volume (default: Vbound)")
    parser.add_argument("--iteration-cap", type=int, help="Explicit diagnostic cap, not formal")
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--replay-stages", action="store_true",
                        help="Separate per-load/deflection replay of each saved output stage")
    return parser


def source_input(path: Path, case: int, stage_name: str):
    raw = path.read_bytes()
    source = json.loads(raw.decode("utf-8-sig"))
    if source.get("example") != case or source.get("quick", False):
        raise ValueError("source must match case and must not be quick")
    cfg = make_case_config(source["revision_config"]) if "revision_config" in source else load_case(case)
    if list(source["mesh_used"]) != list(cfg["mesh"]):
        raise ValueError("source mesh differs from its reproducible configuration")
    if case == 5 and cfg["load_cases"] != load_case(5)["load_cases"]:
        raise ValueError("case 5 requires the fixed revision pressure -0.0001 N/mm²")
    stage = next(item for item in source["stages"] if item["name"] == stage_name)
    return cfg, deepcopy(stage), hashlib.sha256(raw).hexdigest()


def build_specs(args) -> list[dict]:
    if args.max_trials < 1:
        raise ValueError("max-trials must be positive")
    if args.iteration_cap is not None and args.iteration_cap < 1:
        raise ValueError("iteration-cap must be positive")
    if args.mode == "no-addition" and args.case not in (2, 5):
        raise ValueError("no-addition revision cases are 2 and 5")
    source_modes = ("compare", "sweep", "reanalyze")
    if args.mode in source_modes and args.source is None:
        raise ValueError("this mode requires --source")
    if args.mode not in source_modes and args.source is not None:
        raise ValueError("--source is only supported for compare/sweep/reanalyze")
    if args.mode in source_modes and args.mesh is not None:
        raise ValueError("source mesh is fixed; use --meshes for reanalysis")
    if args.mode in ("compare", "sweep") and args.stage != "geometry":
        raise ValueError("rationalization comparisons must start from geometry")
    base = load_case(args.case)
    input_stage, input_sha = None, None
    if args.source is not None:
        base, input_stage, input_sha = source_input(args.source, args.case, args.stage)
    if args.mesh is not None:
        base["mesh"] = list(args.mesh)
    base.pop("further_rationalization_relaxation", None)
    base["rationalization_relaxation"] = 0.05
    base["algorithm"]["enable_addition"] = args.mode != "no-addition"
    # Explicitly reset experiment controls inherited from saved revision inputs.
    base["algorithm"]["rationalization_policy"] = "smooth"
    base["algorithm"]["rationalization_quantile_multiplier"] = 1.0
    if args.iteration_cap is not None:
        for key in ("sizing_max_iterations", "active_set_max_iterations",
                    "geometry_max_iterations", "rationalization_max_iterations",
                    "rationalization_geometry_iterations"):
            base["algorithm"][key] = args.iteration_cap
    variants = [(None, None, None, None)]
    if args.mode == "compare":
        variants = [(p, 1.0, None, None) for p in args.policies]
    elif args.mode == "sweep":
        variants = [("smooth", a, b, None) for a, b in product(args.alpha_multipliers, args.betas)]
    elif args.mode == "reanalyze":
        variants = [(None, None, None, m) for m in args.meshes]
    specs = []
    for index, (policy, alpha, beta, mesh) in enumerate(variants, 1):
        cfg = deepcopy(base)
        if policy is not None:
            cfg["algorithm"]["rationalization_policy"] = policy
            cfg["algorithm"]["rationalization_quantile_multiplier"] = alpha
        if beta is not None:
            import math
            if not all(math.isfinite(v) and v > 0 for v in
                       (beta, args.beta_initial, args.beta_increment)) or args.beta_initial > beta:
                raise ValueError("beta controls must be finite, positive, and initial <= maximum")
            cfg["algorithm"].update(rationalization_beta=beta,
                                    rationalization_beta_initial=args.beta_initial,
                                    rationalization_beta_increment=args.beta_increment)
        RibLayoutOptimizer(object(), cfg)  # validate controls before any trial starts
        specs.append({
            "id": f"{index:03d}_{args.mode}_{policy or args.case}",
            "mode": args.mode, "config": cfg, "input_stage": deepcopy(input_stage),
            "source_sha256": input_sha,
            "input_design_sha256": object_hash(input_stage) if input_stage is not None else None,
            "source": str(args.source.resolve()) if args.source else None,
            "reanalysis_mesh": list(mesh) if mesh is not None else None,
            "longitudinal_count": args.longitudinal_count, "volume": args.volume,
            "iteration_cap": args.iteration_cap, "write_images": not args.no_images,
            "replay_stages": args.replay_stages,
        })
    if len(specs) > args.max_trials:
        raise ValueError(f"plan has {len(specs)} trials; explicitly raise --max-trials")
    return specs


def reserve_output(path: Path, resume: bool) -> Path:
    path = path.resolve()
    # Revision artifacts must never enter Code (in particular Code/results).
    if path == CODE_ROOT or CODE_ROOT in path.parents:
        raise ValueError("revision output must be outside Code")
    if path.exists():
        if not resume or not path.is_dir():
            raise FileExistsError("output exists; choose a new root or use --resume")
    else:
        path.mkdir(parents=True)
    return path


def longitudinal_layout(cfg: dict, count: int, volume: float | None):
    """Full-length x ribs at equally spaced interior y stations, equal thickness."""
    import numpy as np
    if count < 1:
        raise ValueError("longitudinal-count must be positive")
    target = cfg["volume_bound"] if volume is None else float(volume)
    if not np.isfinite(target) or target <= 0 or target > cfg["volume_bound"]:
        raise ValueError("baseline volume must be positive and <= Vbound")
    width, height = cfg["domain"]
    thickness = target/(count*width*cfg["rib"]["height"])
    if not cfg["rib"]["lower"] <= thickness <= cfg["rib"]["upper"]:
        raise ValueError("equal-volume baseline thickness is outside rib bounds")
    ribs = [Rib((0.0, float(y)), (width, float(y)), cfg["rib"]["height"],
                f"L{i}", cfg["rib"]["segments"])
            for i, y in enumerate(np.linspace(0, height, count+2)[1:-1], 1)]
    return ribs, np.full(count, thickness)


def response_record(model, ribs, thicknesses, result) -> dict:
    import numpy as np
    return {
        "compliance": float(result.compliance),
        "load_compliances": list(map(float, result.load_compliances)),
        "load_weights": list(map(float, model.load_weights)),
        "volume": RibLayoutOptimizer.volume(ribs, thicknesses),
        "max_abs_base_w_by_load": [float(np.max(np.abs(u[:model.ndof][2::6])))
                                   for u in result.displacements],
        "load_resultants": [f.reshape(-1, 6)[:, :3].sum(axis=0).tolist()
                            for f in model.load_vectors],
    }


def execute_trial(spec: dict, trial: Path) -> dict:
    cfg = deepcopy(spec["config"])
    mode = spec["mode"]
    started = time.perf_counter()
    diagnostics = {}
    if mode in ("full", "no-addition"):
        artifacts = run_optimization(cfg)
    else:
        optimizer = RibLayoutOptimizer(build_model(cfg), cfg)
        if mode == "longitudinal":
            ribs, thicknesses = longitudinal_layout(cfg, spec["longitudinal_count"], spec["volume"])
            result = optimizer.analyze(ribs, thicknesses)
            stage_name = "longitudinal_equal_volume"
        else:
            stage = spec["input_stage"]
            ribs, thicknesses = saved_design(stage, cfg["rib"]["segments"])
            result = optimizer.analyze(ribs, thicknesses)
            if not (0.0 < float(stage["compliance"]) < float("inf")) or not (
                0.0 < result.compliance < float("inf")
            ):
                raise ValueError("source/replay compliance must be finite and positive")
            mismatch = validate_saved_compliance(stage["name"], stage["compliance"], result.compliance, 1e-6)
            diagnostics["source_replay"] = {
                **response_record(optimizer.model, ribs, thicknesses, result),
                "relative_mismatch": mismatch, "analyses": 1,
                "saved_compliance": stage["compliance"],
            }
            if mode == "reanalyze":
                cfg["mesh"] = spec["reanalysis_mesh"]
                optimizer = RibLayoutOptimizer(build_model(cfg), cfg)
                result = optimizer.analyze(ribs, thicknesses)
                stage_name = "fixed_layout_reanalysis"
                diagnostics["optimization_held_fixed"] = True
            else:
                # Absolute bound uses the immutable saved geometry compliance.
                # Replay reconstructs displacements, and must pass before using it.
                from dataclasses import replace
                reference = replace(result, compliance=float(stage["compliance"]))
                ribs, thicknesses, result = optimizer.rationalize(ribs, thicknesses, reference, 0.05)
                stage_name = "rationalized"
                diagnostics["absolute_cref"] = 1.05*float(stage["compliance"])
                diagnostics["final_exact_bound_satisfied"] = result.compliance <= diagnostics["absolute_cref"]
                # A fallback may carry saved compliance; always verify the actual final design.
                exact = optimizer.analyze(ribs, thicknesses)
                diagnostics["final_verification"] = response_record(optimizer.model, ribs, thicknesses, exact)
                diagnostics["final_exact_bound_satisfied"] = exact.compliance <= diagnostics["absolute_cref"]
                if not diagnostics["final_exact_bound_satisfied"]:
                    raise RuntimeError("final exact compliance exceeds the shared absolute Cref")
                result = exact
        diagnostics["response"] = response_record(optimizer.model, ribs, thicknesses, result)
        run = OptimizationRun(stages=[Stage(stage_name, list(ribs), thicknesses.copy(),
                                            result.compliance, optimizer.analysis_count)], log=optimizer.log)
        if mode in ("compare", "sweep"):
            run.rationalization_histories["rationalized"] = deepcopy(optimizer.rationalization_history)
        artifacts = RunArtifacts(cfg, optimizer, run, time.perf_counter()-started)
    options = deepcopy(importlib.import_module(f"example{cfg['number']}").OUTPUT_OPTIONS)
    options["write_stage_images"] = spec["write_images"]
    output = save_optimization_results(artifacts, trial, options, quick=False)
    result_path = output/"results.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["revision_config"] = cfg
    payload["revision"] = {
        "mode": mode, "enable_addition": cfg["algorithm"].get("enable_addition", True),
        "iteration_cap": spec["iteration_cap"],
        "diagnostic_iteration_cap": spec["iteration_cap"] is not None,
        "input_design_sha256": spec["input_design_sha256"],
        "diagnostics": diagnostics,
    }
    if spec["replay_stages"]:
        model = build_model(cfg)
        payload["revision"]["stage_replay"] = [
            {"stage": stage.name, "analyses": 1,
             **response_record(model, stage.ribs, stage.thicknesses,
                               model.analyze(stage.ribs, stage.thicknesses))}
            for stage in artifacts.run.stages
        ]
    write_json(result_path, payload)
    return {"result": result_path.relative_to(trial).as_posix(),
            "result_sha256": sha256(result_path), "diagnostics": diagnostics}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    specs = build_specs(args)
    print(strict_json_dumps([{
        "trial": s["id"], "mode": s["mode"], "mesh": s["config"]["mesh"],
        "enable_addition": s["config"]["algorithm"]["enable_addition"],
        "policy": s["config"]["algorithm"]["rationalization_policy"],
        "input_design_sha256": s["input_design_sha256"],
    } for s in specs], indent=2), flush=True)
    if not args.execute:
        return 0
    root = reserve_output(args.output, args.resume)
    provenance = source_provenance(CODE_ROOT, Path(__file__))
    for spec in specs:
        trial = root/spec["id"]
        fingerprint = object_hash({"spec": spec, "source": provenance["source_aggregate_sha256"]})
        checkpoint = trial/"checkpoint.json"
        if trial.exists():
            previous = json.loads(checkpoint.read_text(encoding="utf-8"))
            if (args.resume and previous.get("status") == "complete"
                    and previous.get("fingerprint") == fingerprint
                    and sha256(trial/previous["result"]) == previous["result_sha256"]):
                print(f"Verified complete trial: {trial}", flush=True)
                continue
            raise FileExistsError(f"refusing to overwrite incomplete or mismatched trial: {trial}")
        trial.mkdir()
        record = {"status": "started", "started_utc": datetime.now(timezone.utc).isoformat(),
                  "fingerprint": fingerprint, "spec": spec, "executable_provenance": provenance}
        write_json(checkpoint, record)
        try:
            with (trial/"console.log").open("x", encoding="utf-8") as stream:
                with redirect_stdout(stream), redirect_stderr(stream):
                    result = execute_trial(spec, trial)
            after = source_provenance(CODE_ROOT, Path(__file__))
            if after["source_aggregate_sha256"] != provenance["source_aggregate_sha256"]:
                raise RuntimeError("source changed during trial")
            if spec["source"] and sha256(Path(spec["source"])) != spec["source_sha256"]:
                raise RuntimeError("input source changed during trial")
            record.update(result, status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                          source_after_sha256=after["source_aggregate_sha256"])
            write_json(checkpoint, record)
        except BaseException as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                          traceback=traceback.format_exc(), completed_utc=datetime.now(timezone.utc).isoformat())
            write_json(checkpoint, record)
            raise
        print(f"Completed trial: {trial}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
