"""Plan or run all five fixed-base/parallel-layer SIMP comparison cases.

Default is read-only planning. Execution requires --execute and a new output
directory outside Code; it never overwrites rib-layout results or old evidence.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import sys
import traceback

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rib_layout_env import configure_runtime
configure_runtime()
import numpy as np
from rib_layout_core import load_case
from rib_layout_algorithms.density_shell import ParallelLayerModel
from tools.run_robustness_study import source_provenance

CODE_ROOT = Path(__file__).resolve().parents[1]


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def output_path(value: Path) -> Path:
    target = value.resolve()
    if target == CODE_ROOT or CODE_ROOT in target.parents:
        raise ValueError("output must be outside Code")
    if target.exists():
        raise ValueError("output must be a new, nonexisting directory")
    return target


def response_summary(response: dict) -> dict:
    return {key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in response.items()
            if key not in ("physical_density", "physical_gradient", "gradient", "displacement")}


def save_plots(folder: Path, model: ParallelLayerModel, result: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    extent = [0, model.width, 0, model.height]
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    im = ax.imshow(result["final"]["physical_density"].reshape(model.ny, model.nx),
                   origin="lower", extent=extent, cmap="gray_r", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set(title="Continuous SIMP physical density", xlabel="x (mm)", ylabel="y (mm)")
    fig.colorbar(im, ax=ax, label="Physical density")
    fig.savefig(folder/"density.png", dpi=180)
    plt.close(fig)
    history = result["history"]
    fig, axes = plt.subplots(3, 1, figsize=(7, 8), constrained_layout=True)
    axes[0].plot([r["iteration"] for r in history], [r["compliance"] for r in history])
    axes[0].set(ylabel="Weighted compliance (N mm)", title=result["termination_reason"])
    axes[1].plot([r["iteration"] for r in history], [r["volume"]/model.budget for r in history])
    axes[1].axhline(1, color="k", linestyle="--")
    axes[1].set(xlabel="OC iteration", ylabel="Design volume / budget")
    axes[2].plot([r["iteration"] for r in history], [r["raw_change"] for r in history], label="Raw design variables")
    axes[2].plot([r["iteration"] for r in history], [r["physical_change"] for r in history], label="Physical density")
    axes[2].set(xlabel="OC iteration", ylabel="Maximum density change")
    axes[2].legend()
    fig.savefig(folder/"history.png", dpi=160)
    plt.close(fig)
    displacement = result["final"]["displacement"]
    fig, axes = plt.subplots(1, displacement.shape[1], figsize=(6*displacement.shape[1], 4),
                             squeeze=False, constrained_layout=True)
    for m, ax in enumerate(axes[0]):
        field = displacement[2::6, m].reshape(model.ny+1, model.nx+1)
        im = ax.imshow(field, origin="lower", extent=extent, cmap="coolwarm")
        ax.set(title=f"Base w, load {m+1} (mm)", xlabel="x (mm)", ylabel="y (mm)")
        fig.colorbar(im, ax=ax)
    fig.savefig(folder/"displacement.png", dpi=160)
    plt.close(fig)


def execute_case(spec: dict, folder: Path) -> dict:
    folder.mkdir(exist_ok=False)
    cfg, options = spec["config"], spec["options"]
    checkpoint = dict(status="started", started_utc=utc(), specification=spec)
    write_json(folder/"checkpoint.json", checkpoint)
    try:
        model = ParallelLayerModel(cfg, **options)
        with (folder/"history.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = ["iteration", "compliance", "volume", "raw_change", "physical_change",
                      "relative_objective_change", "relative_residual", "analyses"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            def progress(row):
                writer.writerow(row)
                stream.flush()
                if row["iteration"] == 1 or row["iteration"] % 10 == 0:
                    print(f"case {cfg['number']} iteration {row['iteration']}: "
                          f"C={row['compliance']:.9g}, V={row['volume']:.9g}, "
                          f"raw_change={row['raw_change']:.3g}, "
                          f"physical_change={row['physical_change']:.3g}", flush=True)
                checkpoint.update(last_iteration=row)
                write_json(folder/"checkpoint.json", checkpoint)
            result = model.optimize(**spec["optimization"], callback=progress)
        rho = result["final"]["physical_density"]
        np.save(folder/"raw_density.npy", result["raw_density"])
        np.save(folder/"physical_density.npy", rho)
        np.save(folder/"displacement.npy", result["final"]["displacement"])
        np.savetxt(folder/"density.csv", rho.reshape(model.ny, model.nx), delimiter=",")
        summary = dict(case=cfg["number"], config=cfg, model=model.metadata(),
                       optimization=spec["optimization"],
                       initialization=result["initialization"],
                       initial_uniform=response_summary(result["initial"]),
                       final_continuous=response_summary(result["final"]),
                       converged=result["converged"], termination_reason=result["termination_reason"],
                       iterations=result["iterations"], optimization_analyses=result["analysis_count"],
                       total_analyses=model.analysis_count, elapsed_seconds=result["elapsed_seconds"],
                       grayness=float(np.mean(4*rho*(1-rho))),
                       process_status="complete", completed_utc=utc())
        if not spec["no_images"]:
            save_plots(folder, model, result)
        write_json(folder/"results.json", summary)
        inventory = {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
                     for p in folder.iterdir() if p.is_file() and p.name != "checkpoint.json"}
        write_json(folder/"SHA256SUMS.json", inventory)
        checkpoint.update(status="complete", completed_utc=utc(),
                          converged=result["converged"], termination_reason=result["termination_reason"])
        write_json(folder/"checkpoint.json", checkpoint)
        return summary
    except Exception as error:
        checkpoint.update(status="failed", completed_utc=utc(), error_type=type(error).__name__,
                          error=str(error), traceback=traceback.format_exc())
        write_json(folder/"checkpoint.json", checkpoint)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", type=int, choices=range(1, 6), default=[1, 2, 3, 4, 5])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--mesh", nargs=2, type=int, metavar=("NX", "NY"), help="explicit diagnostic mesh override")
    parser.add_argument("--layer-thickness", type=float, help="mm; default rib height minus half base thickness")
    parser.add_argument("--penal", type=float, default=3.)
    parser.add_argument("--emin-ratio", type=float, default=0.)
    parser.add_argument("--filter-radius", type=float, help="mm; default 1.5 times largest cell dimension")
    parser.add_argument("--linear-solver", choices=("auto", "pardiso", "superlu"), default="auto")
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--move", type=float, default=.1)
    parser.add_argument("--density-tolerance", type=float, default=.005)
    parser.add_argument("--raw-density-tolerance", type=float, default=.01,
                        help="maximum raw-variable change, aligned with the top88 stopping measure")
    parser.add_argument("--objective-tolerance", type=float, default=1e-4)
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args(argv)
    try:
        output = output_path(args.output)
        if args.mesh and min(args.mesh) < 1:
            raise ValueError("mesh dimensions must be positive")
        if len(set(args.cases)) != len(args.cases):
            raise ValueError("duplicate cases are not allowed")
        if args.max_iterations < 1 or not (0 < args.move <= 1):
            raise ValueError("positive iteration limit and 0<move<=1 required")
        if not all(np.isfinite([args.penal, args.emin_ratio, args.density_tolerance,
                               args.raw_density_tolerance, args.objective_tolerance])) or not (
                args.penal >= 1 and 0 <= args.emin_ratio < 1 and args.density_tolerance > 0 and
                args.raw_density_tolerance > 0 and args.objective_tolerance > 0):
            raise ValueError("finite penal>=1, 0<=emin-ratio<1 and positive tolerances required")
        if args.filter_radius is not None and (not np.isfinite(args.filter_radius) or args.filter_radius <= 0):
            raise ValueError("filter radius must be finite and positive")
        specs = []
        for number in args.cases:
            cfg = load_case(number)
            if args.mesh:
                cfg["mesh"] = args.mesh
            options = dict(layer_thickness=args.layer_thickness, penal=args.penal,
                           emin_ratio=args.emin_ratio, filter_radius=args.filter_radius,
                           linear_solver=args.linear_solver)
            # Validate without building/factorizing a model during read-only planning.
            td = args.layer_thickness if args.layer_thickness is not None else cfg["rib"]["height"]-cfg["wall_thickness"]/2
            capacity = float(np.prod(cfg["domain"])*td)
            if not np.isfinite(td) or td <= 0 or capacity < cfg["volume_bound"]-1e-10:
                raise ValueError(f"case {number}: positive layer thickness and sufficient volume capacity required")
            specs.append(dict(config=cfg, options=options, no_images=args.no_images,
                              diagnostic_mesh_override=bool(args.mesh),
                              optimization=dict(max_iterations=args.max_iterations, move=args.move,
                                                density_tolerance=args.density_tolerance,
                                                raw_density_tolerance=args.raw_density_tolerance,
                                                objective_tolerance=args.objective_tolerance)))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    plan = dict(output=str(output), execute=args.execute, cases=specs)
    print(json.dumps(plan, indent=2, allow_nan=False), flush=True)
    if not args.execute:
        return 0
    output.mkdir(parents=True, exist_ok=False)
    before = source_provenance(CODE_ROOT, Path(__file__))
    manifest = dict(status="started", started_utc=utc(), plan=plan,
                    plan_sha256=hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
                    source_before=before, cases=[])
    write_json(output/"manifest.json", manifest)
    try:
        for spec in specs:
            number = spec["config"]["number"]
            result = execute_case(spec, output/f"example_{number}")
            manifest["cases"].append(dict(case=number, process_status="complete",
                                          converged=result["converged"],
                                          termination_reason=result["termination_reason"],
                                          results=f"example_{number}/results.json",
                                          results_sha256=hashlib.sha256((output/f"example_{number}/results.json").read_bytes()).hexdigest(),
                                          inventory_sha256=hashlib.sha256((output/f"example_{number}/SHA256SUMS.json").read_bytes()).hexdigest()))
            write_json(output/"manifest.json", manifest)
        after = source_provenance(CODE_ROOT, Path(__file__))
        manifest["source_after"] = after
        if before["source_aggregate_sha256"] != after["source_aggregate_sha256"]:
            raise RuntimeError("source changed during density comparison")
        manifest.update(status="complete", completed_utc=utc(), source_unchanged=True)
        write_json(output/"manifest.json", manifest)
    except Exception as error:
        manifest.update(status="failed", completed_utc=utc(), error_type=type(error).__name__, error=str(error))
        write_json(output/"manifest.json", manifest)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
