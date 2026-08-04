"""Verify production compliance sensitivities with equilibrium reanalyses.

The default checks the independent thickness and endpoint variables actually
used by the mirror-reduced production optimizer.  Optional full-space checks
are retained as a separate diagnostic scope.  Mesh-trace changes and invalid
geometry perturbations remain explicit component records.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rib_layout_core import build_model, load_case
from rib_layout_algorithms.model import Rib
from rib_layout_algorithms.optimization import geometry_move_freeze_reasons
from rib_layout_algorithms.symmetry import build_mirror_variable_map, mirror_axes
from tools.run_geometry_restart_diagnostic import (
    stage_design,
    validate_saved_compliance,
    validate_source_metadata,
)
from tools.run_robustness_study import source_provenance


@dataclass(frozen=True)
class FiniteDifferenceResult:
    derivative: float
    stencil: str
    sample_points: tuple[float, ...]
    sample_values: tuple[float, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, default=2, choices=(1, 2, 3, 4))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage", default="geometry",
        choices=(
            "initial_sizing", "adaptive", "geometry", "rationalized",
            "further_rationalized",
        ),
    )
    parser.add_argument(
        "--verification-space", choices=("reduced", "full", "both"),
        default="reduced",
        help="Variable space to verify (default: production reduced space).",
    )
    parser.add_argument(
        "--thickness-steps", type=float, nargs="+",
        default=(1.0e-2, 3.0e-3, 1.0e-3),
        help="Relative thickness perturbations.",
    )
    parser.add_argument(
        "--endpoint-steps", type=float, nargs="+",
        default=(1.0e-2, 3.0e-3, 1.0e-3),
        help="Endpoint perturbations as fractions of one shell cell.",
    )
    parser.add_argument(
        "--rib-indices", type=int, nargs="+", default=None,
        help=(
            "Optional one-based rib selection.  Reduced variables are retained "
            "when their mirror group/coordinate relation touches a selected rib."
        ),
    )
    parser.add_argument("--source-compliance-tolerance", type=float, default=1.0e-6)
    return parser


def relative_error(analytical: float, numerical: float) -> float:
    scale = max(abs(float(analytical)), abs(float(numerical)), 1.0e-12)
    return abs(float(analytical)-float(numerical))/scale


def finite_difference(
    base_value: float,
    lower: float,
    upper: float,
    nominal: float,
    requested_delta: float,
    evaluate,
) -> FiniteDifferenceResult:
    """Return a second-order derivative at ``nominal`` on bounded samples.

    Interior clipped samples use the exact three-point unequal-grid derivative
    at the nominal point.  At a bound, two equally spaced points on the
    available side supply a conventional three-point one-sided derivative.
    """
    x0 = float(nominal)
    lower = float(lower)
    upper = float(upper)
    delta = float(requested_delta)
    if delta <= 0.0 or lower > x0 or x0 > upper:
        raise ValueError("invalid finite-difference point, bounds, or step")
    tolerance = 1.0e-14*max(abs(upper-lower), 1.0)
    hm = min(delta, x0-lower)
    hp = min(delta, upper-x0)
    f0 = float(base_value)
    if hm > tolerance and hp > tolerance:
        xm, xp = x0-hm, x0+hp
        fm, fp = float(evaluate(xm)), float(evaluate(xp))
        derivative = (
            -hp/(hm*(hm+hp))*fm
            + (hp-hm)/(hm*hp)*f0
            + hm/(hp*(hm+hp))*fp
        )
        stencil = (
            "three_point_centered_equal"
            if np.isclose(hm, hp, rtol=1.0e-12, atol=tolerance)
            else "three_point_unequal"
        )
        return FiniteDifferenceResult(
            float(derivative), stencil, (xm, x0, xp), (fm, f0, fp)
        )
    if upper-x0 > 2.0*tolerance:
        h = min(delta, 0.5*(upper-x0))
        if h <= tolerance:
            raise ValueError("forward finite-difference stencil collapsed")
        x1, x2 = x0+h, x0+2.0*h
        f1, f2 = float(evaluate(x1)), float(evaluate(x2))
        return FiniteDifferenceResult(
            float((-3.0*f0+4.0*f1-f2)/(2.0*h)),
            "three_point_forward_one_sided", (x0, x1, x2), (f0, f1, f2),
        )
    if x0-lower > 2.0*tolerance:
        h = min(delta, 0.5*(x0-lower))
        if h <= tolerance:
            raise ValueError("backward finite-difference stencil collapsed")
        x1, x2 = x0-h, x0-2.0*h
        f1, f2 = float(evaluate(x1)), float(evaluate(x2))
        return FiniteDifferenceResult(
            float((3.0*f0-4.0*f1+f2)/(2.0*h)),
            "three_point_backward_one_sided", (x2, x1, x0), (f2, f1, f0),
        )
    raise ValueError("finite-difference stencil has insufficient bounded space")


def trace_signature(model, rib: Rib) -> dict:
    points = np.asarray(model.rib_bottom_points(rib), float)
    midpoints = 0.5*(points[:-1]+points[1:]) if len(points) >= 2 else []
    cells = [
        [
            min(max(int(np.floor(point[0]/model.dx)), 0), model.nx-1),
            min(max(int(np.floor(point[1]/model.dy)), 0), model.ny-1),
        ]
        for point in midpoints
    ]
    return {"point_count": int(len(points)), "segment_cells": cells}


def moved_rib(rib: Rib, coordinate: int, value: float) -> Rib:
    points = np.asarray([rib.p0, rib.p1], float)
    points[coordinate//2, coordinate % 2] = float(value)
    return Rib(tuple(points[0]), tuple(points[1]), rib.height, rib.name, rib.segments)


def ribs_from_coordinates(ribs: list[Rib], coordinates: np.ndarray) -> list[Rib]:
    points = np.asarray(coordinates, float).reshape(len(ribs), 4)
    return [
        Rib(tuple(point[:2]), tuple(point[2:]), rib.height, rib.name, rib.segments)
        for point, rib in zip(points, ribs)
    ]


def validate_trial_layout(base_ribs: list[Rib], trial_ribs: list[Rib], cfg: dict) -> None:
    reasons = geometry_move_freeze_reasons(
        base_ribs, trial_ribs, 0.25*float(cfg["initial_rib_cell_size"])
    )
    if reasons:
        readable = {
            base_ribs[index].name: sorted(values)
            for index, values in reasons.items()
        }
        raise ValueError(f"invalid geometry perturbation: {readable}")


def trace_change_detail(
    model,
    base_ribs: list[Rib],
    sampled_layouts: list[tuple[float, list[Rib]]],
    affected_indices: list[int],
) -> tuple[bool, str, list[dict]]:
    """Compare every sampled trace with the nominal trace for affected ribs."""
    details: list[dict] = []
    changed = False
    for index in sorted(set(affected_indices)):
        base = trace_signature(model, base_ribs[index])
        samples = []
        for point, layout in sampled_layouts:
            signature = trace_signature(model, layout[index])
            differs = signature != base
            changed = changed or differs
            samples.append({
                "variable_value": float(point),
                "signature": signature,
                "differs_from_base": differs,
            })
        details.append({
            "rib_index": index+1,
            "rib_name": base_ribs[index].name,
            "base_signature": base,
            "samples": samples,
        })
    return (
        changed,
        "root_trace_changed_from_nominal" if changed else "trace_preserved",
        details,
    )


def selected_reduced_variables(variable_map, selected_ribs: set[int]) -> tuple[list[int], list[int]]:
    thickness_variables = [
        variable for variable, group in enumerate(variable_map.rib_groups)
        if selected_ribs.intersection(group)
    ]
    coordinate_variables = []
    for variable in range(variable_map.coordinate_count):
        nodes = np.flatnonzero(variable_map.coordinate_variable == variable)
        if any(int(node)//4 in selected_ribs for node in nodes):
            coordinate_variables.append(variable)
    return thickness_variables, coordinate_variables


def reduced_gradient_data(
    variable_map,
    full_thickness_gradient: np.ndarray,
    full_coordinate_gradient: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        variable_map.reduce_thickness_gradient(full_thickness_gradient),
        variable_map.reduce_coordinate_gradient(full_coordinate_gradient.ravel()),
    )


def invalid_record(base: dict, exc: Exception) -> dict:
    return {
        **base,
        "process_status": "invalid_perturbation",
        "valid": False,
        "analytical": base.get("analytical"),
        "finite_difference": None,
        "absolute_error": None,
        "relative_error": None,
        "nonsmooth_trace_change": None,
        "trace_change_reason": "invalid_or_degenerate_perturbation",
        "error": f"{type(exc).__name__}: {exc}",
    }


def successful_record(
    base: dict,
    analytical: float,
    result: FiniteDifferenceResult,
) -> dict:
    return {
        **base,
        "process_status": "complete",
        "valid": True,
        "stencil": result.stencil,
        "sample_points": list(result.sample_points),
        "sample_compliances": list(result.sample_values),
        "analytical": float(analytical),
        "finite_difference": float(result.derivative),
        "absolute_error": abs(float(analytical)-float(result.derivative)),
        "relative_error": relative_error(analytical, result.derivative),
    }


def error_summary(records: list[dict], variable_space: str, derivative: str) -> dict:
    selected = [
        record for record in records
        if record["variable_space"] == variable_space
        and record["derivative"] == derivative
    ]
    valid = [record for record in selected if record.get("valid")]
    smooth = [
        record for record in valid if record.get("nonsmooth_trace_change") is False
    ]
    return {
        "record_count": len(selected),
        "valid_record_count": len(valid),
        "invalid_record_count": len(selected)-len(valid),
        "smooth_record_count": len(smooth),
        "nonsmooth_record_count": sum(
            record.get("nonsmooth_trace_change") is True for record in valid
        ),
        "maximum_relative_error_smooth": max(
            (record["relative_error"] for record in smooth), default=None
        ),
        "median_relative_error_smooth": float(np.median([
            record["relative_error"] for record in smooth
        ])) if smooth else None,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    created_utc = datetime.now(timezone.utc).isoformat()
    if any(step <= 0.0 for step in (*args.thickness_steps, *args.endpoint_steps)):
        raise ValueError("all finite-difference steps must be positive")
    if args.source_compliance_tolerance < 0.0:
        raise ValueError("source compliance tolerance must be nonnegative")

    source = json.loads(args.source.read_text(encoding="utf-8"))
    cfg = load_case(args.case, quick=False)
    validate_source_metadata(source, args.case, cfg)
    ribs, thicknesses, stage = stage_design(
        source, args.stage, int(cfg["rib"]["segments"])
    )
    selected = (
        set(range(len(ribs))) if args.rib_indices is None
        else {value-1 for value in args.rib_indices}
    )
    if not selected or any(index < 0 or index >= len(ribs) for index in selected):
        raise ValueError("rib indices must select at least one saved-stage rib")

    model = build_model(cfg)
    started = time.perf_counter()
    base_result = model.analyze(ribs, thicknesses)
    mismatch = validate_saved_compliance(
        args.stage, float(stage["compliance"]), float(base_result.compliance),
        args.source_compliance_tolerance,
    )
    full_t_gradient = np.asarray(
        model.compliance_gradient(ribs, thicknesses, base_result), float
    )
    full_p_gradient = np.asarray(
        model.geometry_gradient(
            ribs, thicknesses, base_result,
            float(cfg["algorithm"]["geometry_fd_fraction"])
            * float(cfg["initial_rib_cell_size"]),
        ), float
    )
    variable_map = build_mirror_variable_map(
        ribs, mirror_axes(cfg), *map(float, cfg["domain"])
    )
    reduced_t_gradient, reduced_p_gradient = reduced_gradient_data(
        variable_map, full_t_gradient, full_p_gradient
    )
    records: list[dict] = []
    fea_count = 1
    t_lower, t_upper = float(cfg["rib"]["lower"]), float(cfg["rib"]["upper"])
    thickness_scale = max(float(cfg["rib"]["initial"]), 1.0e-12)
    coordinates = np.asarray([[*rib.p0, *rib.p1] for rib in ribs], float).ravel()

    def run_component(
        base_record: dict,
        analytical: float,
        lower: float,
        upper: float,
        nominal: float,
        delta: float,
        evaluate,
        trace_layouts: dict[float, list[Rib]] | None = None,
        affected: list[int] | None = None,
    ) -> None:
        try:
            result = finite_difference(
                float(base_result.compliance), lower, upper, nominal, delta, evaluate
            )
            record = successful_record(base_record, analytical, result)
            if trace_layouts is None:
                record.update({
                    "nonsmooth_trace_change": False,
                    "trace_change_reason": "not_applicable",
                    "affected_trace_detail": [],
                })
            else:
                sampled = [
                    (point, trace_layouts[point])
                    for point in result.sample_points if point != nominal
                ]
                nonsmooth, reason, detail = trace_change_detail(
                    model, ribs, sampled, affected or []
                )
                record.update({
                    "nonsmooth_trace_change": nonsmooth,
                    "trace_change_reason": reason,
                    "affected_trace_detail": detail,
                })
        except Exception as exc:
            record = invalid_record(base_record, exc)
        records.append(record)

    if args.verification_space in {"reduced", "both"}:
        reduced_t = variable_map.reduce_thicknesses(thicknesses)
        reduced_p = variable_map.reduce_coordinates(coordinates)
        full_bounds = np.tile(
            np.asarray([[0.0, cfg["domain"][0]], [0.0, cfg["domain"][1]]], float),
            (2*len(ribs), 1),
        )
        reduced_bounds = variable_map.reduce_coordinate_bounds(full_bounds)
        full_scales = np.tile(np.asarray([model.dx, model.dy], float), 2*len(ribs))
        reduced_scales = variable_map.reduce_coordinate_scale(full_scales)
        thickness_variables, coordinate_variables = selected_reduced_variables(
            variable_map, selected
        )
        for variable in thickness_variables:
            group = variable_map.rib_groups[variable]
            nominal = float(reduced_t[variable])
            for step in args.thickness_steps:
                delta = float(step)*max(abs(nominal), thickness_scale)

                def evaluate(value: float, variable=variable) -> float:
                    nonlocal fea_count
                    trial_reduced = reduced_t.copy()
                    trial_reduced[variable] = value
                    trial_t = variable_map.expand_thicknesses(trial_reduced)
                    fea_count += 1
                    return float(model.analyze(ribs, trial_t).compliance)

                run_component({
                    "variable_space": "reduced",
                    "derivative": "thickness",
                    "variable_index": variable+1,
                    "affected_rib_indices": [index+1 for index in group],
                    "affected_rib_names": [ribs[index].name for index in group],
                    "step_fraction": float(step),
                    "requested_delta": delta,
                }, float(reduced_t_gradient[variable]), t_lower, t_upper,
                    nominal, delta, evaluate)

        for variable in coordinate_variables:
            nodes = np.flatnonzero(variable_map.coordinate_variable == variable)
            affected = sorted({int(node)//4 for node in nodes})
            relations = [
                {
                    "rib_index": int(node)//4+1,
                    "rib_name": ribs[int(node)//4].name,
                    "component": ("p0_x", "p0_y", "p1_x", "p1_y")[int(node)%4],
                    "sign": float(variable_map.coordinate_sign[node]),
                    "offset": float(variable_map.coordinate_offset[node]),
                }
                for node in nodes
            ]
            nominal = float(reduced_p[variable])
            for step in args.endpoint_steps:
                delta = float(step)*float(reduced_scales[variable])
                trial_layouts: dict[float, list[Rib]] = {}

                def evaluate(value: float, variable=variable) -> float:
                    nonlocal fea_count
                    trial_reduced = reduced_p.copy()
                    trial_reduced[variable] = value
                    trial_ribs = ribs_from_coordinates(
                        ribs, variable_map.expand_coordinates(trial_reduced)
                    )
                    validate_trial_layout(ribs, trial_ribs, cfg)
                    trial_layouts[value] = trial_ribs
                    fea_count += 1
                    return float(model.analyze(trial_ribs, thicknesses).compliance)

                run_component({
                    "variable_space": "reduced",
                    "derivative": "endpoint",
                    "variable_index": variable+1,
                    "affected_rib_indices": [index+1 for index in affected],
                    "affected_rib_names": [ribs[index].name for index in affected],
                    "coordinate_relations": relations,
                    "step_fraction": float(step),
                    "requested_delta": delta,
                }, float(reduced_p_gradient[variable]),
                    float(reduced_bounds[variable, 0]),
                    float(reduced_bounds[variable, 1]), nominal, delta, evaluate,
                    trial_layouts, affected)

    if args.verification_space in {"full", "both"}:
        for index in sorted(selected):
            for step in args.thickness_steps:
                nominal = float(thicknesses[index])
                delta = float(step)*max(abs(nominal), thickness_scale)

                def evaluate(value: float, index=index) -> float:
                    nonlocal fea_count
                    trial_t = thicknesses.copy()
                    trial_t[index] = value
                    fea_count += 1
                    return float(model.analyze(ribs, trial_t).compliance)

                run_component({
                    "variable_space": "full",
                    "derivative": "thickness",
                    "rib_index": index+1,
                    "rib_name": ribs[index].name,
                    "component": "thickness",
                    "step_fraction": float(step),
                    "requested_delta": delta,
                }, float(full_t_gradient[index]), t_lower, t_upper,
                    nominal, delta, evaluate)

            for coordinate, component in enumerate(
                ("p0_x", "p0_y", "p1_x", "p1_y")
            ):
                axis = coordinate % 2
                nominal = float(coordinates[4*index+coordinate])
                for step in args.endpoint_steps:
                    delta = float(step)*float((model.dx, model.dy)[axis])
                    trial_layouts: dict[float, list[Rib]] = {}

                    def evaluate(value: float, index=index, coordinate=coordinate) -> float:
                        nonlocal fea_count
                        trial_ribs = list(ribs)
                        trial_ribs[index] = moved_rib(ribs[index], coordinate, value)
                        validate_trial_layout(ribs, trial_ribs, cfg)
                        trial_layouts[value] = trial_ribs
                        fea_count += 1
                        return float(model.analyze(trial_ribs, thicknesses).compliance)

                    run_component({
                        "variable_space": "full",
                        "derivative": "endpoint",
                        "rib_index": index+1,
                        "rib_name": ribs[index].name,
                        "component": component,
                        "step_fraction": float(step),
                        "requested_delta": delta,
                    }, float(full_p_gradient[index, coordinate]), 0.0,
                        float(cfg["domain"][axis]), nominal, delta, evaluate,
                        trial_layouts, [index])

    args.output.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for record in records for key in record})
    with (args.output/"sensitivity_components.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                key: json.dumps(value, separators=(",", ":"))
                if isinstance(value, (dict, list)) else value
                for key, value in record.items()
            })
    scopes = (
        ["reduced"] if args.verification_space == "reduced"
        else ["full"] if args.verification_space == "full"
        else ["reduced", "full"]
    )
    payload = {
        "case": args.case,
        "started_utc": created_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "process_status": "complete",
        "executable_provenance": source_provenance(
            Path(__file__).resolve().parents[1], Path(__file__)
        ),
        "config": cfg,
        "source": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "source_stage": args.stage,
        "mesh": list(map(int, cfg["mesh"])),
        "saved_compliance": float(stage["compliance"]),
        "reanalyzed_compliance": float(base_result.compliance),
        "source_relative_mismatch": mismatch,
        "verification_space": args.verification_space,
        "rib_indices": [index+1 for index in sorted(selected)],
        "thickness_steps": list(map(float, args.thickness_steps)),
        "endpoint_steps": list(map(float, args.endpoint_steps)),
        "finite_element_analysis_count": fea_count,
        "elapsed_seconds": time.perf_counter()-started,
        "interpretation": (
            "Reduced-space records verify the exact mirror-variable reduction, "
            "expansion, and gradient summation used by production optimization. "
            "Invalid geometry and nominal-to-sample mesh-trace changes are "
            "retained but excluded from smooth derivative-error summaries."
        ),
        "summary": {
            scope: {
                derivative: error_summary(records, scope, derivative)
                for derivative in ("thickness", "endpoint")
            }
            for scope in scopes
        },
        "components": records,
    }
    (args.output/"sensitivity_verification.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    invalid_count = sum(not record.get("valid", False) for record in records)
    print(
        f"Case {args.case} {args.stage}: components={len(records)}, "
        f"FEA={fea_count}, invalid={invalid_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
