"""Numerical audit of Example-III compliance and rib/ground tied interfaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rib_layout_core import build_model, load_case
from rib_layout_env import PROJECT_ROOT
from rib_layout_serialization import strict_json_dumps
from rib_layout_algorithms.model import Rib


DEFAULT_RESULTS = PROJECT_ROOT / "results" / "example_3"


def stage_design(records: list[dict], segments: int) -> tuple[list[Rib], np.ndarray]:
    ribs = [
        Rib(
            tuple(record["p0"]), tuple(record["p1"]),
            float(record["height"]), record["name"], segments,
        )
        for record in records
    ]
    thicknesses = np.asarray([record["thickness"] for record in records], float)
    return ribs, thicknesses


def expanded_solution(model, ribs: list[Rib], thicknesses: np.ndarray) -> dict:
    matrix, free_dofs = model._expanded_stiffness(ribs, thicknesses)
    load = np.zeros(matrix.shape[0])
    load[:model.ndof] = model.load_vectors[0]
    reduced = matrix[free_dofs][:, free_dofs].tocsc()
    reduced_rhs = load[free_dofs]
    reduced_u = np.asarray(model._solve_global(reduced, reduced_rhs), float).ravel()
    displacement = np.zeros(matrix.shape[0])
    displacement[free_dofs] = reduced_u
    residual = reduced @ reduced_u-reduced_rhs
    compliance = float(load @ displacement)
    energy = float(displacement @ (matrix @ displacement))
    difference = matrix-matrix.T
    symmetry_error = 0.0 if difference.nnz == 0 else float(np.max(np.abs(difference.data)))
    matrix_scale = max(float(np.max(np.abs(matrix.data))), 1.0)
    ground_u = displacement[:model.ndof]
    load_node = model.nearest_node((0.0, 50.0))

    translation_scale = max(float(np.max(np.abs(ground_u.reshape(-1, 6)[:, :3]))), 1.0e-30)
    rotation_scale = max(float(np.max(np.abs(ground_u.reshape(-1, 6)[:, 3:]))), 1.0e-30)
    translation_mismatches: list[float] = []
    rotation_mismatches: list[float] = []
    weight_sum_errors: list[float] = []
    minimum_weights: list[float] = []
    exact_node_tie_errors: list[float] = []
    for rib in ribs:
        bottom = model.rib_bottom_points(rib)
        nodal = [model.response_at(ground_u, point) for point in bottom]
        for point, value in zip(bottom, nodal):
            nodes, weights = model.interpolation(point)
            transformed = sum(
                float(weight)*ground_u[6*int(node):6*int(node)+6]
                for node, weight in zip(nodes, weights)
            )
            exact_node_tie_errors.append(float(np.max(np.abs(value-transformed))))
            weight_sum_errors.append(abs(float(np.sum(weights))-1.0))
            minimum_weights.append(float(np.min(weights)))
        for index in range(len(bottom)-1):
            midpoint = 0.5*(bottom[index]+bottom[index+1])
            slave_midpoint = 0.5*(nodal[index]+nodal[index+1])
            master_midpoint = model.response_at(ground_u, midpoint)
            mismatch = np.abs(slave_midpoint-master_midpoint)
            translation_mismatches.append(float(np.max(mismatch[:3])))
            rotation_mismatches.append(float(np.max(mismatch[3:])))

    return {
        "compliance_fTu": compliance,
        "compliance_uTKu": energy,
        "relative_energy_balance_error": abs(energy-compliance)/max(abs(compliance), 1.0e-30),
        "relative_free_residual": float(np.linalg.norm(residual)/max(np.linalg.norm(reduced_rhs), 1.0e-30)),
        "relative_symmetry_error": symmetry_error/matrix_scale,
        "expanded_dofs": int(matrix.shape[0]),
        "expanded_nnz": int(matrix.nnz),
        "load_node_y_displacement": float(ground_u[6*load_node+1]),
        "max_exact_tied_node_error": max(exact_node_tie_errors, default=0.0),
        "max_interpolation_weight_sum_error": max(weight_sum_errors, default=0.0),
        "minimum_interpolation_weight": min(minimum_weights, default=0.0),
        "max_between_node_translation_mismatch": max(translation_mismatches, default=0.0),
        "max_between_node_rotation_mismatch": max(rotation_mismatches, default=0.0),
        "relative_between_node_translation_mismatch": max(translation_mismatches, default=0.0)/translation_scale,
        "relative_between_node_rotation_mismatch": max(rotation_mismatches, default=0.0)/rotation_scale,
        "ground_displacement": ground_u,
    }


def condensed_compliance(model, ribs: list[Rib], thicknesses: np.ndarray) -> dict:
    matrix = model.stiffness(ribs, thicknesses)
    reduced = matrix[model.free_dofs][:, model.free_dofs].tocsc()
    rhs = model.load_vectors[0][model.free_dofs]
    reduced_u = np.asarray(model._solve_global(reduced, rhs), float).ravel()
    residual = reduced @ reduced_u-rhs
    displacement = np.zeros(model.ndof)
    displacement[model.free_dofs] = reduced_u
    compliance = float(model.load_vectors[0] @ displacement)
    energy = float(displacement @ (matrix @ displacement))
    return {
        "compliance_fTu": compliance,
        "compliance_uTKu": energy,
        "relative_energy_balance_error": abs(energy-compliance)/max(abs(compliance), 1.0e-30),
        "relative_free_residual": float(np.linalg.norm(residual)/max(np.linalg.norm(rhs), 1.0e-30)),
        "condensed_dofs": int(matrix.shape[0]),
        "condensed_nnz": int(matrix.nnz),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Example-III compliance and rib/ground interfaces"
    )
    parser.add_argument(
        "--geometry-results", type=Path,
        default=DEFAULT_RESULTS / "results.json",
    )
    parser.add_argument(
        "--rational-results", type=Path,
        default=DEFAULT_RESULTS / "diagnostic_results.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=DEFAULT_RESULTS / "fea_connection_audit.json",
    )
    args = parser.parse_args()
    source = json.loads(args.geometry_results.read_text(encoding="utf-8"))
    rational = json.loads(args.rational_results.read_text(encoding="utf-8"))
    cfg = load_case(3, quick=False)
    segments = int(cfg["rib"]["segments"])
    geometry_stage = next(stage for stage in source["stages"] if stage["name"] == "geometry")
    designs = {
        "geometry": stage_design(geometry_stage["ribs"], segments),
        "rationalized": stage_design(rational["final_ribs"], segments),
    }

    audit: dict[str, object] = {
        "case": 3,
        "mesh": cfg["mesh"],
        "interface_subdivisions_per_cell": cfg["interface_subdivisions_per_cell"],
        "designs": {},
    }
    for name, (ribs, thicknesses) in designs.items():
        entry = {
            "rib_count": len(ribs),
            "rib_volume": float(sum(rib.length*rib.height*t for rib, t in zip(ribs, thicknesses))),
            "solvers": {},
        }
        for solver_name, threads in (("pardiso", 1), ("superlu", 1)):
            solver_cfg = json.loads(json.dumps(cfg))
            solver_cfg["linear_solver"] = solver_name
            solver_cfg["linear_solver_threads"] = threads
            model = build_model(solver_cfg)
            result = expanded_solution(model, ribs, thicknesses)
            result.pop("ground_displacement")
            entry["solvers"][solver_name] = result
        condensed_cfg = json.loads(json.dumps(cfg))
        condensed_cfg["linear_solver"] = "superlu"
        condensed_cfg["linear_solver_threads"] = 1
        condensed_model = build_model(condensed_cfg)
        entry["condensed_reference"] = condensed_compliance(
            condensed_model, ribs, thicknesses
        )
        entry["expanded_condensed_relative_difference"] = abs(
            entry["solvers"]["superlu"]["compliance_fTu"]
            - entry["condensed_reference"]["compliance_fTu"]
        )/entry["condensed_reference"]["compliance_fTu"]
        audit["designs"][name] = entry

    geometry_c = audit["designs"]["geometry"]["solvers"]["superlu"]["compliance_fTu"]
    rational_c = audit["designs"]["rationalized"]["solvers"]["superlu"]["compliance_fTu"]
    audit["rationalized_compliance_change"] = rational_c/geometry_c-1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(strict_json_dumps(audit, indent=2), encoding="utf-8")
    print(strict_json_dumps(audit, indent=2))


if __name__ == "__main__":
    main()
