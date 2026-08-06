from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from rib_layout_algorithms.model import Rib
from rib_layout_core import load_case
from rib_layout_serialization import portable_artifact_path, strict_json_dumps
from rib_layout_algorithms.symmetry import (
    build_mirror_variable_map,
    missing_mirror_partners,
    mirror_axes,
)
from tools.run_full_pool_baseline import (
    build_parser as full_pool_parser,
    main as full_pool_main,
    rationalization_termination_reason,
    sizing_termination_reason,
    stage_metrics,
)
from tools.run_geometry_restart_diagnostic import (
    LEGACY_HISTORY_FIELDS,
    build_parser as restart_parser,
    compliance_distribution,
    failed_restart_record,
    legacy_single_restart_payload,
    perturbed_ribs,
    restart_initial_volume,
    stage_design,
    validate_saved_compliance as restart_validate_saved_compliance,
    validate_source_metadata as restart_validate_source_metadata,
    write_legacy_history,
)
from tools.run_robustness_study import (
    build_parser as robustness_parser,
    build_study_specs,
    fixed_layout_mesh_reanalysis,
    main as robustness_main,
    source_provenance,
    mesh_value,
    starting_layout_variant,
    validate_common_reanalysis_mesh,
)
from tools.run_sensitivity_verification import (
    build_parser as sensitivity_parser,
    finite_difference,
    reduced_gradient_data,
    relative_error,
    trace_change_detail,
)
from tools.run_topology_lifting_diagnostic import (
    build_parser as lifting_parser,
    validate_saved_compliance,
    validate_source_metadata,
    lifted_design,
)
from verification.run_response_only_160x80 import script_identity


class DiagnosticToolTests(unittest.TestCase):
    def test_response_only_identity_uses_its_own_entry_script(self):
        code_root = Path(__file__).resolve().parents[1]
        captured = {}

        def fake_provenance(root, entry_tool):
            captured["root"] = root
            captured["entry_tool"] = entry_tool
            return {"entry_tool_sha256": hashlib.sha256(
                entry_tool.read_bytes()
            ).hexdigest()}

        identity = script_identity(
            code_root, fake_provenance, portable_artifact_path
        )
        expected = code_root / "verification" / "run_response_only_160x80.py"
        self.assertEqual(captured["root"], code_root)
        self.assertEqual(captured["entry_tool"], expected)
        self.assertEqual(
            identity["executable_provenance"]["entry_tool"],
            "verification/run_response_only_160x80.py",
        )
        self.assertEqual(
            identity["script_sha256"],
            identity["executable_provenance"]["entry_tool_sha256"],
        )
        self.assertNotIn(
            "run_robustness_study.py",
            identity["executable_provenance"]["entry_tool"],
        )

    def test_strict_json_and_absolute_source_path_normalization(self):
        code_root = Path(__file__).resolve().parents[1]
        absolute_source = code_root / "results" / "example_2" / "results.json"
        self.assertEqual(
            portable_artifact_path(absolute_source, code_root),
            "results/example_2/results.json",
        )
        self.assertEqual(strict_json_dumps({"ratio": 0.5}), '{"ratio": 0.5}')
        with self.assertRaisesRegex(ValueError, "Out of range float values"):
            strict_json_dumps({"ratio": float("nan")})

    def test_restart_parser_defaults_are_deterministic(self):
        args = restart_parser().parse_args([
            "--case", "3", "--source", "results.json", "--output", "audit",
        ])
        self.assertEqual(args.restarts, 1)
        self.assertEqual(args.multistarts, 0)
        self.assertEqual(args.seed, 20260803)
        self.assertAlmostEqual(args.thickness_perturbation, 0.10)
        self.assertEqual(args.endpoint_perturbation, 0.0)

    def test_restart_parser_accepts_rationalized_stages_and_absence_is_clear(self):
        args = restart_parser().parse_args([
            "--case", "4", "--source", "results.json", "--output", "audit",
            "--stage", "further_rationalized",
        ])
        self.assertEqual(args.stage, "further_rationalized")
        with self.assertRaisesRegex(ValueError, "absent"):
            stage_design({"stages": [{"name": "geometry"}]}, "rationalized", 10)

    def test_endpoint_perturbation_preserves_symmetry_and_domain_bounds(self):
        ribs = [
            Rib((0.0, 0.0), (1.0, 1.0), 2.0, "A"),
            Rib((4.0, 0.0), (3.0, 1.0), 2.0, "B"),
        ]
        optimizer = SimpleNamespace(
            cfg={"domain": [4.0, 2.0], "mirror_symmetry": ["x"]},
            model=SimpleNamespace(dx=0.5, dy=0.5),
        )
        moved = perturbed_ribs(
            optimizer, ribs, np.random.default_rng(1234), 0.25
        )
        coordinates = np.asarray([[*rib.p0, *rib.p1] for rib in moved])
        self.assertTrue(np.all(coordinates[:, (0, 2)] >= 0.0))
        self.assertTrue(np.all(coordinates[:, (0, 2)] <= 4.0))
        self.assertTrue(np.all(coordinates[:, (1, 3)] >= 0.0))
        self.assertTrue(np.all(coordinates[:, (1, 3)] <= 2.0))
        self.assertEqual(
            missing_mirror_partners(moved, ("x",), 4.0, 2.0), []
        )

    def test_restart_distribution_reports_best_median_worst_and_feasible(self):
        records = [
            {"run": 1, "kind": "restart", "final_compliance": 3.0, "analysis_count": 4, "termination_reason": "a", "final_feasible": True},
            {"run": 2, "kind": "multistart", "final_compliance": 1.0, "analysis_count": 5, "termination_reason": "b", "final_feasible": False},
            {"run": 3, "kind": "multistart", "final_compliance": 2.0, "analysis_count": 6, "termination_reason": "c", "final_feasible": True},
        ]
        summary = compliance_distribution(records)
        self.assertEqual(summary["best"]["run"], 2)
        self.assertEqual(summary["median"]["run"], 3)
        self.assertEqual(summary["worst"]["run"], 1)
        self.assertEqual(summary["best_feasible"]["run"], 3)

    def test_restart_validates_source_reanalysis_and_perturbed_volume(self):
        cfg = {"mesh": [40, 20]}
        source = {"example": 2, "quick": False, "mesh_used": [40, 20]}
        restart_validate_source_metadata(source, 2, cfg)
        with self.assertRaisesRegex(ValueError, "case mismatch"):
            restart_validate_source_metadata({**source, "example": 3}, 2, cfg)
        with self.assertRaisesRegex(ValueError, "mesh mismatch"):
            restart_validate_source_metadata(
                {**source, "mesh_used": [20, 10]}, 2, cfg
            )
        self.assertAlmostEqual(
            restart_validate_saved_compliance("geometry", 10.0, 10.000001, 1e-6),
            1e-7,
        )
        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            restart_validate_saved_compliance("geometry", 10.0, 10.1, 1e-6)

        optimizer = SimpleNamespace(
            volume=lambda ribs, thicknesses: sum(
                rib.length*float(thickness)
                for rib, thickness in zip(ribs, thicknesses)
            )
        )
        saved = [Rib((0.0, 0.0), (1.0, 0.0), 2.0, "A")]
        perturbed = [Rib((0.0, 0.0), (2.0, 0.0), 2.0, "A")]
        self.assertEqual(
            restart_initial_volume(optimizer, perturbed, np.array([0.5])), 1.0
        )
        self.assertNotEqual(
            restart_initial_volume(optimizer, saved, np.array([0.5])), 1.0
        )

    def test_failed_restart_record_separates_process_and_convergence(self):
        try:
            raise RuntimeError("deliberate")
        except RuntimeError as exc:
            record = failed_restart_record(2, "multistart", "start", 0.0, exc)
        self.assertEqual(record["process_status"], "failed")
        self.assertFalse(record["optimizer_converged"])
        self.assertIn("RuntimeError: deliberate", record["error"])
        self.assertIn("Traceback", record["traceback"])

    def test_full_pool_parser_accepts_multiple_cases(self):
        args = full_pool_parser().parse_args([
            "--case", "1", "2", "--output", "audit", "--quick",
        ])
        self.assertEqual(args.case, [1, 2])
        self.assertTrue(args.quick)
        self.assertEqual(args.post_sizing_policy, "sizing-only")

    def test_full_pool_parser_and_stage_metrics_support_optional_policy(self):
        args = full_pool_parser().parse_args([
            "--case", "2", "--output", "audit",
            "--post-sizing-policy", "rationalization",
            "--rationalization-relaxation", "0.075",
        ])
        self.assertEqual(args.post_sizing_policy, "rationalization")
        optimizer = SimpleNamespace(volume=lambda ribs, thicknesses: 0.4)
        result = SimpleNamespace(compliance=2.5)
        metrics = stage_metrics(
            "sizing", [Rib((0, 0), (1, 0), 2, "A")], np.array([0.2]),
            result, optimizer, 3, 1.25,
        )
        self.assertEqual(metrics["rib_count"], 1)
        self.assertEqual(metrics["analyses"], 3)
        self.assertEqual(metrics["phase_process_status"], "complete")
        self.assertIsNone(metrics["phase_converged"])

    def test_full_pool_phase_termination_is_unambiguous(self):
        self.assertEqual(
            sizing_termination_reason(SimpleNamespace(
                log=["sizing SCA converged: outer=2"]
            )),
            "converged",
        )
        self.assertEqual(
            sizing_termination_reason(SimpleNamespace(
                log=["sizing SCA warning: outer iteration limit 3 reached"]
            )),
            "iteration_limit",
        )
        accepted = [{"event": "post_filter_geometry", "accepted": True}]
        self.assertEqual(
            rationalization_termination_reason(accepted, 0.05, 8, 5),
            "accepted_rib_deleted_design",
        )
        failed = [
            {"event": "filtering_attempt"},
            {"event": "post_filter_geometry", "accepted": False},
        ]
        self.assertEqual(
            rationalization_termination_reason(failed, 0.05, 8, 8),
            "restored_input_no_accepted_deletion",
        )

    def test_full_pool_persists_failed_cases_and_continues(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with (
                mock.patch(
                    "tools.run_full_pool_baseline.source_provenance",
                    return_value={"source_aggregate_sha256": "test"},
                ),
                mock.patch(
                    "tools.run_full_pool_baseline.load_case", return_value={}
                ),
                mock.patch(
                    "tools.run_full_pool_baseline.run_case",
                    side_effect=RuntimeError("deliberate"),
                ),
            ):
                exit_code = full_pool_main([
                    "--case", "1", "2", "--output", str(output)
                ])
            self.assertEqual(exit_code, 2)
            aggregate = json.loads(
                (output/"full_pool_results.json").read_text(encoding="utf-8")
            )
            self.assertEqual(aggregate["process_status"], "complete_with_failures")
            self.assertEqual(len(aggregate["cases"]), 2)
            self.assertTrue((output/"full_pool_case_1.json").is_file())
            self.assertTrue((output/"full_pool_case_2.json").is_file())
            self.assertTrue(all(
                record["process_status"] == "failed"
                for record in aggregate["cases"]
            ))

    def test_sensitivity_parser_and_bounded_finite_difference(self):
        args = sensitivity_parser().parse_args([
            "--source", "results.json", "--output", "audit",
        ])
        self.assertEqual(args.case, 2)
        self.assertEqual(args.stage, "geometry")
        result = finite_difference(
            1.0, 0.0, 2.0, 1.0, 0.1, lambda value: value*value
        )
        self.assertAlmostEqual(result.derivative, 2.0)
        self.assertEqual(result.stencil, "three_point_centered_equal")
        self.assertAlmostEqual(relative_error(2.0, result.derivative), 0.0)
        unequal = finite_difference(
            0.05**2, 0.0, 2.0, 0.05, 0.1, lambda value: value*value
        )
        self.assertAlmostEqual(unequal.derivative, 0.1)
        self.assertEqual(unequal.stencil, "three_point_unequal")
        boundary = finite_difference(
            0.0, 0.0, 2.0, 0.0, 0.1, lambda value: value*value
        )
        self.assertAlmostEqual(boundary.derivative, 0.0)
        self.assertEqual(boundary.stencil, "three_point_forward_one_sided")

    def test_reduced_mirror_thickness_and_endpoint_gradients(self):
        ribs = [
            Rib((0.0, 0.0), (1.0, 1.0), 2.0, "A"),
            Rib((4.0, 0.0), (3.0, 1.0), 2.0, "B"),
        ]
        variable_map = build_mirror_variable_map(ribs, ("x",), 4.0, 2.0)
        coordinates = np.asarray([[*rib.p0, *rib.p1] for rib in ribs]).ravel()
        reduced_coordinates = variable_map.reduce_coordinates(coordinates)
        full_coordinate_gradient = np.arange(1.0, 9.0)
        reduced_t, reduced_p = reduced_gradient_data(
            variable_map, np.array([2.0, 3.0]), full_coordinate_gradient
        )
        self.assertTrue(np.allclose(reduced_t, [5.0]))
        for variable in range(variable_map.coordinate_count):
            step = 1e-6
            plus = reduced_coordinates.copy()
            minus = reduced_coordinates.copy()
            plus[variable] += step
            minus[variable] -= step
            numerical = (
                full_coordinate_gradient
                @ variable_map.expand_coordinates(plus)
                - full_coordinate_gradient
                @ variable_map.expand_coordinates(minus)
            )/(2*step)
            self.assertAlmostEqual(reduced_p[variable], numerical, places=7)

    def test_trace_change_detection_compares_samples_with_nominal(self):
        class TraceModel:
            dx = dy = 1.0
            nx = ny = 4

            @staticmethod
            def rib_bottom_points(rib):
                return np.asarray([rib.p0, rib.p1], float)

        base = [Rib((0.0, 0.0), (1.0, 0.0), 2.0, "A")]
        preserved = [Rib((0.1, 0.0), (1.1, 0.0), 2.0, "A")]
        changed = [Rib((1.1, 0.0), (2.1, 0.0), 2.0, "A")]
        nonsmooth, reason, detail = trace_change_detail(
            TraceModel(), base, [(0.1, preserved), (1.1, changed)], [0]
        )
        self.assertTrue(nonsmooth)
        self.assertEqual(reason, "root_trace_changed_from_nominal")
        self.assertFalse(detail[0]["samples"][0]["differs_from_base"])
        self.assertTrue(detail[0]["samples"][1]["differs_from_base"])

    def test_robustness_specs_are_one_factor_and_start_layouts_are_closed(self):
        args = robustness_parser().parse_args([
            "--output", "audit", "--studies", "mesh", "pool", "starts",
        ])
        cfg = load_case(2)
        specs = build_study_specs(cfg, args)
        self.assertEqual(len(specs), 9)
        self.assertEqual(mesh_value("20x10"), (20, 10))
        mesh_spec = next(spec for spec in specs if spec["factor"] == "mesh")
        self.assertEqual(mesh_spec["config"]["algorithm"], cfg["algorithm"])
        for name in ("all_orbits", "even_orbits", "odd_orbits"):
            ribs = starting_layout_variant(cfg, name)
            self.assertTrue(ribs)
            self.assertEqual(
                missing_mirror_partners(
                    ribs, mirror_axes(cfg), *map(float, cfg["domain"])
                ),
                [],
            )

    def test_common_reanalysis_mesh_validation_and_tool_provenance(self):
        specs = [
            {"config": {"mesh": [20, 10]}},
            {"config": {"mesh": [40, 20]}},
        ]
        validate_common_reanalysis_mesh((40, 20), specs)
        with self.assertRaisesRegex(ValueError, "at least as fine"):
            validate_common_reanalysis_mesh((39, 20), specs)
        code_root = Path(__file__).resolve().parents[1]
        provenance = source_provenance(
            code_root, code_root/"tools"/"run_robustness_study.py"
        )
        expected = sorted(
            path.relative_to(code_root).as_posix()
            for path in (code_root/"tools").rglob("*.py")
        )
        self.assertEqual(provenance["tool_files"], expected)
        self.assertEqual(provenance["tool_file_count"], len(expected))
        self.assertEqual(len(provenance["tools_aggregate_sha256"]), 64)
        self.assertIsInstance(provenance["git_dirty"], bool)

    def test_fixed_layout_mesh_study_holds_the_design_fixed(self):
        stage = {"ribs": [{"name": "A"}]}
        with mock.patch(
            "tools.run_robustness_study.reanalyze_on_mesh",
            side_effect=lambda same_stage, cfg, mesh: {
                "mesh": list(mesh),
                "same_stage_object": same_stage is stage,
            },
        ):
            result = fixed_layout_mesh_reanalysis(
                stage, {"mesh": [20, 10]}, [(20, 10), (40, 20)]
            )
        self.assertTrue(result["optimization_held_fixed"])
        self.assertEqual(
            [record["mesh"] for record in result["meshes"]],
            [[20, 10], [40, 20]],
        )
        self.assertTrue(all(
            record["same_stage_object"] for record in result["meshes"]
        ))

    def test_robustness_total_timing_includes_common_reanalysis(self):
        stage = SimpleNamespace(
            name="geometry",
            ribs=[Rib((0.0, 0.0), (1.0, 0.0), 2.0, "A")],
            thicknesses=np.array([0.2]),
            compliance=1.0,
            analyses=2,
            note="",
        )

        class FakeOptimizer:
            def __init__(self, model, cfg):
                self.model = model
                self.cfg = cfg
                self.analysis_count = 2
                self.geometry_termination_reason = "converged"
                self.rationalization_histories = {}
                self.log = []

            @staticmethod
            def volume(ribs, thicknesses):
                return 0.4

            @staticmethod
            def short_rib_length_threshold():
                return 0.25

            @staticmethod
            def run(initial, candidates):
                return SimpleNamespace(stages=[stage], rationalization_histories={})

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with (
                mock.patch(
                    "tools.run_robustness_study.RibLayoutOptimizer", FakeOptimizer
                ),
                mock.patch(
                    "tools.run_robustness_study.build_model",
                    return_value=SimpleNamespace(dx=1.0, dy=1.0),
                ),
                mock.patch(
                    "tools.run_robustness_study.starting_layout_variant",
                    return_value=stage.ribs,
                ),
                mock.patch(
                    "tools.run_robustness_study.candidate_ribs", return_value=[]
                ),
                mock.patch(
                    "tools.run_robustness_study.reanalyze_on_mesh",
                    return_value={
                        "process_status": "complete",
                        "mesh": [40, 20],
                        "compliance": 1.1,
                    },
                ),
                mock.patch(
                    "tools.run_robustness_study.source_provenance",
                    return_value={"source_aggregate_sha256": "test"},
                ),
                mock.patch(
                    "tools.run_robustness_study.time.perf_counter",
                    side_effect=[10.0, 12.0, 15.0],
                ),
            ):
                exit_code = robustness_main([
                    "--case", "2", "--output", str(output),
                    "--studies", "retention",
                    "--retention-thresholds", "0.7", "--max-runs", "1",
                    "--common-reanalysis-mesh", "40", "20",
                ])
            self.assertEqual(exit_code, 0)
            payload = json.loads(
                (output/"robustness_study.json").read_text(encoding="utf-8")
            )
        record = payload["runs"][0]
        self.assertEqual(record["optimizer_elapsed_seconds"], 2.0)
        self.assertEqual(record["elapsed_seconds"], 5.0)
        self.assertGreater(
            record["elapsed_seconds"], record["optimizer_elapsed_seconds"]
        )
        self.assertIsNotNone(record["completed_utc"])
        self.assertEqual(record["common_reanalysis"]["process_status"], "complete")

    def test_lifting_reuses_reduced_state_and_adds_deleted_at_lower_bound(self):
        full = [
            Rib((0.0, 0.0), (1.0, 0.0), 2.0, "A"),
            Rib((0.0, 1.0), (1.0, 1.0), 2.0, "B"),
        ]
        moved_a = Rib((0.0, 0.1), (1.0, 0.1), 2.0, "A")
        ribs, thicknesses, names = lifted_design(
            full, np.array([0.2, 0.3]), [moved_a], np.array([0.4]), 1.0e-6
        )
        self.assertEqual([rib.name for rib in ribs], ["A", "B"])
        self.assertEqual(ribs[0], moved_a)
        self.assertTrue(np.allclose(thicknesses, [0.4, 1.0e-6]))
        self.assertEqual(names, ["B"])

    def test_lifting_parser_exposes_stage_selection(self):
        args = lifting_parser().parse_args([
            "--case", "4", "--source", "results.json", "--output", "audit",
            "--reduced-stage", "further_rationalized",
        ])
        self.assertEqual(args.full_stage, "geometry")
        self.assertEqual(args.reduced_stage, "further_rationalized")
        self.assertEqual(args.source_compliance_tolerance, 1.0e-6)

    def test_lifting_rejects_incompatible_metadata_and_compliance(self):
        cfg = {"mesh": [40, 20]}
        source = {"example": 2, "quick": False, "mesh_used": [40, 20]}
        validate_source_metadata(source, 2, cfg)
        with self.assertRaisesRegex(ValueError, "case mismatch"):
            validate_source_metadata({**source, "example": 3}, 2, cfg)
        with self.assertRaisesRegex(ValueError, "mesh mismatch"):
            validate_source_metadata(
                {**source, "mesh_used": [20, 10]}, 2, cfg
            )
        self.assertAlmostEqual(
            validate_saved_compliance("geometry", 10.0, 10.000001, 1.0e-6),
            1.0e-7,
        )
        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            validate_saved_compliance("geometry", 10.0, 10.1, 1.0e-6)

    def test_single_restart_payload_and_csv_retain_legacy_schema(self):
        record = {
            "initial_compliance": 2.0,
            "geometry_analysis_count": 3,
            "outer_records": 2,
            "initial_volume": 1.0,
            "final_volume": 1.0,
            "final_compliance": 1.5,
            "relative_compliance_change": -0.25,
            "initial_ribs": [],
            "final_ribs": [],
            "iteration_history": [],
            "log": [],
        }
        code_root = Path(__file__).resolve().parents[1]
        absolute_source = code_root / "results" / "example_1" / "results.json"
        payload = legacy_single_restart_payload(
            case=1, source=absolute_source, stage_name="geometry",
            mesh=[20, 20], rib_count=1, saved_compliance=2.0, record=record,
        )
        self.assertEqual(payload["source"], "results/example_1/results.json")
        self.assertEqual(payload["restart_compliance"], 2.0)
        self.assertEqual(payload["restart_initialization_fea"], 1)
        self.assertEqual(payload["geometry_outer_fea"], 3)
        self.assertEqual(payload["outer_iterations"], 2)

        rib = Rib((0.0, 0.0), (1.0, 0.0), 2.0, "A")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"legacy.csv"
            write_legacy_history(
                path, [], [rib], np.array([0.2]), 2.0, 0.4, 1.0, 0.001
            )
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
        self.assertEqual(rows[0], [*LEGACY_HISTORY_FIELDS, "t_A"])
        self.assertEqual(rows[1][0], "0")

    def test_diagnostic_help_supports_module_and_direct_invocation(self):
        code_root = Path(__file__).resolve().parents[1]
        modules = (
            "run_geometry_restart_diagnostic",
            "run_topology_lifting_diagnostic",
            "run_full_pool_baseline",
            "run_sensitivity_verification",
            "run_robustness_study",
        )
        for name in modules:
            commands = (
                [sys.executable, "-B", "-m", f"tools.{name}", "--help"],
                [sys.executable, "-B", str(code_root/"tools"/f"{name}.py"), "--help"],
            )
            for command in commands:
                completed = subprocess.run(
                    command, cwd=code_root, capture_output=True, text=True,
                    timeout=30,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("usage:", completed.stdout.lower())


if __name__ == "__main__":
    unittest.main()
