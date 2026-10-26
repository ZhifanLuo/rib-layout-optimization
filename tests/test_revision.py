"""Focused revision contracts; all numerical fixtures are bounded unit tests."""

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from rib_layout_core import build_model, initial_ribs, load_case
from rib_layout_algorithms.model import Rib
from rib_layout_algorithms.model_shell import ShellStiffenedPlateModel
from rib_layout_algorithms.optimization import RibLayoutOptimizer, Stage, OptimizationRun
from rib_layout_algorithms.pressure import uniform_q4_pressure
from rib_layout_algorithms.symmetry import mirror_groups, missing_mirror_partners
from tools import run_revision_experiments as revision
from tools.run_sensitivity_verification import build_parser as sensitivity_parser
from tools.run_geometry_restart_diagnostic import validate_source_metadata
from tools.run_robustness_study import source_provenance


def mesh_stub(nx, ny):
    return SimpleNamespace(nx=nx, ny=ny, dx=100/nx, dy=50/ny,
                           ndof=6*(nx+1)*(ny+1), node=lambda x, y: y*(nx+1)+x)


class PressureTests(unittest.TestCase):
    def test_exact_pressure_resultants_moments_and_nodal_shares_under_refinement(self):
        pressure = load_case(5)["load_cases"][0]["pressure"]
        for nx, ny in ((1, 1), (4, 2), (40, 20), (80, 40), (13, 7)):
            with self.subTest(mesh=(nx, ny)):
                model = mesh_stub(nx, ny)
                nodal = uniform_q4_pressure(model, pressure).reshape(-1, 6)
                z = nodal[:, 2]
                x, y = np.meshgrid(np.linspace(0, 100, nx+1), np.linspace(0, 50, ny+1))
                self.assertAlmostEqual(float(z.sum()), -0.5, places=14)
                # Moment about global origin: Mx=sum(y Fz), My=-sum(x Fz).
                self.assertAlmostEqual(float(z@y.ravel()), -12.5, places=12)
                self.assertAlmostEqual(float(-z@x.ravel()), 25.0, places=12)
                self.assertTrue(np.array_equal(nodal[:, [0, 1, 3, 4, 5]], np.zeros((len(z), 5))))
                quarter = -0.0001*model.dx*model.dy/4
                self.assertEqual(z[0], quarter)
                if nx > 1 and ny > 1:
                    self.assertEqual(z[1], 2*quarter)
                    self.assertEqual(z[nx+2], 4*quarter)
                np.testing.assert_array_equal(z.reshape(ny+1, nx+1), z.reshape(ny+1, nx+1)[::-1])

    def test_pressure_config_and_base_clamp(self):
        cfg = load_case(5)
        self.assertEqual(cfg["mesh"], [80, 40])
        self.assertEqual(cfg["wall_thickness"], 0.5)
        self.assertEqual(cfg["volume_bound"], 1500)
        self.assertEqual(cfg["rib"]["initial"], 0.5)
        self.assertEqual(cfg["algorithm"]["active_set_max_iterations"], 10)
        self.assertEqual(cfg["algorithm"]["sizing_max_iterations"], 100)
        self.assertEqual(len(initial_ribs(cfg)), 16)
        self.assertFalse(missing_mirror_partners(initial_ribs(cfg), ("y",), 100, 50))
        cfg["mesh"] = [4, 2]
        model = build_model(cfg)
        self.assertEqual(len(model.fixed_dofs), 6*3)
        self.assertTrue(np.all(model.fixed_dofs < model.ndof))
        np.testing.assert_array_equal(model.load_vectors[0], uniform_q4_pressure(model, cfg["load_cases"][0]["pressure"]))
        self.assertAlmostEqual(float(model.load_vectors[0][2::6].sum()), -0.5)
        self.assertTrue(all(model.node_xy(n)[0] == 100 for n in model.fixed_dofs[::6]//6))

    def test_pressure_superposes_point_force_and_rejects_bad_config(self):
        cfg = load_case(5)
        cfg["mesh"] = [2, 1]
        cfg["load_cases"][0]["forces"] = [{"point": [0, 0], "value": [1, 2, 3]}]
        model = build_model(cfg)
        self.assertAlmostEqual(model.load_vectors[0][2::6].sum(), 2.5)
        for pressure in ({"type": "bad", "value": 1}, {"type": "uniform", "value": np.nan}):
            with self.assertRaises(ValueError):
                uniform_q4_pressure(mesh_stub(1, 1), pressure)


class PolicyTests(unittest.TestCase):
    def optimizer(self, policy="smooth", symmetry=False):
        cfg = load_case(5)
        cfg["mirror_symmetry"] = ["y"] if symmetry else []
        cfg["algorithm"]["rationalization_policy"] = policy
        cfg["algorithm"]["rationalization_geometry_iterations"] = 3
        model = SimpleNamespace(width=100, height=50, dx=2.5, dy=2.5)
        return RibLayoutOptimizer(model, cfg)

    def fixture(self):
        ribs = [Rib((0, float(y)), (50, float(y)), 5, f"R{i}")
                for i, y in enumerate((5, 45, 10, 40, 20, 30))]
        return ribs, np.array([0.01, 0.01, 0.2, 0.2, 0.4, 0.4])

    def test_direct_geometry_parity_with_identity_eq18_including_restoration(self):
        ribs, thicknesses = self.fixture()
        all_histories = []
        all_calls = []
        outputs = []
        for policy in ("smooth", "direct_deletion_geometry", "direct_deletion_sizing"):
            optimizer = self.optimizer(policy, symmetry=True)
            optimizer.cfg["algorithm"]["rationalization_quantile_multiplier"] = 4
            bounds = np.tile([[0, 100], [0, 50], [0, 100], [0, 50]], (6, 1, 1))
            optimizer._solve_rationalization_eq18 = Mock(return_value=(
                list(ribs), thicknesses.copy(), SimpleNamespace(compliance=100.0), bounds,
            ))
            calls = []
            def recovery(active, t, *args, **kwargs):
                calls.append([r.name for r in active])
                # First deletion fails narrowly above exact bound (but below 1.001*Cref).
                response = SimpleNamespace(compliance=105.00001 if len(calls) == 1 else 105.0)
                optimizer.analysis_count += 1
                return list(active), t.copy(), response
            optimizer.optimize_geometry = Mock(side_effect=recovery)
            optimizer.size = Mock(side_effect=lambda r, t, **kw: recovery(r, t)[1:])
            out = optimizer.rationalize(list(ribs), thicknesses.copy(), SimpleNamespace(compliance=100.0), 0.05)
            outputs.append(([r.name for r in out[0]], out[1].tolist(), out[2].compliance))
            self.assertEqual(optimizer._solve_rationalization_eq18.call_count, int(policy == "smooth"))
            if policy == "direct_deletion_sizing":
                optimizer.optimize_geometry.assert_not_called()
                self.assertEqual(optimizer.size.call_args.kwargs["maxiter"], 3)
                self.assertEqual([r.p0 for r in out[0]], [r.p0 for r in ribs if r.name in outputs[-1][0]])
            else:
                optimizer.size.assert_not_called()
            reference = optimizer.rationalization_history[0]
            self.assertEqual(reference["cref"], 105)
            self.assertEqual(reference["final_acceptance_limit"], 105)
            self.assertAlmostEqual(reference["reference_quantile"], 4/6)
            all_histories.append([event for event in optimizer.rationalization_history
                                  if event["event"] in ("filtering_attempt", "restoration_after_failed_validation")])
            all_calls.append(calls)
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], outputs[2])
        self.assertEqual(all_calls[0], all_calls[1])
        self.assertEqual(all_calls[0], all_calls[2])
        for histories in all_histories:
            self.assertEqual(len([e for e in histories if e["event"] == "filtering_attempt"]), 2)
            restoration = next(e for e in histories if e["event"] == "restoration_after_failed_validation")
            self.assertEqual(restoration["restored_names"], ["R2", "R3"])

    def test_all_members_deletion_and_failed_attempts_restore_original(self):
        ribs, t = self.fixture()
        for policy in revision.POLICIES[1:]:
            optimizer = self.optimizer(policy, symmetry=True)
            # One thin member cannot qualify the ordinary threshold batch.
            # The fallback deletes its full orbit, then restores it on failure.
            partial = np.array([0.01, 0.4, 0.4, 0.4, 0.4, 0.4])
            optimizer.optimize_geometry = Mock(side_effect=lambda r, t, *a, **k: (r, t, SimpleNamespace(compliance=106.0)))
            optimizer.size = Mock(side_effect=lambda r, t, **k: (t, SimpleNamespace(compliance=106.0)))
            original = SimpleNamespace(compliance=100.0)
            out = optimizer.rationalize(ribs, partial, original, 0.05)
            self.assertIs(out[2], original)
            np.testing.assert_array_equal(out[1], partial)
            attempts = [e for e in optimizer.rationalization_history if e["event"] == "filtering_attempt"]
            self.assertEqual(len(attempts), 1)
            self.assertEqual(attempts[0]["removed_names"], ["R0", "R1"])
            self.assertTrue(attempts[0]["fallback_used"])
            restoration = next(e for e in optimizer.rationalization_history
                               if e["event"] == "restoration_after_failed_validation")
            self.assertEqual(restoration["restored_names"], ["R0", "R1"])
            self.assertEqual(restoration["remaining_removed_names"], [])
            optimizer.optimize_geometry = Mock(side_effect=lambda r, t, *a, **k: (r, t, SimpleNamespace(compliance=106.0)))
            optimizer.size = Mock(side_effect=lambda r, t, **k: (t, SimpleNamespace(compliance=106.0)))
            out = optimizer.rationalize(ribs, t, original, 0.05)
            self.assertEqual(out[0], ribs)
            np.testing.assert_array_equal(out[1], t)
            self.assertIs(out[2], original)
            self.assertFalse(any(e["event"] == "direct_deletion_fallback"
                                 for e in optimizer.rationalization_history))

    def test_fallback_selects_thinnest_group_with_policy_recovery_and_bounds(self):
        ribs, _ = self.fixture()
        t = np.array([0.4, 0.4, 0.2, 0.2, 0.3, 0.3])
        for policy in revision.POLICIES[1:]:
            with self.subTest(policy=policy):
                optimizer = self.optimizer(policy, symmetry=True)
                optimizer.model.analyze = Mock(return_value=SimpleNamespace(compliance=105.0))
                optimizer._solve_rationalization_eq18 = Mock(side_effect=AssertionError("direct policy"))
                def recovery(active, values, current=None, bounds=None, iterations=None, **kw):
                    self.assertEqual([r.name for r in active], ["R0", "R1", "R4", "R5"])
                    self.assertFalse(missing_mirror_partners(active, ("y",), 100, 50))
                    np.testing.assert_array_equal(values, t[[0, 1, 4, 5]])
                    if policy == "direct_deletion_geometry":
                        self.assertIsNone(current)
                        np.testing.assert_array_equal(bounds, np.tile(
                            [[0, 100], [0, 50], [0, 100], [0, 50]], (4, 1, 1)))
                        self.assertEqual(iterations, 3)
                        self.assertEqual(kw["initial_move_step"], 0.5)
                    else:
                        self.assertEqual(kw["maxiter"], 3)
                    result = optimizer.analyze(active, values)
                    return list(active), values.copy(), result
                optimizer.optimize_geometry = Mock(side_effect=recovery)
                optimizer.size = Mock(side_effect=lambda r, t, **kw: recovery(r, t, **kw)[1:])
                out = optimizer.rationalize(ribs, t, SimpleNamespace(compliance=100.0), 0.05)
                self.assertEqual(out[2].compliance, 105.0)  # Exact Cref is accepted.
                self.assertEqual(optimizer.analysis_count, 1)
                self.assertEqual(optimizer.optimize_geometry.call_count, int(policy.endswith("geometry")))
                self.assertEqual(optimizer.size.call_count, int(policy.endswith("sizing")))
                optimizer._solve_rationalization_eq18.assert_not_called()
                fallback = next(e for e in optimizer.rationalization_history if e["event"] == "direct_deletion_fallback")
                self.assertEqual(fallback["selected_indices"], [3, 4])
                self.assertEqual(fallback["maximum_deletion_attempts"], 1)
                self.assertEqual([g["names"] for g in fallback["ranked_groups"]],
                                 [["R2", "R3"], ["R4", "R5"], ["R0", "R1"]])
                attempts = [e for e in optimizer.rationalization_history if e["event"] == "filtering_attempt"]
                self.assertEqual(len(attempts), 1)
                self.assertTrue(attempts[0]["recovery_solve_attempted"])
                self.assertEqual([r.p0 for r in out[0]], [ribs[i].p0 for i in (0, 1, 4, 5)])
                np.testing.assert_array_equal(t, [0.4, 0.4, 0.2, 0.2, 0.3, 0.3])

    def test_fallback_ranking_uses_group_mean_then_original_index(self):
        ribs, _ = self.fixture()
        for t, expected in ((np.full(6, 0.2), ["R0", "R1"]),
                            (np.array([0.01, 0.8, 0.3, 0.3, 0.4, 0.4]), ["R2", "R3"])):
            optimizer = self.optimizer("direct_deletion_sizing", symmetry=True)
            optimizer.size = Mock(side_effect=lambda r, t, **k: (t, SimpleNamespace(compliance=105.0)))
            optimizer.rationalize(ribs, t, SimpleNamespace(compliance=100.0), 0.05)
            fallback = next(e for e in optimizer.rationalization_history if e["event"] == "direct_deletion_fallback")
            self.assertEqual(fallback["selected_names"], expected)
            self.assertEqual(fallback["ranking"], "saved_input_group_mean_thickness_ascending_then_original_index")

    def test_fallback_rejects_and_restores_one_four_member_orbit_without_retry(self):
        ribs = [Rib((x0, y), (x1, y), 5, f"R{i}")
                for i, (x0, x1, y) in enumerate((
                    (5, 25, 5), (75, 95, 5), (5, 25, 45), (75, 95, 45),
                    (5, 25, 10), (75, 95, 10), (5, 25, 40), (75, 95, 40)))]
        t = np.array([0.2]*4 + [0.4]*4)
        for policy in revision.POLICIES[1:]:
            optimizer = self.optimizer(policy)
            optimizer.mirror_axes = ("x", "y")
            def recovery(active, values, *a, **kw):
                self.assertEqual([r.name for r in active], ["R4", "R5", "R6", "R7"])
                self.assertFalse(missing_mirror_partners(active, ("x", "y"), 100, 50))
                moved = [Rib((r.p0[0]+1, r.p0[1]), r.p1, r.height, r.name) for r in active]
                return moved, np.full(len(values), 0.9), SimpleNamespace(compliance=105.00001)
            optimizer.optimize_geometry = Mock(side_effect=recovery)
            optimizer.size = Mock(side_effect=lambda r, t, **kw: recovery(r, t, **kw)[1:])
            original = SimpleNamespace(compliance=100.0)
            out = optimizer.rationalize(ribs, t, original, 0.05)
            self.assertEqual(out[0], ribs)
            np.testing.assert_array_equal(out[1], t)
            self.assertIs(out[2], original)
            self.assertEqual(optimizer.optimize_geometry.call_count + optimizer.size.call_count, 1)
            restoration = next(e for e in optimizer.rationalization_history
                               if e["event"] == "restoration_after_failed_validation")
            self.assertEqual(restoration["restored_names"], ["R0", "R1", "R2", "R3"])
            self.assertEqual(restoration["remaining_removed_names"], [])

    def test_direct_failed_fea_is_counted_including_singular_recovery(self):
        ribs, ordinary_t = self.fixture()
        for policy in revision.POLICIES[1:]:
            for fallback in (False, True):
                for fail_on in (1, 2):
                    with self.subTest(policy=policy, fallback=fallback, fail_on=fail_on):
                        optimizer = self.optimizer(policy, symmetry=True)
                        t = np.full(6, 0.2) if fallback else ordinary_t
                        optimizer.model.analyze = Mock(side_effect=(
                            [SimpleNamespace(compliance=110.0)]*(fail_on-1)
                            + [np.linalg.LinAlgError("singular fixture")]))
                        def recovery(active, values, *a, **kw):
                            optimizer.analyze(active, values)
                            optimizer.analyze(active, values)
                            self.fail("singular FEA should interrupt recovery")
                        optimizer.optimize_geometry = Mock(side_effect=recovery)
                        optimizer.size = Mock(side_effect=recovery)
                        original = SimpleNamespace(compliance=100.0)
                        out = optimizer.rationalize(ribs, t, original, 0.05)
                        self.assertIs(out[2], original)
                        self.assertEqual(out[0], ribs)
                        np.testing.assert_array_equal(out[1], t)
                        self.assertEqual(optimizer.analysis_count, fail_on)
                        self.assertEqual(optimizer.model.analyze.call_count, fail_on)
                        post = next(e for e in optimizer.rationalization_history
                                    if e["event"] in ("post_filter_geometry", "post_filter_sizing"))
                        self.assertEqual(post["fea"], fail_on)
                        self.assertFalse(post["accepted"])
                        self.assertIsNone(post["compliance"])

    def test_short_light_ordinary_batch_never_triggers_fallback(self):
        ribs, _ = self.fixture()
        ribs[:2] = [Rib(r.p0, (1, r.p1[1]), r.height, r.name) for r in ribs[:2]]
        t = np.full(6, 0.2)
        for policy in revision.POLICIES[1:]:
            optimizer = self.optimizer(policy, symmetry=True)
            optimizer.optimize_geometry = Mock(side_effect=lambda r, t, *a, **k: (r, t, SimpleNamespace(compliance=106.0)))
            optimizer.size = Mock(side_effect=lambda r, t, **k: (t, SimpleNamespace(compliance=106.0)))
            original = SimpleNamespace(compliance=100.0)
            out = optimizer.rationalize(ribs, t, original, 0.05)
            self.assertIs(out[2], original)
            self.assertFalse(any(e["event"] == "direct_deletion_fallback" for e in optimizer.rationalization_history))
            attempt = next(e for e in optimizer.rationalization_history if e["event"] == "filtering_attempt")
            self.assertEqual(attempt["short_light_removed_indices"], [1, 2])
            self.assertFalse(attempt["fallback_used"])

    def test_smooth_empty_batch_and_original_small_set_guard_remain_unchanged(self):
        ribs, _ = self.fixture()
        t = np.full(6, 0.2)
        optimizer = self.optimizer("smooth", symmetry=True)
        original = SimpleNamespace(compliance=100.0)
        optimizer._solve_rationalization_eq18 = Mock(return_value=(ribs, t, original, np.zeros((6, 4, 2))))
        optimizer.optimize_geometry = Mock(side_effect=AssertionError("no smooth fallback"))
        out = optimizer.rationalize(ribs, t, original, 0.05)
        self.assertIs(out[2], original)
        self.assertFalse(any(e["event"] == "direct_deletion_fallback" for e in optimizer.rationalization_history))
        for policy in revision.POLICIES:
            optimizer = self.optimizer(policy)
            optimizer.optimize_geometry = Mock(side_effect=AssertionError("small-set guard"))
            optimizer.size = Mock(side_effect=AssertionError("small-set guard"))
            out = optimizer.rationalize(ribs[:3], t[:3], original, 0.05)
            self.assertIs(out[2], original)
            self.assertEqual(optimizer.rationalization_history, [])

    def test_quantile_controls_and_validation(self):
        optimizer = self.optimizer()
        values = [0.01, 0.2, 0.4, 0.6]
        for multiplier in (0.0, 0.5, 1.0, 2.0, 10.0):
            optimizer.cfg["algorithm"]["rationalization_quantile_multiplier"] = multiplier
            self.assertEqual(optimizer.rationalization_reference_threshold(values),
                             max(0.05, np.quantile(values, min(1, multiplier/4))))
        for key, bad in (("enable_addition", "false"), ("rationalization_policy", "oops"),
                         ("rationalization_quantile_multiplier", -1),
                         ("rationalization_quantile_multiplier", np.nan)):
            cfg = load_case(5)
            cfg["algorithm"][key] = bad
            with self.assertRaises(ValueError):
                RibLayoutOptimizer(object(), cfg)

    def test_no_addition_retains_filter_geometry_and_rationalization(self):
        cfg = load_case(5)
        cfg["algorithm"]["enable_addition"] = False
        optimizer = RibLayoutOptimizer(object(), cfg)
        ribs, t = self.fixture()
        response = SimpleNamespace(compliance=100)
        optimizer.size = Mock(return_value=(t, response))
        optimizer.filter_until_stable = Mock(return_value=(ribs, t, response, 1))
        optimizer._select_addition_candidates = Mock(side_effect=AssertionError("addition disabled"))
        optimizer.optimize_geometry = Mock(return_value=(ribs, t, response))
        optimizer.rationalize = Mock(return_value=(ribs, t, response))
        result = optimizer.run(ribs, ribs)
        optimizer.filter_until_stable.assert_called_once()
        optimizer._select_addition_candidates.assert_not_called()
        optimizer.optimize_geometry.assert_called_once()
        optimizer.rationalize.assert_called_once()
        self.assertEqual([s.name for s in result.stages], ["initial_sizing", "adaptive", "geometry", "rationalized"])


class DriverTests(unittest.TestCase):
    def args(self, *extra):
        return revision.build_parser().parse_args(["--mode", "full", "--case", "5", "--output", "unused", *extra])

    def test_plan_is_read_only_and_case5_help_works(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/"new"
            self.assertEqual(revision.main(["--mode", "full", "--case", "5", "--output", str(output)]), 0)
            self.assertFalse(output.exists())
        for name in ("example5.py", "tools/run_revision_experiments.py"):
            run = subprocess.run([sys.executable, "-B", name, "--help"], cwd=revision.CODE_ROOT,
                                 capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr)
        args = sensitivity_parser().parse_args(["--case", "5", "--source", "x", "--output", "y"])
        self.assertEqual(args.case, 5)
        validate_source_metadata({"example": 5, "mesh_used": [80, 40], "quick": False}, 5, load_case(5))

    def test_no_overwrite_and_no_code_results(self):
        with self.assertRaises(ValueError):
            revision.reserve_output(revision.CODE_ROOT/"results"/"revision", False)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileExistsError):
                revision.reserve_output(Path(temp), False)

    def test_identical_input_and_bounded_trial_plan(self):
        cfg = load_case(5)
        stage = {"name": "geometry", "compliance": 1.0, "ribs": []}
        with patch.object(revision, "source_input", return_value=(cfg, stage, "abc")):
            args = self.args("--mode", "compare", "--source", "input.json", "--max-trials", "3")
            specs = revision.build_specs(args)
            self.assertEqual(len(specs), 3)
            self.assertEqual(len({s["input_design_sha256"] for s in specs}), 1)
            for spec in specs:
                self.assertEqual(spec["config"]["rationalization_relaxation"], 0.05)
            specs[0]["input_stage"]["compliance"] = 9.0
            self.assertEqual(specs[1]["input_stage"]["compliance"], 1.0)
            args.max_trials = 1
            with self.assertRaises(ValueError):
                revision.build_specs(args)
        spec = revision.build_specs(self.args("--mode", "no-addition"))[0]
        self.assertIs(spec["config"]["algorithm"]["enable_addition"], False)

    def test_equal_volume_regular_layout(self):
        cfg = load_case(5)
        for count in (3, 5):
            ribs, t = revision.longitudinal_layout(cfg, count, None)
            self.assertAlmostEqual(RibLayoutOptimizer.volume(ribs, t), 1500.0)
            self.assertFalse(missing_mirror_partners(ribs, ("y",), 100, 50))
        with self.assertRaises(ValueError):
            revision.longitudinal_layout(cfg, 1, 1600)

    def test_small_mesh_full_and_all_comparison_policies_standard_output(self):
        # Bounded numerical integration test, not a formal revision experiment.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.args("--mesh", "4x2", "--iteration-cap", "1", "--no-images")
            spec = revision.build_specs(args)[0]
            spec["config"]["rib"]["segments"] = 2
            spec["config"]["interface_subdivisions_per_cell"] = 1
            spec["config"]["linear_solver"] = "superlu"
            trial = root/"full"
            trial.mkdir()
            result = revision.execute_trial(spec, trial)
            source = trial/result["result"]
            payload = json.loads(source.read_text(encoding="utf-8"))
            self.assertEqual([s["name"] for s in payload["stages"]],
                             ["initial_sizing", "adaptive", "geometry", "rationalized"])
            self.assertAlmostEqual(payload["revision_config"]["load_cases"][0]["pressure"]["value"], -0.0001)
            args = self.args("--mode", "compare", "--source", str(source),
                             "--max-trials", "3", "--iteration-cap", "1", "--no-images")
            specs = revision.build_specs(args)
            limits = []
            for spec in specs:
                destination = root/spec["id"]
                destination.mkdir()
                result = revision.execute_trial(spec, destination)
                self.assertTrue(result["diagnostics"]["final_exact_bound_satisfied"])
                limits.append(result["diagnostics"]["absolute_cref"])
                self.assertTrue((destination/result["result"]).is_file())
                self.assertFalse(list(destination.rglob("*.png")))
            self.assertEqual(len(set(limits)), 1)

    def test_provenance_includes_new_tool_test_and_case(self):
        result = source_provenance(revision.CODE_ROOT, Path(revision.__file__))
        self.assertIn("tools/run_revision_experiments.py", result["tool_files"])
        # Independent aggregate including all requested core/test/tool files.
        from tools.run_robustness_study import _aggregate_hash
        paths = set()
        for pattern in ("*.py", "*.bat", "requirements.txt"):
            paths.update(revision.CODE_ROOT.glob(pattern))
        for directory in ("rib_layout_algorithms", "tests", "tools"):
            paths.update((revision.CODE_ROOT/directory).rglob("*.py"))
        for pattern in ("*.yaml", "*.yml", "*.json"):
            paths.update((revision.CODE_ROOT/"configs").glob(pattern))
        self.assertIn(revision.CODE_ROOT/"tests"/"test_revision.py", paths)
        self.assertIn(revision.CODE_ROOT/"example5.py", paths)
        self.assertEqual(result["source_aggregate_sha256"], _aggregate_hash(list(paths), revision.CODE_ROOT.parent))

    def test_checkpoint_standard_output_resume_and_failure_are_durable(self):
        cfg = load_case(5)
        ribs, thicknesses = revision.longitudinal_layout(cfg, 3, None)
        stages = [Stage(name, ribs, thicknesses.copy(), 1, 1)
                  for name in ("initial_sizing", "adaptive", "geometry", "rationalized")]
        artifacts = SimpleNamespace(config=cfg, run=OptimizationRun(stages=stages), elapsed_seconds=0.1,
                                    optimizer=SimpleNamespace(analysis_count=4, rationalization_history=[]))
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/"trials"
            argv = ["--mode", "no-addition", "--case", "5", "--output", str(output), "--execute", "--no-images"]
            def fake_run(config):
                artifacts.config = config
                return artifacts
            with patch.object(revision, "run_optimization", side_effect=fake_run) as run:
                self.assertEqual(revision.main(argv), 0)
                checkpoint = json.loads(next(output.glob("*/checkpoint.json")).read_text())
                self.assertEqual(checkpoint["status"], "complete")
                payload = json.loads(next(output.glob("*/example_5/results.json")).read_text(encoding="utf-8"))
                validate_source_metadata(payload, 5, cfg)
                self.assertFalse(payload["revision"]["enable_addition"])
                self.assertEqual(len(payload["stages"]), 4)
                self.assertEqual(revision.main([*argv, "--resume"]), 0)
                self.assertEqual(run.call_count, 1)
            failed = Path(temp)/"failed"
            argv[argv.index(str(output))] = str(failed)
            with patch.object(revision, "execute_trial", side_effect=RuntimeError("test failure")):
                with self.assertRaises(RuntimeError):
                    revision.main(argv)
            record = json.loads(next(failed.glob("*/checkpoint.json")).read_text())
            self.assertEqual(record["status"], "failed")
            self.assertIn("test failure", record["error"])


@unittest.skipUnless(os.environ.get("RIB_REVISION_BASELINE_CODE"), "optional immutable backup parity")
class BackupParityTests(unittest.TestCase):
    def test_original_four_cases_bounded_pipeline_bitwise_parity(self):
        baseline = Path(os.environ["RIB_REVISION_BASELINE_CODE"])
        def module(name, relative):
            spec = importlib.util.spec_from_file_location(name, baseline/relative)
            loaded = importlib.util.module_from_spec(spec)
            sys.modules[name] = loaded
            spec.loader.exec_module(loaded)
            return loaded
        old_opt = module("rib_layout_algorithms._revision_legacy_opt", "rib_layout_algorithms/optimization.py")
        old_model = module("rib_layout_algorithms._revision_legacy_model", "rib_layout_algorithms/model_shell.py")
        old_core = module("_revision_legacy_core", "rib_layout_core.py")
        for number in range(1, 5):
            with self.subTest(case=number):
                self.assertEqual(load_case(number), old_core.load_case(number))
                self.assertEqual((baseline/f"example{number}.py").read_bytes(),
                                 (revision.CODE_ROOT/f"example{number}.py").read_bytes())
                cfg = load_case(number)
                cfg["mesh"] = [4, 2]
                cfg["rib"]["segments"] = 2
                cfg["interface_subdivisions_per_cell"] = 1
                cfg["linear_solver"] = "superlu"
                cfg.pop("further_rationalization_relaxation", None)
                for key in ("sizing_max_iterations", "active_set_max_iterations", "geometry_max_iterations",
                            "rationalization_max_iterations", "rationalization_geometry_iterations"):
                    cfg["algorithm"][key] = 1
                all_ribs = initial_ribs(cfg)
                groups = mirror_groups(all_ribs, cfg["mirror_symmetry"], *cfg["domain"])
                selected = {i for group in groups[:2] for i in group}
                ribs = [r for i, r in enumerate(all_ribs) if i in selected]
                outcomes = []
                for cls, model_cls in ((old_opt.RibLayoutOptimizer, old_model.ShellStiffenedPlateModel),
                                       (RibLayoutOptimizer, ShellStiffenedPlateModel)):
                    with patch("rib_layout_core.ShellStiffenedPlateModel", model_cls):
                        model = build_model(cfg)
                    optimizer = cls(model, deepcopy(cfg))
                    result = optimizer.run(list(ribs), all_ribs[:8])
                    outcomes.append({"stages": [revision.stage_to_dict(s) for s in result.stages],
                                     "log": result.log, "analyses": optimizer.analysis_count,
                                     "history": optimizer.rationalization_history})
                self.assertEqual(outcomes[0], outcomes[1])
                self.assertEqual(revision.strict_json_dumps(outcomes[0]), revision.strict_json_dumps(outcomes[1]))
        self.assertEqual((baseline/"example_all.bat").read_bytes(),
                         (revision.CODE_ROOT/"example_all.bat").read_bytes())
        self.assertEqual((baseline/"rib_layout_output.py").read_bytes(),
                         (revision.CODE_ROOT/"rib_layout_output.py").read_bytes())


if __name__ == "__main__":
    unittest.main()
