from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy.sparse import csr_matrix, diags

from rib_layout_core import load_case
from rib_layout_algorithms.density_shell import ParallelLayerModel, offset_transform
from tools import run_density_comparison as cli


def config(symmetry=()):
    cfg = load_case(5)
    cfg.update(domain=[8., 6.], mesh=[4, 3], wall_thickness=.4,
               volume_bound=15., mirror_symmetry=list(symmetry),
               supports={"type": "edge", "edge": "right"},
               load_cases=[{"weight": 1., "forces": [{"point": [0., 6.], "value": [0., -1., -2.]}]},
                           {"weight": 3., "forces": [{"point": [0., 0.], "value": [1., 0., -1.]}]}])
    cfg["rib"]["height"] = 2.
    return cfg


class ParallelLayerTests(unittest.TestCase):
    def test_zero_start_is_rejected_but_zero_response_and_partial_zero_start_are_valid(self):
        model = ParallelLayerModel(config(), linear_solver="superlu")
        zero = np.zeros(model.nelem)
        self.assertGreater(model.analyze(zero)["compliance"], 0.)
        analyses = model.analysis_count
        with self.assertRaisesRegex(ValueError, "nonzero initial density"):
            model.optimize(initial_raw=zero)
        self.assertEqual(model.analysis_count, analyses)
        partial = np.full(model.nelem, model.volume_fraction)
        partial[::2] = 0.
        result = model.optimize(initial_raw=partial, max_iterations=2)
        self.assertGreater(result["final"]["volume"], 0.)
        self.assertLessEqual(result["final"]["volume"], model.budget*(1+1e-10))

    def test_extreme_finite_weights_match_equivalent_normalized_problem_without_mutating_input(self):
        for supplied, expected in (([1e308, 1e308, 0.], [.5, .5, 0.]),
                                   ([1., 7., 0.], [.125, .875, 0.])):
            with self.subTest(weights=supplied):
                cfg = config()
                cfg["load_cases"].append(deepcopy(cfg["load_cases"][0]))
                for load, weight in zip(cfg["load_cases"], supplied):
                    load["weight"] = weight
                original = deepcopy(cfg)
                reference_cfg = deepcopy(cfg)
                for load, weight in zip(reference_cfg["load_cases"], expected):
                    load["weight"] = weight
                with np.errstate(over="raise", invalid="raise"):
                    model = ParallelLayerModel(cfg, linear_solver="superlu")
                reference = ParallelLayerModel(reference_cfg, linear_solver="superlu")
                raw = np.linspace(.1, .7, model.nelem)
                actual, baseline = model.analyze(raw), reference.analyze(raw)
                self.assertEqual(cfg, original)
                self.assertEqual(model.config, original)
                self.assertGreater(actual["compliance"], 0.)
                np.testing.assert_allclose(actual["load_weights"], expected, rtol=1e-15)
                np.testing.assert_allclose(actual["compliance"], baseline["compliance"], rtol=1e-13)
                np.testing.assert_allclose(actual["gradient"], baseline["gradient"], rtol=1e-13)

    def test_good_equilibrium_keeps_original_solution_without_fallback(self):
        model = ParallelLayerModel(config(), linear_solver="superlu")
        matrix = diags([2., 5.]).tocsc()
        rhs = np.array([[2.], [10.]])
        original = np.array([[1.], [2.]])
        with mock.patch.object(model.base, "_solve_global", return_value=original) as solve:
            solution, details = model._equilibrium_solve(matrix, rhs)
        self.assertIs(solution, original)
        self.assertEqual(solve.call_count, 1)
        self.assertEqual(details["method"], "original")

    def test_ill_scaled_equilibrium_fallback_refines_original_equation(self):
        model = ParallelLayerModel(config(), linear_solver="superlu")
        matrix = diags([1e-12, 1e12]).tocsc()
        rhs = np.array([[1.], [2.]])
        calls = []
        def limited_precision(system, force):
            calls.append(system.copy())
            answer = np.linalg.solve(system.toarray(), force)
            return answer*(1+2e-7) if len(calls) <= 2 else answer
        with mock.patch.object(model.base, "_solve_global", side_effect=limited_precision):
            solution, details = model._equilibrium_solve(matrix, rhs)
        self.assertEqual(details["method"], "equilibrated_fallback")
        self.assertEqual(details["refinement_steps"], 1)
        np.testing.assert_allclose(calls[1].diagonal(), 1., rtol=1e-15)
        np.testing.assert_allclose(matrix@solution, rhs, rtol=1e-14)
        self.assertLess(details["relative_residual"], 1e-7)
        np.testing.assert_array_equal(matrix.diagonal(), [1e-12, 1e12])

    def test_equilibrium_fallback_does_not_accept_wrong_equation_or_nonfinite_solution(self):
        model = ParallelLayerModel(config(), linear_solver="superlu")
        matrix = diags([1e-12, 1e12]).tocsc()
        rhs = np.array([[1.], [2.]])
        for bad_solution in (np.zeros_like(rhs), np.full_like(rhs, np.nan)):
            with self.subTest(nonfinite=bool(np.isnan(bad_solution).any())), \
                    mock.patch.object(model.base, "_solve_global", return_value=bad_solution), \
                    self.assertRaisesRegex(RuntimeError, "original-equation residual check"):
                model._equilibrium_solve(matrix, rhs)
        with mock.patch.object(model.base, "_solve_global", side_effect=ValueError("invalid input")), \
                self.assertRaisesRegex(ValueError, "invalid input"):
            model._equilibrium_solve(matrix, rhs)

    def test_offset_map_preserves_all_rigid_rotations(self):
        model = ParallelLayerModel(config(), linear_solver="superlu")
        xyz = np.array([[0, 0, 0], [model.dx, 0, 0],
                        [model.dx, model.dy, 0], [0, model.dy, 0]], float)
        for theta in np.eye(3):
            q = np.concatenate([np.r_[np.cross(theta, p), theta] for p in xyz])
            upper = np.concatenate([np.r_[np.cross(theta, p+[0, 0, model.offset]), theta] for p in xyz])
            np.testing.assert_allclose(model.transform@q, upper, atol=1e-14)
            self.assertLess(abs(q@model.layer_element@q), 1e-8)

    def test_adjacent_layer_full_solid_extension_and_bending_energy(self):
        model = ParallelLayerModel(config(), linear_solver="superlu")
        xyz = np.array([[0, 0, 0], [model.dx, 0, 0],
                        [model.dx, model.dy, 0], [0, model.dy, 0]], float)
        eps, curvature = .002, .003
        extension = np.concatenate([np.array([eps*x, 0, 0, 0, 0, 0]) for x, y, z in xyz])
        bending = np.concatenate([np.array([0, 0, -.5*curvature*x*x, 0, curvature*x, 0]) for x, y, z in xyz])
        coefficient = model.base.E/(1-model.base.nu**2)*model.dx*model.dy
        self.assertAlmostEqual(extension@model.layer_element@extension,
                               coefficient*eps**2*model.layer_thickness, places=9)
        mixed = extension+bending
        mixed_expected = coefficient*(model.layer_thickness*(eps+model.offset*curvature)**2
                                      + model.layer_thickness**3/12*curvature**2)
        self.assertAlmostEqual(mixed@model.layer_element@mixed, mixed_expected, places=8)
        moment = model.layer_thickness*model.offset**2+model.layer_thickness**3/12
        self.assertAlmostEqual(bending@model.layer_element@bending,
                               coefficient*curvature**2*moment, places=8)
        self.assertAlmostEqual(model.metadata()["layer_top_z"], model.config["rib"]["height"])

    def test_exact_base_limit_and_full_solid_positive_stiffness(self):
        model = ParallelLayerModel(config(), linear_solver="superlu")
        k0, k1 = model.stiffness(np.zeros(model.nelem)), model.stiffness(np.ones(model.nelem))
        expected = model.emin_ratio*(k1-model.base.base_stiffness)
        np.testing.assert_allclose((k0-model.base.base_stiffness).toarray(), expected.toarray(), atol=2e-10)
        np.testing.assert_array_equal(k0.toarray(), model.base.base_stiffness.toarray())
        bare = model.evaluate_physical(np.zeros(model.nelem))
        solid = model.evaluate_physical(np.ones(model.nelem))
        self.assertGreater(bare["compliance"], solid["compliance"])
        self.assertLess(solid["relative_residual"], 1e-8)
        np.testing.assert_allclose(solid["load_weights"], [.25, .75])
        self.assertAlmostEqual(solid["compliance"], np.dot([.25, .75], solid["load_compliances"]))

    def test_filter_compliance_and_volume_adjoint_with_mirror_symmetry(self):
        model = ParallelLayerModel(config(("x", "y")), filter_radius=3.5, linear_solver="superlu")
        raw = np.linspace(.15, .75, model.nelem)**1.2
        response = model.analyze(raw)
        eps = 1e-6
        for i in (0, 3, 5, 10):
            delta = np.zeros(model.nelem)
            delta[i] = eps
            plus, minus = model.analyze(raw+delta), model.analyze(raw-delta)
            dc = (plus["compliance"]-minus["compliance"])/(2*eps)
            dv = (plus["volume"]-minus["volume"])/(2*eps)
            self.assertAlmostEqual(dc/response["gradient"][i], 1., delta=3e-5)
            self.assertAlmostEqual(dv/model.volume_gradient()[i], 1., delta=1e-7)
        physical = response["physical_density"].reshape(model.ny, model.nx)
        np.testing.assert_allclose(physical, physical[:, ::-1])
        np.testing.assert_allclose(physical, physical[::-1])

    def test_pressure_conservation_and_exact_base_support_semantics(self):
        cfg = config()
        cfg["load_cases"] = [{"pressure": {"type": "uniform", "value": -.03}}]
        model = ParallelLayerModel(cfg, linear_solver="superlu")
        self.assertAlmostEqual(model.base.load_vectors[0][2::6].sum(), -.03*8*6)
        self.assertTrue(np.any(model.base.load_vectors[0][model.base.fixed_dofs] != 0))
        self.assertEqual(len(model.base.fixed_dofs), 6*(model.ny+1))

    def test_full_density_filter_repairs_only_roundoff_not_bad_inputs_or_filters(self):
        model = ParallelLayerModel(config(), filter_radius=3.5, linear_solver="superlu")
        ones = np.ones(model.nelem)
        np.testing.assert_allclose(model.physical_density(ones), ones, atol=2e-15, rtol=0.)
        # Emulate a few ulps of row-normalization roundoff seen at full density
        # on a refined grid. This is bounded arithmetic repair, not a new bound.
        original_filter = model.filter.copy()
        model.filter = original_filter*(1+4*np.finfo(float).eps)
        repaired = model.physical_density(ones)
        self.assertTrue(np.all(repaired <= 1.))
        np.testing.assert_allclose(repaired, ones, atol=2e-15, rtol=0.)
        with self.assertRaises(ValueError):
            model.physical_density(ones+np.finfo(float).eps)
        with self.assertRaises(ValueError):
            model.evaluate_physical(ones+np.finfo(float).eps)
        model.filter = original_filter*(1+1e-8)
        with self.assertRaisesRegex(ValueError, "beyond floating-point roundoff"):
            model.physical_density(ones)
        model.filter = original_filter*(-1e-8)
        with self.assertRaisesRegex(ValueError, "beyond floating-point roundoff"):
            model.physical_density(ones)

    def test_oc_uses_physical_budget_and_binary_preserves_mirror_orbits(self):
        model = ParallelLayerModel(config(("y",)), linear_solver="superlu")
        result = model.optimize(max_iterations=12)
        self.assertLessEqual(result["final"]["volume"], model.budget*(1+1e-10))
        self.assertLess(result["final"]["compliance"], result["initial"]["compliance"])
        self.assertEqual(result["analysis_count"], result["iterations"]+1)
        rho = result["final"]["physical_density"]
        np.testing.assert_allclose(rho.reshape(3, 4), rho.reshape(3, 4)[::-1])
        binary = model.binary_design(rho)
        self.assertTrue(np.isin(binary, [0., 1.]).all())
        self.assertLessEqual(binary.sum()*model.cell_volume, model.budget+1e-10)
        np.testing.assert_array_equal(binary.reshape(3, 4), binary.reshape(3, 4)[::-1])
        self.assertIn(result["termination_reason"], ("convergence", "iteration_limit"))

    def test_raw_changes_hidden_by_filter_do_not_report_convergence(self):
        model = ParallelLayerModel(config(), linear_solver="superlu")
        # A global averaging filter has an exact nullspace: zero-mean raw
        # changes do not alter the physical design or its actual FE response.
        model.filter = csr_matrix(np.full((model.nelem, model.nelem), 1/model.nelem))
        pattern = np.tile([-1., 1.], model.nelem//2)
        seed = np.full(model.nelem, model.volume_fraction)
        updates = [seed+sign*.02*pattern for sign in (1, -1, 1, -1, 1, -1)]
        with mock.patch.object(model, "_oc_update", side_effect=updates):
            result = model.optimize(max_iterations=6)
        self.assertFalse(result["converged"])
        self.assertEqual(result["termination_reason"], "iteration_limit")
        self.assertTrue(all(row["raw_change"] > .01 for row in result["history"]))
        self.assertTrue(all(row["physical_change"] < 1e-14 for row in result["history"]))
        self.assertTrue(all(row["relative_objective_change"] < 1e-12 for row in result["history"]))
        small_update = seed+.005*pattern
        with mock.patch.object(model, "_oc_update", return_value=small_update):
            settled = model.optimize(max_iterations=6)
        self.assertTrue(settled["converged"])
        self.assertEqual(settled["iterations"], 3)

    def test_raw_convergence_tolerance_must_be_finite_and_positive(self):
        model = ParallelLayerModel(config(), linear_solver="superlu")
        for invalid in (0., -1., float("nan"), float("inf")):
            with self.subTest(tolerance=invalid), self.assertRaises(ValueError):
                model.optimize(raw_density_tolerance=invalid)

    def test_raw_restart_matches_uninterrupted_oc_trajectory(self):
        controls = dict(density_tolerance=1e-30, raw_density_tolerance=1e-30,
                        objective_tolerance=1e-30)
        full_model = ParallelLayerModel(config(), linear_solver="superlu")
        uninterrupted = full_model.optimize(max_iterations=12, **controls)
        first_model = ParallelLayerModel(config(), linear_solver="superlu")
        first = first_model.optimize(max_iterations=5, **controls)
        seed = first["raw_density"].copy()
        restart_model = ParallelLayerModel(config(), linear_solver="superlu")
        restarted = restart_model.optimize(max_iterations=7, initial_raw=seed, **controls)
        self.assertEqual((first["iterations"], restarted["iterations"], uninterrupted["iterations"]), (5, 7, 12))
        np.testing.assert_array_equal(seed, first["raw_density"])
        np.testing.assert_allclose(restarted["raw_density"], uninterrupted["raw_density"], atol=1e-12, rtol=1e-12)
        self.assertAlmostEqual(restarted["initial"]["compliance"], first["final"]["compliance"], places=10)
        self.assertAlmostEqual(restarted["final"]["compliance"], uninterrupted["final"]["compliance"], places=10)
        self.assertEqual(first["initialization"]["type"], "uniform_at_budget")
        self.assertEqual(restarted["initialization"]["type"], "provided_raw_density")

    def test_initial_raw_validation_symmetry_and_physical_budget(self):
        model = ParallelLayerModel(config(("y",)), linear_solver="superlu")
        invalid_seeds = (np.ones(model.nelem-1), np.full(model.nelem, -.01),
                         np.full(model.nelem, 1.01), np.full(model.nelem, np.nan),
                         np.ones(model.nelem), np.ones((3, 4)))
        for seed in invalid_seeds:
            with self.subTest(shape=seed.shape, first=float(seed.flat[0])), self.assertRaises(ValueError):
                model.optimize(max_iterations=1, initial_raw=seed)
        seed = np.linspace(.01, .05, model.nelem)
        original = seed.copy()
        expected = model.analyze(model.symmetric(seed))
        result = model.optimize(max_iterations=1, initial_raw=seed)
        np.testing.assert_array_equal(seed, original)
        np.testing.assert_allclose(result["initial"]["physical_density"], expected["physical_density"], atol=1e-15)
        self.assertLessEqual(result["initialization"]["physical_volume"], model.budget)

    def test_invalid_geometry_capacity_filter_and_densities_rejected(self):
        for options in ({"layer_thickness": 0}, {"penal": .5}, {"emin_ratio": 1.}, {"filter_radius": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ParallelLayerModel(config(), **options)
        cfg = config()
        cfg["volume_bound"] = 1e9
        with self.assertRaisesRegex(ValueError, "capacity"):
            ParallelLayerModel(cfg)
        model = ParallelLayerModel(config(), linear_solver="superlu")
        with self.assertRaises(ValueError):
            model.analyze(np.full(model.nelem, -1.))


class DensityCliTests(unittest.TestCase):
    def test_plan_only_all_cases_never_constructs_or_executes_model(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/"new"
            with mock.patch.object(cli, "execute_case") as execute, redirect_stdout(io.StringIO()) as output:
                self.assertEqual(cli.main(["--output", str(target)]), 0)
            self.assertEqual([s["config"]["number"] for s in json.loads(output.getvalue())["cases"]], [1,2,3,4,5])
            execute.assert_not_called()
            self.assertFalse(target.exists())

    def test_raw_tolerance_is_planned_and_continuous_execution_has_no_binary_reanalysis(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/"planned"
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(cli.main(["--cases", "5", "--output", str(target),
                                           "--raw-density-tolerance", ".007"]), 0)
            plan = json.loads(output.getvalue())
            self.assertEqual(plan["cases"][0]["optimization"]["raw_density_tolerance"], .007)
            spec = dict(config=config(), options=dict(linear_solver="superlu"),
                        optimization=dict(max_iterations=3, raw_density_tolerance=.01), no_images=True)
            completed = Path(folder)/"completed"
            with mock.patch.object(ParallelLayerModel, "binary_design", side_effect=AssertionError("binary called")), \
                    redirect_stdout(io.StringIO()):
                result = cli.execute_case(spec, completed)
            self.assertEqual(result["total_analyses"], result["iterations"]+1)
            self.assertEqual(result["initialization"]["type"], "uniform_at_budget")
            self.assertIn("final_continuous", result)
            self.assertFalse(any("binary" in key or "threshold" in key for key in result))
            self.assertFalse(any("binary" in path.name or "threshold" in path.name for path in completed.iterdir()))
            self.assertIn("raw_change", (completed/"history.csv").read_text().splitlines()[0])

    def test_existing_and_source_outputs_rejected(self):
        with self.assertRaises(ValueError):
            cli.output_path(cli.CODE_ROOT/"results/new")
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            cli.output_path(Path(folder))

    def test_execute_failure_preserves_manifest_and_case_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/"failure"
            with mock.patch.object(cli, "ParallelLayerModel", side_effect=RuntimeError("solver failure")), \
                    redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "solver failure"):
                cli.main(["--cases", "5", "--output", str(target), "--execute"])
            self.assertEqual(json.loads((target/"manifest.json").read_text())["status"], "failed")
            self.assertEqual(json.loads((target/"example_5/checkpoint.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
