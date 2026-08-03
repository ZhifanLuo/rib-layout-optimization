from __future__ import annotations

import os
import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from rib_layout_core import (
    build_model, candidate_ribs, initial_ribs, load_case,
)
from rib_layout_algorithms.frame import frame_global_stiffness
from rib_layout_algorithms.optimization import (
    RibLayoutOptimizer,
    Stage,
    collinear_covered,
    collinear_overlap,
    geometry_move_freeze_reasons,
    sca_step_converged,
    solve_geometry_convex_subproblem,
    solve_rationalization_convex_subproblem,
    smooth_member_count,
    smooth_member_count_gradient,
)
from rib_layout_algorithms.model import AnalysisResult, Rib, StiffenedPlateModel
from rib_layout_algorithms.plotting import (
    _format_3d_axis,
    _load_label_position,
    _place_panel_caption,
    _rib_footprint,
    _rib_prism_faces,
    example1_detailed_panels,
    example2_detailed_timeline,
    paper_style_panels,
)
from rib_layout_algorithms.shell import shell_q4_stiffness
from rib_layout_algorithms.model_shell import ShellStiffenedPlateModel
from rib_layout_algorithms.move_limit import EnhancedMMAMoveLimit
from rib_layout_algorithms.symmetry import (
    build_mirror_variable_map,
    mirror_axes,
    missing_mirror_partners,
)
from tools.run import rationalization_paths


class CoreTests(unittest.TestCase):
    def test_progress_message_is_logged_and_flushed_immediately(self):
        cfg = load_case(1, quick=True)
        optimizer = RibLayoutOptimizer(object(), cfg)
        ribs = initial_ribs(cfg)[:3]
        result = SimpleNamespace(compliance=1.23456789)
        expected = (
            "[progress] sizing optimization is finished, with compliance = "
            "1.234568, ribs = 3"
        )

        with patch("builtins.print") as progress_print:
            optimizer._report_progress("sizing optimization", ribs, result)

        progress_print.assert_called_once_with(expected, flush=True)
        self.assertEqual(optimizer.log, [expected])

    def test_deterministic_runtime_is_mandatory(self):
        self.assertEqual(os.environ.get("MKL_NUM_THREADS"), "1")
        self.assertEqual(os.environ.get("OMP_NUM_THREADS"), "1")
        self.assertEqual(os.environ.get("MKL_DYNAMIC"), "FALSE")
        cfg = load_case(1, quick=True)
        args = (
            cfg["domain"][0], cfg["domain"][1], 1, 1,
            cfg["wall_thickness"], cfg["material"]["E"],
            cfg["material"]["nu"], cfg["load_cases"], cfg["supports"],
        )
        with self.assertRaisesRegex(ValueError, "sensitivity_workers=1"):
            ShellStiffenedPlateModel(*args, sensitivity_workers=2)
        with self.assertRaisesRegex(ValueError, "linear_solver_threads=1"):
            ShellStiffenedPlateModel(*args, linear_solver_threads=2)

    def test_geometry_convex_dual_solver_matches_slsqp_reference(self):
        from scipy.optimize import minimize

        current = np.array([0.4, 0.7, 1.0, 2.0, 3.0, 4.0])
        a = np.array([0.8, 0.3])
        gp = np.array([-0.2, 0.1, -0.4, 0.3])
        vg = np.array([2.0, 1.5, 0.5, -0.2, 0.3, -0.1])
        lower = np.array([0.1, 0.1, 0.0, 0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 5.0, 5.0, 5.0, 5.0])
        volume_at_current = 4.0
        volume_bound = 3.7
        proximal = 0.2
        compliance_scale = 2.5
        coordinate_scale = np.array([5.0, 4.0, 5.0, 4.0])

        result = solve_geometry_convex_subproblem(
            current, a, gp, vg, volume_at_current, volume_bound,
            proximal, compliance_scale, coordinate_scale, lower, upper,
        )

        def objective(y):
            dt = np.sum(a*(1.0/y[:2]-1.0/current[:2]))
            dp = y[2:]-current[2:]
            return dt+gp@dp+0.5*proximal*compliance_scale*np.sum(
                (dp/coordinate_scale)**2
            )

        reference = minimize(
            objective,
            current,
            method="SLSQP",
            bounds=list(zip(lower, upper)),
            constraints=[{
                "type": "ineq",
                "fun": lambda y: volume_bound-
                (volume_at_current+vg@(y-current)),
            }],
            options={"maxiter": 1000, "ftol": 1.0e-12},
        )
        self.assertTrue(reference.success)
        self.assertLessEqual(volume_at_current+vg@(result-current), volume_bound+1e-10)
        self.assertAlmostEqual(objective(result), objective(reference.x), places=8)

    def test_rationalization_dual_solver_matches_slsqp_reference(self):
        from scipy.optimize import minimize

        current = np.array([0.5, 0.7, 1.0, 2.0, 3.0, 4.0])
        count_gradient = np.array([1.2, 0.7, 0.0, 0.0, 0.0, 0.0])
        reciprocal = np.array([0.8, 0.5])
        geometry_gradient = np.array([-0.2, 0.1, -0.15, 0.08])
        volume_gradient = np.array([2.0, 1.5, 0.2, -0.1, 0.15, -0.05])
        lower = np.array([0.1, 0.1, 0.0, 0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 5.0, 5.0, 5.0, 5.0])
        thickness_scale = np.array([0.5, 0.7])
        coordinate_scale = np.array([5.0, 4.0, 5.0, 4.0])
        proximal = 0.2
        compliance_scale = 4.0
        compliance_at_current = 4.0
        compliance_bound = 4.3
        volume_at_current = 3.0
        volume_bound = 2.9

        result = solve_rationalization_convex_subproblem(
            current=current,
            count_gradient=count_gradient,
            reciprocal_coefficients=reciprocal,
            geometry_gradient=geometry_gradient,
            volume_gradient=volume_gradient,
            compliance_at_current=compliance_at_current,
            compliance_bound=compliance_bound,
            volume_at_current=volume_at_current,
            volume_bound=volume_bound,
            proximal=proximal,
            compliance_scale=compliance_scale,
            thickness_scale=thickness_scale,
            coordinate_scale=coordinate_scale,
            lower=lower,
            upper=upper,
        )

        def objective(y):
            difference = y-current
            return float(
                count_gradient@difference
                + 0.5*proximal*np.sum((difference[:2]/thickness_scale)**2)
                + 0.5*proximal*np.sum((difference[2:]/coordinate_scale)**2)
            )

        def compliance_residual(y):
            difference = y-current
            return float(
                compliance_at_current
                + np.sum(reciprocal*(1.0/y[:2]-1.0/current[:2]))
                + geometry_gradient@difference[2:]
                + 0.5*proximal*compliance_scale
                * np.sum((difference[2:]/coordinate_scale)**2)
                - compliance_bound
            )

        def volume_residual(y):
            return float(
                volume_at_current+volume_gradient@(y-current)-volume_bound
            )

        reference = minimize(
            objective,
            current,
            method="SLSQP",
            bounds=list(zip(lower, upper)),
            constraints=[
                {"type": "ineq", "fun": lambda y: -compliance_residual(y)},
                {"type": "ineq", "fun": lambda y: -volume_residual(y)},
            ],
            options={"maxiter": 2000, "ftol": 1.0e-12},
        )
        self.assertTrue(reference.success)
        self.assertTrue(result.success)
        self.assertLessEqual(compliance_residual(result.x), 1.0e-9)
        self.assertLessEqual(volume_residual(result.x), 1.0e-9)
        self.assertAlmostEqual(objective(result.x), objective(reference.x), places=9)

    def test_dual_solver_activates_reachable_slack_volume_bound(self):
        current = np.array([0.5, 0.0])
        result = solve_geometry_convex_subproblem(
            current=current,
            reciprocal_coefficients=np.array([1.0]),
            geometry_gradient=np.array([0.0]),
            volume_gradient=np.array([1.0, 1.0]),
            volume_at_current=0.5,
            volume_bound=1.5,
            proximal=1.0,
            compliance_scale=1.0,
            coordinate_scale=np.array([1.0]),
            lower=np.array([0.1, -1.0]),
            upper=np.array([1.0, 1.0]),
        )
        volume = 0.5+np.array([1.0,1.0])@(result-current)
        self.assertAlmostEqual(volume, 1.5, places=10)
        # The unconstrained solution [1,0] has volume 1.0 and must not be
        # returned directly when equality is reachable inside the move box.
        self.assertGreater(result[1], 0.0)

    def test_dual_solver_returns_minimum_residual_when_volume_bound_unreachable(self):
        current = np.array([0.5, 0.0])
        result = solve_geometry_convex_subproblem(
            current=current,
            reciprocal_coefficients=np.array([1.0]),
            geometry_gradient=np.array([0.0]),
            volume_gradient=np.array([1.0, 1.0]),
            volume_at_current=0.5,
            volume_bound=3.0,
            proximal=1.0,
            compliance_scale=1.0,
            coordinate_scale=np.array([1.0]),
            lower=np.array([0.1, -1.0]),
            upper=np.array([1.0, 1.0]),
        )
        volume = 0.5+np.array([1.0,1.0])@(result-current)
        self.assertAlmostEqual(volume, 2.0, places=10)
        self.assertTrue(np.allclose(result, [1.0,1.0]))

    def test_rib_plot_uses_physical_thickness_and_positive_height(self):
        rib = Rib((5.0, 5.0), (15.0, 5.0), 2.0, "horizontal")
        footprint = _rib_footprint(rib, 2.0, (20.0, 20.0))
        self.assertAlmostEqual(float(np.ptp(footprint[:, 1])), 2.0)
        self.assertAlmostEqual(float(np.min(footprint[:, 0])), 5.0)
        self.assertAlmostEqual(float(np.max(footprint[:, 0])), 15.0)
        faces = _rib_prism_faces(rib, 2.0, (20.0, 20.0))
        z_coordinates = {point[2] for face in faces for point in face}
        self.assertEqual(z_coordinates, {0.0, 2.0})

    def test_boundary_rib_footprint_is_clipped_to_ground_shell(self):
        rib = Rib((0.0, 0.0), (20.0, 0.0), 2.0, "boundary")
        footprint = _rib_footprint(rib, 2.0, (20.0, 20.0))
        self.assertGreaterEqual(float(footprint[:, 0].min()), 0.0)
        self.assertLessEqual(float(footprint[:, 0].max()), 20.0)
        self.assertGreaterEqual(float(footprint[:, 1].min()), 0.0)
        self.assertLessEqual(float(footprint[:, 1].max()), 20.0)
        self.assertAlmostEqual(float(footprint[:, 1].max()), 1.0)

    def test_q4_shell_is_symmetric_and_has_rigid_modes(self):
        xyz = np.array([[0., 0., 0.], [2., 0., 0.], [2., 1., 0.], [0., 1., 0.]])
        k = shell_q4_stiffness(xyz, 0.1, 70000., 0.3)
        self.assertTrue(np.allclose(k, k.T, atol=1e-8))
        eigenvalues = np.linalg.eigvalsh(k)
        self.assertGreaterEqual(eigenvalues.min(), -1e-7)
        tolerance = max(abs(eigenvalues[-1]), 1.0) * 1.0e-10
        self.assertEqual(np.count_nonzero(np.abs(eigenvalues) < tolerance), 6)

    def test_q4_shell_constant_membrane_and_shear_patch(self):
        E, nu, thickness = 70000.0, 0.3, 0.1
        width, height = 2.0, 1.0
        xyz = np.array([[0., 0., 0.], [width, 0., 0.], [width, height, 0.], [0., height, 0.]])
        k = shell_q4_stiffness(xyz, thickness, E, nu)

        strain = np.array([0.01, -0.004, 0.006])
        membrane = np.zeros(24)
        for i, (x, y, _) in enumerate(xyz):
            membrane[6*i] = strain[0]*x + 0.5*strain[2]*y
            membrane[6*i+1] = strain[1]*y + 0.5*strain[2]*x
        plane = E/(1-nu**2)*np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
        exact_membrane = width*height*thickness*float(strain @ plane @ strain)
        self.assertAlmostEqual(float(membrane @ k @ membrane), exact_membrane, places=10)

        shear = np.array([0.002, -0.003])
        transverse = np.zeros(24)
        for i, (x, y, _) in enumerate(xyz):
            transverse[6*i+2] = shear[0]*x + shear[1]*y
        shear_modulus = E/(2*(1+nu))
        exact_shear = width*height*(5/6)*shear_modulus*thickness*float(shear @ shear)
        self.assertAlmostEqual(float(transverse @ k @ transverse), exact_shear, places=10)

    def test_condensed_vertical_shell_rib_is_stable(self):
        cfg = load_case(1, quick=True)
        cfg["mesh"] = [4, 4]
        model = ShellStiffenedPlateModel(
            cfg["domain"][0], cfg["domain"][1], 4, 4,
            cfg["wall_thickness"], cfg["material"]["E"], cfg["material"]["nu"],
            cfg["load_cases"], cfg["supports"],
        )
        rib = initial_ribs(cfg)[0]
        k = model.rib_stiffness(rib, 0.2)
        self.assertTrue(np.allclose(k.toarray(), k.toarray().T, atol=1e-7))
        result = model.analyze([rib], [0.2])
        self.assertTrue(np.isfinite(result.compliance))
        self.assertGreater(result.compliance, 0.0)

    def test_expanded_tied_rib_analysis_matches_condensed_reference(self):
        from scipy.sparse.linalg import splu

        cfg = load_case(1, quick=True)
        model = ShellStiffenedPlateModel(
            cfg["domain"][0], cfg["domain"][1], 4, 4,
            cfg["wall_thickness"], cfg["material"]["E"], cfg["material"]["nu"],
            cfg["load_cases"], cfg["supports"],
            interface_subdivisions_per_cell=4,
        )
        ribs = initial_ribs(cfg)[:3]
        thicknesses = np.array([0.2, 0.15, 0.1])
        expanded = model.analyze(ribs, thicknesses)
        condensed_matrix = model.stiffness(ribs, thicknesses)
        free_matrix = condensed_matrix[model.free_dofs][:, model.free_dofs].tocsc()
        solver = splu(free_matrix)
        reference_compliance = 0.0
        for weight, load in zip(model.load_weights, model.load_vectors):
            displacement = np.zeros(model.ndof)
            displacement[model.free_dofs] = solver.solve(load[model.free_dofs])
            reference_compliance += float(weight)*(load@displacement)
        self.assertLess(
            abs(expanded.compliance-reference_compliance)/reference_compliance,
            1.0e-9,
        )

    def test_global_solver_preserves_single_column_rhs_shape(self):
        from scipy.sparse import eye

        cfg = load_case(1, quick=True)
        model = build_model(cfg)
        rhs = np.arange(1.0, 6.0)[:, None]
        cleanup_calls = []
        model._pardiso_spsolve = lambda matrix, values: np.asarray(values)
        model._pardiso_solver = SimpleNamespace(
            free_memory=lambda: cleanup_calls.append(True)
        )
        solution = model._solve_global(eye(5, format="csc"), rhs)
        self.assertEqual(solution.shape, rhs.shape)
        self.assertTrue(np.allclose(solution, rhs))
        self.assertEqual(cleanup_calls, [True])

    def test_global_solver_releases_pardiso_after_fallback(self):
        from scipy.sparse import eye

        cfg = load_case(1, quick=True)
        model = build_model(cfg)
        rhs = np.arange(1.0, 6.0)
        events = []

        def failed_pardiso(matrix, values):
            events.append("pardiso")
            raise RuntimeError("force SuperLU fallback")

        def factorize(matrix):
            events.append("factorize")

            def solve(values):
                events.append("fallback")
                return np.asarray(values)

            return SimpleNamespace(solve=solve)

        model.linear_solver = "auto"
        model._pardiso_spsolve = failed_pardiso
        model._factorize_spd = factorize
        model._pardiso_solver = SimpleNamespace(
            free_memory=lambda: events.append("release")
        )

        solution = model._solve_global(eye(5, format="csc"), rhs)
        self.assertTrue(np.allclose(solution, rhs))
        self.assertEqual(events, ["pardiso", "factorize", "fallback", "release"])

    def test_global_solver_releases_pardiso_when_solve_raises(self):
        from scipy.sparse import eye

        cfg = load_case(1, quick=True)
        model = build_model(cfg)
        cleanup_calls = []
        model.linear_solver = "pardiso"
        model._pardiso_spsolve = lambda matrix, values: (_ for _ in ()).throw(
            RuntimeError("Pardiso failure")
        )
        model._pardiso_solver = SimpleNamespace(
            free_memory=lambda: cleanup_calls.append(True)
        )

        with self.assertRaisesRegex(RuntimeError, "Pardiso failure"):
            model._solve_global(eye(5, format="csc"), np.ones(5))
        self.assertEqual(cleanup_calls, [True])

    def test_global_solver_releases_pardiso_when_fallback_raises(self):
        from scipy.sparse import eye

        cfg = load_case(1, quick=True)
        model = build_model(cfg)
        cleanup_calls = []
        model.linear_solver = "auto"
        model._pardiso_spsolve = lambda matrix, values: np.full_like(values, np.nan)
        model._factorize_spd = lambda matrix: SimpleNamespace(
            solve=lambda values: (_ for _ in ()).throw(RuntimeError("fallback failure"))
        )
        model._pardiso_solver = SimpleNamespace(
            free_memory=lambda: cleanup_calls.append(True)
        )

        with self.assertRaisesRegex(RuntimeError, "fallback failure"):
            model._solve_global(eye(5, format="csc"), np.ones(5))
        self.assertEqual(cleanup_calls, [True])

    def test_shell_rib_stiffness_cache_is_lru_bounded(self):
        cfg = load_case(1, quick=True)
        model = ShellStiffenedPlateModel(
            cfg["domain"][0], cfg["domain"][1], 4, 4,
            cfg["wall_thickness"], cfg["material"]["E"], cfg["material"]["nu"],
            cfg["load_cases"], cfg["supports"],
            rib_cache_max_entries=3,
        )
        ribs = initial_ribs(cfg)[:4]
        first_key = model._rib_numeric_state_key(ribs[0], 0.2)
        for rib in ribs[:3]:
            model.rib_stiffness(rib, 0.2)
        # A cache hit makes the first item most recently used.
        model.rib_stiffness(ribs[0], 0.2)
        model.rib_stiffness(ribs[3], 0.2)
        self.assertEqual(len(model._rib_cache), 3)
        self.assertIn(first_key, model._rib_cache)
        self.assertNotIn(
            model._rib_numeric_state_key(ribs[1], 0.2), model._rib_cache
        )

    def test_sub_rounding_endpoint_update_uses_fresh_numeric_rib_basis(self):
        model = ShellStiffenedPlateModel(
            1.0, 1.0, 2, 2, 0.1, 70000.0, 0.3,
            [{"forces": [{"point": [1.0, 1.0], "value": [0.0, 0.0, 1.0]}]}],
            {"type": "edge", "edge": "left"},
            interface_subdivisions_per_cell=4,
        )
        rib = Rib((0.2, 0.2), (0.8, 0.7), 0.2, "interior", 4)
        moved = Rib(
            rib.p0,
            (rib.p1[0] + 4.0e-11, rib.p1[1]),
            rib.height,
            rib.name,
            rib.segments,
        )
        thickness = np.array([0.1])

        self.assertEqual(rib.key, moved.key)
        self.assertNotEqual(
            model._rib_numeric_geometry_key(rib),
            model._rib_numeric_geometry_key(moved),
        )
        self.assertNotEqual(
            model._rib_numeric_state_key(rib, thickness[0]),
            model._rib_numeric_state_key(rib, thickness[0] + 4.0e-11),
        )
        result = model.analyze([rib], thickness)
        original_bottom = model._rib_sparse_basis(rib)[0]

        gradient = model.geometry_gradient(
            [moved], thickness, result, step=1.0e-6
        )
        moved_bottom = model._rib_sparse_basis(moved)[0]

        self.assertTrue(np.all(np.isfinite(gradient)))
        self.assertGreater(
            float(np.max(np.abs(moved_bottom-original_bottom))), 1.0e-12
        )
        self.assertTrue(np.array_equal(
            moved_bottom, model.rib_bottom_points(moved)
        ))

    def test_rib_ground_interface_trace_is_refined_and_compatible(self):
        model = ShellStiffenedPlateModel(
            1.0, 1.0, 1, 1, 0.1, 70000.0, 0.3,
            [{"forces": [{"point": [1.0, 1.0], "value": [0.0, 0.0, 1.0]}]}],
            {"type": "points", "points": [[0.0, 0.0]]},
            interface_subdivisions_per_cell=4,
        )
        rib = Rib((0.0, 0.0), (1.0, 1.0), 0.2, "diagonal", 1)
        points = model.rib_bottom_points(rib)
        self.assertEqual(len(points), 5)

        # Master Q4 scalar trace is s^2 for nodal values [0,0,1,0].
        ground = np.zeros(model.ndof)
        ground[6*model.node(1, 1)] = 1.0
        nodal_values = np.array([model.response_at(ground, point)[0] for point in points])
        self.assertTrue(np.allclose(nodal_values, np.linspace(0.0, 1.0, 5)**2))
        mismatch = []
        for i in range(len(points)-1):
            midpoint = 0.5*(points[i]+points[i+1])
            slave_value = 0.5*(nodal_values[i]+nodal_values[i+1])
            mismatch.append(abs(slave_value-model.response_at(ground, midpoint)[0]))
        self.assertLessEqual(max(mismatch), 1.0/64.0 + 1.0e-14)

        # With no artificial ground springs, a uniform translation has zero energy.
        rigid = np.zeros(model.ndof)
        rigid[0::6] = 1.0
        scale = max(float(np.max(np.abs(model.base_stiffness.diagonal()))), 1.0)
        self.assertLess(abs(float(rigid @ (model.base_stiffness @ rigid))), scale*1.0e-10)

    def test_boundary_geometry_sensitivity_matches_true_fea_difference(self):
        model = ShellStiffenedPlateModel(
            1.0, 1.0, 1, 1, 0.1, 70000.0, 0.3,
            [{"forces": [{"point": [1.0, 0.0], "value": [0.0, -1.0, 0.0]}]}],
            {"type": "edge", "edge": "left"},
            interface_subdivisions_per_cell=4,
        )
        rib = Rib((0.0, 0.0), (1.0, 0.0), 0.2, "bottom", 1)
        thickness = np.array([0.1])
        result = model.analyze([rib], thickness)
        gradient = model.geometry_gradient([rib], thickness, result, step=0.01)[0, 1]
        h = 1.0e-6
        moved = Rib((0.0, h), (1.0, 0.0), 0.2, "bottom", 1)
        true_difference = (model.analyze([moved], thickness).compliance-result.compliance)/h
        self.assertLess(abs(gradient-true_difference)/max(abs(true_difference), 1.0e-16), 0.01)

    def test_interior_envelope_sensitivities_match_true_fea_differences(self):
        model = ShellStiffenedPlateModel(
            1.0, 1.0, 2, 2, 0.1, 70000.0, 0.3,
            [{"forces": [{"point": [1.0, 1.0], "value": [0.0, 0.0, 1.0]}]}],
            {"type": "edge", "edge": "left"},
            interface_subdivisions_per_cell=4,
        )
        rib = Rib((0.2, 0.2), (0.8, 0.7), 0.2, "interior", 4)
        thickness = np.array([0.1])
        result = model.analyze([rib], thickness)
        h = 1.0e-5

        thickness_gradient = model.compliance_gradient(
            [rib], thickness, result
        )[0]
        true_thickness = (
            model.analyze([rib], [thickness[0]+h]).compliance
            - model.analyze([rib], [thickness[0]-h]).compliance
        )/(2*h)
        self.assertLess(
            abs(thickness_gradient-true_thickness)/max(abs(true_thickness),1e-16),
            1.0e-4,
        )

        geometry_gradient = model.geometry_gradient(
            [rib], thickness, result, h
        )[0, 2]
        plus = Rib(rib.p0, (rib.p1[0]+h, rib.p1[1]), rib.height, rib.name, rib.segments)
        minus = Rib(rib.p0, (rib.p1[0]-h, rib.p1[1]), rib.height, rib.name, rib.segments)
        true_geometry = (
            model.analyze([plus], thickness).compliance
            - model.analyze([minus], thickness).compliance
        )/(2*h)
        centered_geometry = model.geometry_gradient_centered_difference(
            [rib], thickness, result, h
        )[0, 2]
        self.assertLess(
            abs(geometry_gradient-true_geometry)/max(abs(true_geometry),1e-16),
            1.0e-3,
        )
        self.assertLess(
            abs(centered_geometry-true_geometry)/max(abs(true_geometry),1e-16),
            1.0e-3,
        )
        self.assertLess(
            abs(geometry_gradient-centered_geometry)/max(abs(true_geometry),1e-16),
            2.0e-3,
        )

    def test_grid_intersection_shape_derivative_matches_trace_difference(self):
        model = ShellStiffenedPlateModel(
            1.0, 1.0, 2, 2, 0.1, 70000.0, 0.3,
            [{"forces": [{"point": [1.0, 1.0], "value": [0.0, 0.0, 1.0]}]}],
            {"type": "edge", "edge": "left"},
            interface_subdivisions_per_cell=4,
        )
        rib = Rib((0.1, 0.2), (0.9, 0.7), 0.2, "grid_crossing", 4)
        bottom, derivatives = model._rib_bottom_points_and_shape_derivatives(rib)
        self.assertTrue(np.allclose(bottom, model.rib_bottom_points(rib)))

        h = 1.0e-6
        plus = Rib(rib.p0, (rib.p1[0]+h, rib.p1[1]), rib.height, rib.name, rib.segments)
        minus = Rib(rib.p0, (rib.p1[0]-h, rib.p1[1]), rib.height, rib.name, rib.segments)
        trace_difference = (
            model.rib_bottom_points(plus)-model.rib_bottom_points(minus)
        )/(2*h)
        self.assertTrue(np.allclose(
            derivatives[:, :, 2], trace_difference, rtol=1.0e-7, atol=1.0e-9
        ))

    def test_grid_node_generalized_shape_sensitivity_matches_true_fea(self):
        model = ShellStiffenedPlateModel(
            1.0, 1.0, 2, 2, 0.1, 70000.0, 0.3,
            [{"forces": [{"point": [1.0, 1.0], "value": [0.0, 0.0, 1.0]}]}],
            {"type": "edge", "edge": "left"},
            interface_subdivisions_per_cell=4,
        )
        rib = Rib((0.25, 0.25), (0.75, 0.75), 0.2, "node_crossing", 2)
        thickness = np.array([0.1])
        result = model.analyze([rib], thickness)
        h = 1.0e-6
        analytic = model.geometry_gradient(
            [rib], thickness, result, h
        )[0, 2]
        centered = model.geometry_gradient_centered_difference(
            [rib], thickness, result, h
        )[0, 2]
        plus = Rib(rib.p0, (rib.p1[0]+h, rib.p1[1]), rib.height, rib.name, rib.segments)
        minus = Rib(rib.p0, (rib.p1[0]-h, rib.p1[1]), rib.height, rib.name, rib.segments)
        true_difference = (
            model.analyze([plus], thickness).compliance
            - model.analyze([minus], thickness).compliance
        )/(2*h)
        scale = max(abs(true_difference), 1.0e-16)
        self.assertLess(abs(analytic-true_difference)/scale, 1.0e-4)
        self.assertLess(abs(analytic-centered)/scale, 1.0e-4)

    def test_sizing_returns_best_feasible_true_fea_incumbent(self):
        cfg = load_case(1, quick=True)

        class WorseningModel:
            def __init__(self):
                self.calls = 0

            def analyze(self, ribs, thicknesses):
                self.calls += 1
                return SimpleNamespace(compliance=float(self.calls))

            @staticmethod
            def compliance_gradient(ribs, thicknesses, result):
                return -np.ones(len(ribs))

        model = WorseningModel()
        optimizer = RibLayoutOptimizer(model, cfg)
        thicknesses, result = optimizer.size(initial_ribs(cfg)[:2], maxiter=1)
        self.assertEqual(model.calls, 2)
        self.assertEqual(result.compliance, 1.0)
        self.assertEqual(optimizer.sizing_history[-1]["best_feasible_outer"], 0)
        self.assertTrue(any("best feasible" in line for line in optimizer.log))

    def test_geometry_rejects_worse_true_response_contracts_and_retries(self):
        cfg = load_case(1, quick=True)
        cfg["algorithm"]["geometry_max_iterations"] = 1
        cfg["algorithm"]["geometry_true_response_max_retries"] = 2

        class ResponseModel:
            width, height = cfg["domain"]
            dx = width/cfg["mesh"][0]
            dy = height/cfg["mesh"][1]

            def __init__(self):
                self.responses = iter((1.2, 0.9))

            @staticmethod
            def compliance_gradient(ribs, thicknesses, result):
                return -np.ones(len(ribs))

            @staticmethod
            def geometry_gradient(ribs, thicknesses, result, step):
                return np.zeros((len(ribs), 4))

            def analyze(self, ribs, thicknesses):
                return SimpleNamespace(compliance=next(self.responses))

        rib = Rib((0.0, 0.0), (10.0, 10.0), 2.0, "A")
        optimizer = RibLayoutOptimizer(ResponseModel(), cfg)
        history = []
        with patch(
            "rib_layout_algorithms.optimization.solve_geometry_convex_subproblem",
            side_effect=lambda *args: np.asarray(args[0], float).copy(),
        ):
            _, _, result = optimizer.optimize_geometry(
                [rib], np.array([0.2]), SimpleNamespace(compliance=1.0),
                iteration_history=history,
            )
        self.assertEqual(result.compliance, 0.9)
        self.assertEqual(history[0]["response_retry_count"], 1)
        self.assertEqual(
            [trial["accepted"] for trial in history[0]["response_trials"]],
            [False, True],
        )
        self.assertAlmostEqual(
            history[0]["response_trials"][1]["move_global"], 0.375
        )

    def test_geometry_returns_initial_incumbent_when_rejection_is_disabled(self):
        cfg = load_case(1, quick=True)
        cfg["algorithm"]["geometry_max_iterations"] = 1
        cfg["algorithm"]["geometry_true_response_rejection"] = False

        class WorseningGeometryModel:
            width, height = cfg["domain"]
            dx = width/cfg["mesh"][0]
            dy = height/cfg["mesh"][1]

            @staticmethod
            def compliance_gradient(ribs, thicknesses, result):
                return -np.ones(len(ribs))

            @staticmethod
            def geometry_gradient(ribs, thicknesses, result, step):
                return np.zeros((len(ribs), 4))

            @staticmethod
            def analyze(ribs, thicknesses):
                return SimpleNamespace(compliance=1.1)

        rib = Rib((0.0, 0.0), (10.0, 10.0), 2.0, "A")
        optimizer = RibLayoutOptimizer(WorseningGeometryModel(), cfg)
        _, _, result = optimizer.optimize_geometry(
            [rib], np.array([0.2]), SimpleNamespace(compliance=1.0)
        )
        self.assertEqual(result.compliance, 1.0)
        self.assertTrue(any("best feasible" in line for line in optimizer.log))

    def test_geometry_first_equal_compliance_feasible_trial_becomes_incumbent(self):
        cfg = load_case(1, quick=True)
        cfg["volume_bound"] = 1.0
        cfg["algorithm"]["geometry_max_iterations"] = 1

        class EqualResponseModel:
            width, height = cfg["domain"]
            dx = width/cfg["mesh"][0]
            dy = height/cfg["mesh"][1]

            @staticmethod
            def compliance_gradient(ribs, thicknesses, result):
                return -np.ones(len(ribs))

            @staticmethod
            def geometry_gradient(ribs, thicknesses, result, step):
                return np.zeros((len(ribs), 4))

            @staticmethod
            def analyze(ribs, thicknesses):
                return SimpleNamespace(compliance=1.0)

        rib = Rib((0.0, 0.0), (10.0, 10.0), 2.0, "A")
        optimizer = RibLayoutOptimizer(EqualResponseModel(), cfg)
        history = []
        with patch(
            "rib_layout_algorithms.optimization.solve_geometry_convex_subproblem",
            side_effect=lambda *args: np.asarray(args[0], float).copy(),
        ):
            _, final_t, result = optimizer.optimize_geometry(
                [rib], np.array([0.2]), SimpleNamespace(compliance=1.0),
                iteration_history=history,
            )
        self.assertEqual(result.compliance, 1.0)
        self.assertLessEqual(optimizer.volume([rib], final_t), 1.001)
        self.assertEqual(history[0]["best_feasible_outer"], 1)
        self.assertTrue(history[0]["is_best_feasible"])

    def test_geometry_backtracking_failure_has_explicit_termination_reason(self):
        cfg = load_case(1, quick=True)
        cfg["algorithm"]["geometry_max_iterations"] = 1
        cfg["algorithm"]["geometry_true_response_max_retries"] = 1

        class AlwaysWorseModel:
            width, height = cfg["domain"]
            dx = width/cfg["mesh"][0]
            dy = height/cfg["mesh"][1]

            @staticmethod
            def compliance_gradient(ribs, thicknesses, result):
                return -np.ones(len(ribs))

            @staticmethod
            def geometry_gradient(ribs, thicknesses, result, step):
                return np.zeros((len(ribs), 4))

            @staticmethod
            def analyze(ribs, thicknesses):
                return SimpleNamespace(compliance=1.2)

        rib = Rib((0.0, 0.0), (10.0, 10.0), 2.0, "A")
        optimizer = RibLayoutOptimizer(AlwaysWorseModel(), cfg)
        history = []
        optimizer.optimize_geometry(
            [rib], np.array([0.2]), SimpleNamespace(compliance=1.0),
            iteration_history=history,
        )
        self.assertEqual(
            optimizer.geometry_termination_reason,
            "true_response_backtracking_failed",
        )
        self.assertEqual(
            history[-1]["termination_reason"],
            "true_response_backtracking_failed",
        )
        self.assertFalse(history[-1]["accepted"])

    def test_near_endpoint_grid_intersection_does_not_create_shell_sliver(self):
        model = ShellStiffenedPlateModel(
            1.0, 1.0, 2, 2, 0.1, 70000.0, 0.3,
            [{"forces": [{"point": [1.0, 0.0], "value": [0.0, -1.0, 0.0]}]}],
            {"type": "edge", "edge": "left"},
            interface_subdivisions_per_cell=4,
        )
        rib = Rib((0.5-1.0e-6, 0.5), (1.0, 0.0), 0.2, "near_grid", 1)
        points = model.rib_bottom_points(rib)
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        self.assertGreater(segment_lengths.min(), 9.0e-4)

    def test_frame_is_symmetric_positive_semidefinite(self):
        k = frame_global_stiffness(np.array([0., 0., 0.]), np.array([2., 1., 0.]), 1000., 0.3, 2., 3., 4., 1.)
        self.assertTrue(np.allclose(k, k.T, atol=1e-10))
        self.assertGreaterEqual(np.linalg.eigvalsh(k).min(), -1e-9)

    def test_case_counts_and_volume(self):
        for number, count in ((1, 8), (2, 16), (3, 64), (4, 64)):
            cfg = load_case(number, quick=True)
            ribs = initial_ribs(cfg)
            expected_length = cfg["initial_rib_cell_size"] * np.sqrt(2.0)
            self.assertEqual(len(ribs), count)
            for rib in ribs:
                delta = np.subtract(rib.p1, rib.p0)
                self.assertAlmostEqual(rib.length, expected_length, places=10)
                self.assertAlmostEqual(abs(delta[0]), abs(delta[1]), places=10)
            self.assertTrue(candidate_ribs(cfg))

    def test_case_mirror_symmetry_and_generated_sets_are_closed(self):
        expected_axes = {1: (), 2: ("x",), 3: ("y",), 4: ("y",)}
        for number, expected in expected_axes.items():
            cfg = load_case(number, quick=True)
            axes = mirror_axes(cfg)
            self.assertEqual(axes, expected)
            self.assertFalse(missing_mirror_partners(
                initial_ribs(cfg), axes, *cfg["domain"]
            ))
            self.assertFalse(missing_mirror_partners(
                candidate_ribs(cfg), axes, *cfg["domain"]
            ))

    def test_addition_completes_a_ranked_candidate_with_its_mirror(self):
        cfg = load_case(2, quick=True)
        left = Rib((0.0, 0.0), (10.0, 10.0), 2.0, "left")
        right = Rib((40.0, 0.0), (30.0, 10.0), 2.0, "right")
        second = Rib((10.0, 0.0), (30.0, 0.0), 2.0, "second")
        scores = {"left": 10.0, "second": 9.0, "right": 1.0}
        optimizer = RibLayoutOptimizer(
            SimpleNamespace(
                candidate_efficiency=lambda rib, thickness, result: scores[rib.name]
            ),
            cfg,
        )
        _, chosen = optimizer._select_addition_candidates(
            [right, second, left], [], SimpleNamespace(), limit=2
        )
        self.assertEqual([rib.name for rib in chosen], ["left", "right"])

    def test_second_seed_may_add_a_mirror_partner_for_three_rib_batch(self):
        cfg = load_case(2, quick=True)
        self_symmetric = Rib(
            (10.0, 0.0), (30.0, 0.0), 2.0, "self_symmetric"
        )
        left = Rib((0.0, 0.0), (10.0, 10.0), 2.0, "left")
        right = Rib((40.0, 0.0), (30.0, 10.0), 2.0, "right")
        scores = {"self_symmetric": 10.0, "left": 9.0, "right": 8.0}
        optimizer = RibLayoutOptimizer(
            SimpleNamespace(
                candidate_efficiency=lambda rib, thickness, result: scores[rib.name]
            ),
            cfg,
        )
        _, chosen = optimizer._select_addition_candidates(
            [right, left, self_symmetric], [], SimpleNamespace(), limit=2
        )
        self.assertEqual(
            [rib.name for rib in chosen],
            ["self_symmetric", "left", "right"],
        )
        self.assertLessEqual(len(chosen), 3)

    def test_filtering_deletes_a_complete_mirror_group(self):
        cfg = load_case(2, quick=True)
        optimizer = RibLayoutOptimizer(
            SimpleNamespace(dx=1.0, dy=1.0), cfg
        )
        ribs = [
            Rib((0.0, 0.0), (10.0, 10.0), 2.0, "left"),
            Rib((40.0, 0.0), (30.0, 10.0), 2.0, "right"),
            Rib((20.0, 0.0), (20.0, 10.0), 2.0, "center"),
        ]
        thicknesses = np.array([0.001, 0.0015, 0.2])
        optimizer.size = lambda active_ribs, active_t: (
            np.asarray(active_t).copy(), SimpleNamespace(compliance=1.0)
        )
        filtered, _, _ = optimizer.filter(
            ribs, thicknesses, SimpleNamespace(compliance=1.0),
            threshold_ratios=[0.01],
        )
        self.assertEqual([rib.name for rib in filtered], ["center"])

    def test_filtering_keeps_a_mirror_group_if_one_member_is_significant(self):
        cfg = load_case(2, quick=True)
        optimizer = RibLayoutOptimizer(
            SimpleNamespace(dx=1.0, dy=1.0), cfg
        )
        ribs = [
            Rib((0.0, 0.0), (10.0, 10.0), 2.0, "left"),
            Rib((40.0, 0.0), (30.0, 10.0), 2.0, "right"),
            Rib((20.0, 0.0), (20.0, 10.0), 2.0, "center"),
        ]
        thicknesses = np.array([0.001, 0.2, 0.2])
        filtered, _, _ = optimizer.filter(
            ribs, thicknesses, SimpleNamespace(compliance=1.0),
            threshold_ratios=[0.01],
        )
        self.assertEqual(filtered, ribs)

    def test_mirror_variable_map_reduces_pairs_and_self_symmetric_ribs(self):
        cfg = load_case(2, quick=True)
        paired = [
            Rib((0.0, 0.0), (10.0, 10.0), 2.0, "left"),
            Rib((40.0, 0.0), (30.0, 10.0), 2.0, "right"),
        ]
        pair_map = build_mirror_variable_map(
            paired, mirror_axes(cfg), *cfg["domain"]
        )
        self.assertEqual(pair_map.thickness_count, 1)
        self.assertEqual(pair_map.coordinate_count, 4)
        self.assertTrue(np.array_equal(
            pair_map.expand_thicknesses([0.3]), [0.3, 0.3]
        ))

        on_axis = [Rib((20.0, 0.0), (20.0, 10.0), 2.0, "axis")]
        axis_map = build_mirror_variable_map(
            on_axis, mirror_axes(cfg), *cfg["domain"]
        )
        self.assertEqual(axis_map.coordinate_count, 2)
        coordinates = axis_map.expand_coordinates([2.0, 8.0])
        self.assertTrue(np.allclose(coordinates, [20.0, 2.0, 20.0, 8.0]))

    def test_continuous_optimizers_keep_reduced_mirror_variables_exact(self):
        cfg = load_case(2, quick=True)
        cfg["volume_bound"] = 100.0
        cfg["algorithm"]["rationalization_max_iterations"] = 1

        class MirrorModel:
            width, height = cfg["domain"]
            dx = dy = 1.0

            @staticmethod
            def analyze(ribs, thicknesses):
                return SimpleNamespace(compliance=1.0)

            @staticmethod
            def compliance_gradient(ribs, thicknesses, result):
                return np.array([-1.0, -2.0])

            @staticmethod
            def geometry_gradient(ribs, thicknesses, result, step):
                return np.array([
                    [-0.10, 0.05, -0.02, 0.03],
                    [0.07, -0.04, 0.08, -0.06],
                ])

        ribs = [
            Rib((0.0, 0.0), (10.0, 10.0), 2.0, "left"),
            Rib((40.0, 0.0), (30.0, 10.0), 2.0, "right"),
        ]
        optimizer = RibLayoutOptimizer(MirrorModel(), cfg)
        thicknesses, sizing_result = optimizer.size(
            ribs, [0.2, 0.4], maxiter=1
        )
        moved, moved_t, geometry_result = optimizer.optimize_geometry(
            ribs, thicknesses, sizing_result, max_iterations_override=1
        )
        rationalized, rationalized_t, _, _ = (
            optimizer._solve_rationalization_eq18(
                moved, moved_t, geometry_result, 1.05, 0.2
            )
        )

        for stage_ribs, stage_t in (
            (ribs, thicknesses),
            (moved, moved_t),
            (rationalized, rationalized_t),
        ):
            variable_map = build_mirror_variable_map(
                stage_ribs, mirror_axes(cfg), *cfg["domain"]
            )
            coordinates = np.array([
                [*rib.p0, *rib.p1] for rib in stage_ribs
            ]).ravel()
            self.assertTrue(np.allclose(
                stage_t,
                variable_map.expand_thicknesses(
                    variable_map.reduce_thicknesses(stage_t)
                ),
            ))
            self.assertTrue(np.allclose(
                coordinates,
                variable_map.expand_coordinates(
                    variable_map.reduce_coordinates(coordinates)
                ),
            ))

    def test_case_specific_rib_heights(self):
        self.assertEqual(load_case(1)["rib"]["height"], 2.0)
        self.assertEqual(load_case(2)["rib"]["height"], 2.0)
        self.assertEqual(load_case(3)["rib"]["height"], 10.0)
        self.assertEqual(load_case(4)["rib"]["height"], 10.0)
        self.assertEqual(load_case(3)["rib"]["upper"], 3.0)
        self.assertEqual(load_case(4)["rib"]["upper"], 3.0)
        for number in (1, 2, 3, 4):
            self.assertEqual(
                load_case(number)["rationalization_relaxation"], 0.05
            )
            self.assertNotIn("extra_relaxation", load_case(number))
        for number in (1, 2):
            self.assertEqual(
                rationalization_paths(number, load_case(number)),
                ((0.05, "5pct"),),
            )
            self.assertNotIn(
                "further_rationalization_relaxation", load_case(number)
            )
        for number in (3, 4):
            self.assertEqual(
                load_case(number)["further_rationalization_relaxation"],
                0.05,
            )
            self.assertEqual(
                rationalization_paths(number, load_case(number)),
                ((0.05, "5pct"), (0.05, "further_5pct")),
            )

    def test_example1_six_panel_workflow_uses_requested_stages(self):
        rib = Rib((0.0, 0.0), (10.0, 10.0), 2.0, "test")
        added_rib = Rib((0.0, 10.0), (10.0, 0.0), 2.0, "added")

        def stage(name: str, ribs=None) -> Stage:
            ribs = [rib] if ribs is None else ribs
            return Stage(name, ribs, np.full(len(ribs), 0.2), 1.0, 1)

        initial = stage("initial_sizing")
        geometry = stage("geometry")
        first_filter = stage("filtering_converged_0")
        addition = stage(
            "member_addition_sizing_round_1", [rib, added_rib]
        )
        later_filter = stage("post_addition_filtering_round_1")
        panels = example1_detailed_panels(
            [initial, stage("adaptive"), geometry, stage("rationalized")],
            [first_filter, addition, later_filter],
        )
        self.assertEqual(len(panels), 6)
        self.assertIsNone(panels[0][1])
        self.assertEqual(
            [panel[1].name for panel in panels[1:]],
            [
                "initial_sizing",
                "filtering_converged_0",
                "member_addition_sizing_round_1",
                "post_addition_filtering_round_1",
                "geometry",
            ],
        )

    def test_examples3_and4_append_further_rationalization_panel(self):
        rib = Rib((0.0, 0.0), (10.0, 10.0), 2.0, "test")

        def stage(name: str) -> Stage:
            return Stage(name, [rib], np.array([0.2]), 1.0, 1)

        stages = [
            stage("initial_sizing"),
            stage("adaptive"),
            stage("geometry"),
            stage("rationalized"),
            stage("further_rationalized"),
        ]
        for number in (3, 4):
            panels = paper_style_panels(stages, [], {"number": number})
            self.assertEqual(len(panels), 6)
            self.assertEqual(panels[-1][0], "(f) After further rationalization")
            self.assertEqual(panels[-1][1].name, "further_rationalized")

    def test_composite_panel_marker_is_large_centered_and_below_panel(self):
        figure = plt.figure()
        try:
            axis = figure.add_subplot(111, projection="3d")
            axis.set_title("old title")
            caption = _place_panel_caption(axis, "(a)")
            self.assertEqual(axis.get_title(), "")
            self.assertEqual(caption.get_position()[0], 0.5)
            self.assertGreater(caption.get_position()[1], 0.0)
            self.assertLess(caption.get_position()[1], 0.2)
            self.assertEqual(caption.get_horizontalalignment(), "center")
            self.assertFalse(caption.get_clip_on())
            self.assertEqual(caption.get_text(), "(a)")
            self.assertGreaterEqual(caption.get_fontsize(), 20)
            self.assertEqual(caption.get_fontweight(), "bold")
        finally:
            plt.close(figure)

    def test_composite_axis_can_hide_all_annotations(self):
        figure = plt.figure()
        try:
            axis = figure.add_subplot(111, projection="3d")
            _format_3d_axis(
                axis,
                load_case(1),
                "unused",
                show_axis_annotations=False,
            )
            self.assertFalse(axis.axison)
        finally:
            plt.close(figure)

    def test_edge_load_labels_are_inset_inside_plot_domain(self):
        cfg = load_case(4)
        width, height = cfg["domain"]
        for point in ([0.0, 0.0], [0.0, height]):
            label = _load_label_position(point, cfg)
            self.assertGreater(label[0], 0.0)
            self.assertLess(label[0], width)
            self.assertGreater(label[1], 0.0)
            self.assertLess(label[1], height)
            self.assertGreater(label[2], 0.0)

    def test_further_rationalization_uses_first_rationalized_design(self):
        optimizer = object.__new__(RibLayoutOptimizer)
        optimizer.cfg = {
            "rationalization_relaxation": 0.05,
            "further_rationalization_relaxation": 0.05,
            "algorithm": {
                "geometry_move_limit_initial": 0.25,
            },
        }
        optimizer.analysis_count = 0
        optimizer.log = []
        optimizer.active_history = []
        optimizer.rationalization_history = []
        calls = []
        base_rib = Rib((0.0, 0.0), (1.0, 1.0), 1.0, "base", 1)

        def result(compliance):
            return SimpleNamespace(compliance=compliance)

        def size(self, ribs):
            self.analysis_count += 2
            return np.full(len(ribs), 0.2), result(10.0)

        def adapt(self, ribs, thicknesses, current, candidates):
            self.analysis_count += 3
            return ribs, thicknesses, result(9.0)

        geometry_move_steps = []

        def optimize_geometry(
            self, ribs, thicknesses, current, *, initial_move_step=None
        ):
            geometry_move_steps.append(initial_move_step)
            self.analysis_count += 4
            return ribs, thicknesses, result(8.0)

        def rationalize(self, ribs, thicknesses, current, relaxation):
            calls.append((tuple(rib.name for rib in ribs), current.compliance))
            self.analysis_count += 5
            pass_number = len(calls)
            new_rib = Rib(
                (0.0, 0.0), (1.0, 1.0), 1.0,
                f"rationalized_{pass_number}", 1,
            )
            self.rationalization_history = [{
                "event": "test_pass",
                "input_names": list(calls[-1][0]),
            }]
            return [new_rib], np.array([0.2]), result(8.0 + pass_number)

        optimizer.size = MethodType(size, optimizer)
        optimizer.adapt = MethodType(adapt, optimizer)
        optimizer.optimize_geometry = MethodType(optimize_geometry, optimizer)
        optimizer.rationalize = MethodType(rationalize, optimizer)

        run = optimizer.run([base_rib], [])

        self.assertEqual(
            [stage.name for stage in run.stages],
            [
                "initial_sizing", "adaptive", "geometry", "rationalized",
                "further_rationalized",
            ],
        )
        self.assertEqual(calls[0], (("base",), 8.0))
        self.assertEqual(calls[1], (("rationalized_1",), 9.0))
        self.assertEqual(geometry_move_steps, [0.25])
        self.assertEqual(run.stages[-1].analyses, 5)
        self.assertEqual(
            set(run.rationalization_histories),
            {"rationalized", "further_rationalized"},
        )
        self.assertEqual(
            [event["pass_name"] for event in optimizer.rationalization_history],
            ["rationalized", "further_rationalized"],
        )

    def test_example2_detailed_timeline_includes_switches_and_restoration(self):
        rib=Rib((0.0,0.0),(10.0,10.0),2.0,"test")

        def stage(name,compliance=1.0):
            return Stage(name,[rib],np.array([0.2]),compliance,1)

        active=[
            stage("filtering_converged_0"),
            stage("member_addition_sizing_round_1",0.8),
            stage("post_addition_filtering_round_1",0.81),
            stage("member_addition_sizing_round_2_rejected",0.805),
        ]
        timeline=example2_detailed_timeline(
            [
                stage("initial_sizing",1.2),stage("adaptive",0.81),
                stage("geometry",0.7),stage("rationalized",0.72),
            ],
            active,
        )
        self.assertEqual(
            [item[1].name for item in timeline],
            [
                "initial_sizing","filtering_converged_0",
                "member_addition_sizing_round_1",
                "post_addition_filtering_round_1",
                "member_addition_sizing_round_2_rejected","adaptive",
                "geometry","rationalized",
            ],
        )
        self.assertIn("restored",timeline[5][0].lower())

    def test_global_addition_limit_and_dynamic_move_limits(self):
        for number in (1, 2, 3, 4):
            self.assertEqual(load_case(number)["algorithm"]["additions_per_iteration"], 2)
        for number in (1, 2, 3, 4):
            algorithm = load_case(number)["algorithm"]
            self.assertEqual(load_case(number)["linear_solver_threads"], 1)
            self.assertEqual(
                algorithm["filter_threshold_ratios"],
                [0.10, 0.01, 0.001, 0.0001, 0.00001],
            )
            self.assertEqual(algorithm["short_rib_shell_cells"], 3.0)
            self.assertEqual(algorithm["short_rib_cell_fraction"], 0.25)
            self.assertEqual(algorithm["short_rib_thickness_factor"], 5.0)
            self.assertNotIn("geometry_move_fraction", algorithm)
            self.assertNotIn("rationalization_move_fraction", algorithm)
            self.assertNotIn("rationalization_geometry_move_limit_initial", algorithm)
            self.assertEqual(algorithm["addition_sizing_improvement_min"], 0.01)
            self.assertEqual(algorithm["active_cycle_improvement_min"], 0.01)
            self.assertEqual(algorithm["filter_tolerance"], 0.01)
            self.assertEqual(algorithm["addition_factor_min_ratio"], 0.70)
            self.assertNotIn("addition_second_factor_min_ratio", algorithm)
            self.assertNotIn("addition_collinear_factor_tolerance", algorithm)
            self.assertEqual(algorithm["rationalization_beta"], 10.0)
            self.assertNotIn("rationalization_reference_quantile_base", algorithm)
            self.assertNotIn(
                "rationalization_reference_quantile_relaxation_factor", algorithm
            )
            self.assertEqual(
                algorithm["rationalization_compliance_tolerance"], 0.001
            )
            expected_geometry_step = 0.05 if number == 2 else 0.50
            expected_rationalization_step = 0.05 if number == 2 else 0.50
            self.assertEqual(
                algorithm["geometry_move_limit_initial"], expected_geometry_step
            )
            self.assertEqual(
                algorithm["rationalization_move_limit_initial"],
                expected_rationalization_step,
            )
            self.assertEqual(algorithm["rationalization_min_iterations"], 11)
            self.assertEqual(
                algorithm["rationalization_dual_tolerance"], 1.0e-9
            )
            self.assertEqual(algorithm["sca_objective_tolerance"], 0.005)
            self.assertEqual(algorithm["sca_constraint_tolerance"], 0.001)
            self.assertEqual(algorithm["sca_design_tolerance"], 0.001)
            self.assertEqual(algorithm["sca_design_guard_tolerance"], 0.010)
            self.assertEqual(
                algorithm["sca_consecutive_convergence_steps"], 2
            )
            self.assertGreaterEqual(algorithm["sizing_max_iterations"], 60)
            self.assertEqual(algorithm["move_limit_initial"], 0.5)
            self.assertEqual(algorithm["move_limit_direction_increase"], 1.2)
            self.assertEqual(algorithm["move_limit_direction_decrease"], 0.7)
            self.assertEqual(
                algorithm["move_limit_direction_zero_tolerance"], 1.0e-6
            )
            self.assertEqual(algorithm["move_limit_unsuccessful_decrease"], 0.75)
            self.assertTrue(algorithm["geometry_true_response_rejection"])
            self.assertEqual(
                algorithm["geometry_true_response_worsening_tolerance"], 1.0e-4
            )
            self.assertEqual(algorithm["geometry_true_response_max_retries"], 4)
            self.assertEqual(algorithm["rationalization_beta_initial"], 1.0)
            self.assertEqual(algorithm["rationalization_beta_increment"], 1.0)

    def test_geometry_and_rationalization_initial_move_steps_must_be_positive(self):
        for name in (
            "geometry_move_limit_initial",
            "rationalization_move_limit_initial",
        ):
            cfg = load_case(1, quick=True)
            cfg["algorithm"][name] = 0.0
            with self.assertRaisesRegex(ValueError, f"{name} must be positive"):
                RibLayoutOptimizer(object(), cfg).run([], [])

    def test_sca_convergence_requires_one_percent_design_guard(self):
        # A flat objective alone cannot stop an SCA stage while design
        # variables still change by more than 1.0%.
        self.assertFalse(sca_step_converged(
            True, 0.0005, 0.011, 0.005, 0.001, 0.010
        ))
        self.assertTrue(sca_step_converged(
            True, 0.0049, 0.0099, 0.005, 0.001, 0.010
        ))
        self.assertFalse(sca_step_converged(
            True, 0.0051, 0.0099, 0.005, 0.001, 0.010
        ))
        # The original design-change branch remains valid inside the guard.
        self.assertTrue(sca_step_converged(
            True, 0.02, 0.0009, 0.005, 0.001, 0.010
        ))
        self.assertFalse(sca_step_converged(
            False, 0.0, 0.0, 0.005, 0.001, 0.010
        ))

    def test_rationalization_tref_quantile_uses_rib_count_and_relaxation(self):
        optimizer = RibLayoutOptimizer(SimpleNamespace(), load_case(3, quick=True))
        self.assertAlmostEqual(
            optimizer.rationalization_reference_quantile(0.02, 20), 0.07
        )
        self.assertAlmostEqual(
            optimizer.rationalization_reference_quantile(0.05, 20), 0.10
        )
        self.assertAlmostEqual(
            optimizer.rationalization_reference_quantile(0.10, 20), 0.15
        )
        self.assertAlmostEqual(
            optimizer.rationalization_reference_quantile(0.05, 5), 0.25
        )
        with self.assertRaises(ValueError):
            optimizer.rationalization_reference_quantile(0.95, 20)
        with self.assertRaises(ValueError):
            optimizer.rationalization_reference_quantile(0.05, 0)

    def test_enhanced_mma_move_limit_direction_and_contraction(self):
        move = EnhancedMMAMoveLimit(np.array([0.0, 0.0]), np.array([2.0, 10.0]))
        lower, upper = move.update(np.array([0.2, 2.0]), 10.0)
        self.assertTrue(np.allclose(lower, [0.0, 1.0]))
        self.assertTrue(np.allclose(upper, [0.7, 3.0]))

        move.update(np.array([0.3, 2.5]), 9.0)
        move.update(np.array([0.4, 3.0]), 8.0)
        self.assertTrue(np.allclose(move.local_step, [1.2, 1.2]))
        lower, upper = move.update(np.array([0.35, 2.8]), 7.0)
        self.assertTrue(np.allclose(move.local_step, [0.84, 0.84]))
        self.assertTrue(np.allclose([lower[0], upper[0]], [0.0, 0.77]))
        move.contract()
        self.assertAlmostEqual(move.global_step, 0.375)

        near_zero = EnhancedMMAMoveLimit(
            np.array([0.0, 0.0]), np.array([2.0, 10.0]),
            direction_zero_tolerance=1.0e-6,
        )
        near_zero.update(np.array([0.2, 2.0]), 10.0)
        near_zero.update(np.array([0.3, 2.5]), 9.0)
        near_zero.update(np.array([0.3+1.0e-8, 3.0]), 8.0)
        # The first normalized move is numerical zero and remains unchanged;
        # the second variable has two significant same-direction moves.
        self.assertTrue(np.allclose(near_zero.local_step, [1.0, 1.2]))

        node_move = EnhancedMMAMoveLimit(
            np.array([0.0, 0.0]), np.array([20.0, 20.0]),
            step_scale=np.array([2.0, 4.0]),
        )
        lower, upper = node_move.update(np.array([10.0, 10.0]), 1.0)
        self.assertTrue(np.allclose(lower, [9.0, 8.0]))
        self.assertTrue(np.allclose(upper, [11.0, 12.0]))

    def test_geometry_coordinate_move_scale_uses_shell_cell_dimensions(self):
        model = SimpleNamespace(width=200.0, height=100.0, dx=2.5, dy=2.5)
        optimizer = RibLayoutOptimizer(model, load_case(3, quick=True))
        scale = optimizer._coordinate_move_step_scale(2)
        self.assertTrue(np.allclose(
            scale,
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        ))

        move = EnhancedMMAMoveLimit(
            np.zeros(4),
            np.array([200.0, 100.0, 200.0, 100.0]),
            step_scale=optimizer._coordinate_move_step_scale(1),
            initial_global_step=0.5,
        )
        lower, upper = move.update(np.array([100.0, 50.0, 100.0, 50.0]), 1.0)
        self.assertTrue(np.allclose(lower, [97.5, 47.5, 97.5, 47.5]))
        self.assertTrue(np.allclose(upper, [102.5, 52.5, 102.5, 52.5]))

    def test_geometry_invalid_move_locally_freezes_without_contracting_gstep(self):
        cfg = load_case(1, quick=True)
        cfg["algorithm"]["geometry_max_iterations"] = 1

        class LocalFreezeModel:
            width, height = cfg["domain"]
            dx = width/cfg["mesh"][0]
            dy = height/cfg["mesh"][1]

            @staticmethod
            def compliance_gradient(ribs, thicknesses, result):
                return -np.ones(len(ribs))

            @staticmethod
            def geometry_gradient(ribs, thicknesses, result, step):
                return np.zeros((len(ribs), 4))

            @staticmethod
            def analyze(ribs, thicknesses):
                return SimpleNamespace(compliance=1.0)

        ribs = [
            Rib((0.0, 0.0), (10.0, 0.0), 2.0, "A"),
            Rib((0.0, 2.0), (10.0, 2.0), 2.0, "B"),
        ]
        thicknesses = np.array([0.2, 0.2])
        optimizer = RibLayoutOptimizer(LocalFreezeModel(), cfg)
        calls = 0

        def invalid_then_valid(*args, **kwargs):
            nonlocal calls
            calls += 1
            current = np.asarray(args[0], float)
            lower = np.asarray(args[-2], float)
            upper = np.asarray(args[-1], float)
            candidate = current.copy()
            if calls == 1:
                # Move A onto stationary B to create a new overlap.
                candidate[2:6] = [0.0, 2.0, 10.0, 2.0]
            else:
                # Only A's four coordinate bounds are fixed. B and both
                # thickness variables retain nonzero move intervals.
                self.assertTrue(np.array_equal(lower[2:6], current[2:6]))
                self.assertTrue(np.array_equal(upper[2:6], current[2:6]))
                self.assertTrue(np.any(upper[6:10] > lower[6:10]))
            return candidate

        history = []
        with patch(
            "rib_layout_algorithms.optimization.solve_geometry_convex_subproblem",
            side_effect=invalid_then_valid,
        ):
            optimizer.optimize_geometry(
                ribs, thicknesses, SimpleNamespace(compliance=1.0),
                iteration_history=history,
            )
        self.assertEqual(calls, 2)
        self.assertEqual(history[0]["frozen_geometry_names"], ["A"])
        self.assertEqual(history[0]["move_global_used"], 0.5)
        self.assertTrue(any(
            "local freeze" in line and "A" in line for line in optimizer.log
        ))

    def test_filtering_repeats_until_no_rib_is_deleted(self):
        cfg = load_case(1, quick=True)
        optimizer = RibLayoutOptimizer(object(), cfg)
        ribs = initial_ribs(cfg)[:4]
        thicknesses = np.full(4, cfg["rib"]["initial"])
        result = type("Result", (), {"compliance": 1.0})()

        references = []

        def one_deletion_per_call(
            current_ribs,
            current_t,
            current_result,
            threshold_ratios=None,
            reference_compliance=None,
        ):
            references.append(reference_compliance)
            if len(current_ribs) > 2:
                return current_ribs[:-1], current_t[:-1], current_result
            return current_ribs, current_t, current_result

        optimizer.filter = one_deletion_per_call
        final_ribs, final_t, _, rounds = optimizer.filter_until_stable(ribs, thicknesses, result)
        self.assertEqual(len(final_ribs), 2)
        self.assertEqual(len(final_t), 2)
        self.assertEqual(rounds, 2)
        self.assertEqual(references, [1.0, 1.0, 1.0])

    def test_short_light_rib_enters_performance_validated_filtering(self):
        cfg = load_case(2, quick=True)
        model = SimpleNamespace(dx=1.0, dy=1.0)
        optimizer = RibLayoutOptimizer(model, cfg)
        long_rib = Rib((0.0, 0.0), (10.0, 0.0), 2.0, "long")
        short_rib = Rib((0.0, 1.0), (2.5, 1.0), 2.0, "short")
        thicknesses = np.array([0.2, 0.05])
        optimizer.size = lambda ribs, initial=None, maxiter=None: (
            np.asarray(initial, float).copy(), SimpleNamespace(compliance=100.0)
        )
        filtered_ribs, filtered_t, filtered_result = optimizer.filter(
            [long_rib, short_rib],
            thicknesses,
            SimpleNamespace(compliance=100.0),
            threshold_ratios=[0.1],
        )
        self.assertEqual([rib.name for rib in filtered_ribs], ["long"])
        self.assertTrue(np.allclose(filtered_t, [0.2]))
        self.assertEqual(filtered_result.compliance, 100.0)
        self.assertTrue(any("short_light=1" in line for line in optimizer.log))

    def test_subthreshold_member_addition_is_rejected_and_restored(self):
        cfg = load_case(1, quick=True)
        model = SimpleNamespace(
            candidate_efficiency=lambda *args: 1.0,
        )
        optimizer = RibLayoutOptimizer(model, cfg)
        active = [Rib((0.0, 0.0), (10.0, 10.0), 2.0, "active")]
        candidate = Rib((0.0, 0.0), (10.0, 0.0), 2.0, "candidate")
        thicknesses = np.array([0.2])
        current = SimpleNamespace(compliance=100.0)

        filter_calls = 0
        def stable_filter(ribs, values, result):
            nonlocal filter_calls
            filter_calls += 1
            return ribs, values, result, 0
        optimizer.filter_until_stable = stable_filter
        optimizer.size = lambda ribs, initial=None, maxiter=None: (
            np.full(len(ribs), 0.2), SimpleNamespace(compliance=99.8)
        )
        final_ribs, final_t, final_result = optimizer.adapt(active, thicknesses, current, [candidate])
        self.assertEqual([rib.name for rib in final_ribs], ["active"])
        self.assertTrue(np.array_equal(final_t, thicknesses))
        self.assertEqual(final_result.compliance, 100.0)
        self.assertTrue(optimizer.active_history[-1].name.endswith("_rejected"))
        self.assertEqual(filter_calls, 1)  # initial convergence only; no post-rejection filtering

    def test_full_active_cycle_below_one_percent_stops_before_next_addition(self):
        cfg = load_case(1, quick=True)
        cfg["algorithm"]["additions_per_iteration"] = 1
        scores = {"candidate_1": 2.0, "candidate_2": 1.0}
        optimizer = RibLayoutOptimizer(
            SimpleNamespace(
                candidate_efficiency=lambda rib, thickness, result: scores[rib.name]
            ),
            cfg,
        )
        active = [Rib((0.0, 0.0), (10.0, 10.0), 2.0, "active")]
        candidates = [
            Rib((0.0, 5.0), (10.0, 5.0), 2.0, "candidate_1"),
            Rib((0.0, 15.0), (10.0, 15.0), 2.0, "candidate_2"),
        ]
        filter_calls = 0

        def stable_filter(ribs, values, result):
            nonlocal filter_calls
            filter_calls += 1
            if filter_calls == 2:
                return ribs, values, SimpleNamespace(compliance=99.5), 0
            return ribs, values, result, 0

        optimizer.filter_until_stable = stable_filter
        optimizer.size = lambda ribs, initial=None, maxiter=None: (
            np.full(len(ribs), 0.2), SimpleNamespace(compliance=98.0)
        )
        final_ribs, _, final_result = optimizer.adapt(
            active,
            np.array([0.2]),
            SimpleNamespace(compliance=100.0),
            candidates,
        )
        self.assertEqual([rib.name for rib in final_ribs], ["active", "candidate_1"])
        self.assertEqual(final_result.compliance, 99.5)
        self.assertEqual(filter_calls, 2)
        self.assertTrue(any(
            "full addition/filtering cycle decrease 0.500% < 1.000%" in line
            for line in optimizer.log
        ))

    def test_adaptive_stops_when_filtering_retains_no_new_rib(self):
        cfg = load_case(1, quick=True)
        cfg["algorithm"]["additions_per_iteration"] = 1
        scores = {"candidate_1": 2.0, "candidate_2": 1.0}
        optimizer = RibLayoutOptimizer(
            SimpleNamespace(
                candidate_efficiency=lambda rib, thickness, result: scores[rib.name]
            ),
            cfg,
        )
        active = [Rib((0.0, 0.0), (10.0, 10.0), 2.0, "active")]
        candidates = [
            Rib((0.0, 5.0), (10.0, 5.0), 2.0, "candidate_1"),
            Rib((0.0, 15.0), (10.0, 15.0), 2.0, "candidate_2"),
        ]
        filter_calls = 0
        sized_sets = []

        def filtering(ribs, values, result):
            nonlocal filter_calls
            filter_calls += 1
            if filter_calls == 1:
                return ribs, values, result, 0
            # Remove every newly added rib while retaining the pre-addition set.
            return [ribs[0]], np.asarray(values[:1]), SimpleNamespace(
                compliance=80.0
            ), 1

        def sizing(ribs, initial=None, maxiter=None):
            sized_sets.append([rib.name for rib in ribs])
            return np.full(len(ribs), 0.2), SimpleNamespace(compliance=50.0)

        optimizer.filter_until_stable = filtering
        optimizer.size = sizing
        final_ribs, _, final_result = optimizer.adapt(
            active,
            np.array([0.2]),
            SimpleNamespace(compliance=100.0),
            candidates,
        )
        self.assertEqual([rib.name for rib in final_ribs], ["active"])
        self.assertEqual(final_result.compliance, 80.0)
        self.assertEqual(sized_sets, [["active", "candidate_1"]])
        self.assertEqual(filter_calls, 2)
        self.assertTrue(any(
            "no newly added rib retained" in line
            and "entering geometry optimization" in line
            for line in optimizer.log
        ))

    def test_member_addition_keeps_covering_longer_candidate(self):
        cfg = load_case(1, quick=True)
        half = Rib((0.0, 0.0), (10.0, 0.0), 2.0, "half")
        full = Rib((0.0, 0.0), (20.0, 0.0), 2.0, "full")
        third = Rib((0.0, 5.0), (20.0, 5.0), 2.0, "third")
        model = SimpleNamespace(
            candidate_efficiency=lambda rib, thickness, result: {
                "half": 3.0, "full": 2.4, "third": 1.0,
            }[rib.name],
        )
        optimizer = RibLayoutOptimizer(model, cfg)
        active = [Rib((0.0, 10.0), (10.0, 10.0), 2.0, "active")]
        filter_calls = 0
        def stable_filter(ribs, values, result):
            nonlocal filter_calls
            filter_calls += 1
            return ribs, values, result, 0
        optimizer.filter_until_stable = stable_filter
        optimizer.size = lambda ribs, initial=None, maxiter=None: (
            np.full(len(ribs), 0.2), SimpleNamespace(compliance=50.0)
        )
        final_ribs, _, _ = optimizer.adapt(
            active, np.array([0.2]), SimpleNamespace(compliance=100.0),
            [half, full, third],
        )
        self.assertNotIn("half", [rib.name for rib in final_ribs])
        self.assertIn("full", [rib.name for rib in final_ribs])
        self.assertNotIn("third", [rib.name for rib in final_ribs])
        self.assertEqual(filter_calls, 2)  # initial convergence plus accepted post-addition filtering

    def test_member_addition_ranks_occupied_locations_before_coverage_check(self):
        cfg = load_case(1, quick=True)
        existing = Rib((0.0, 0.0), (20.0, 0.0), 2.0, "existing")
        covered = Rib((5.0, 0.0), (15.0, 0.0), 2.0, "covered")
        valid = Rib((0.0, 5.0), (20.0, 5.0), 2.0, "valid")
        unused = Rib((0.0, 10.0), (20.0, 10.0), 2.0, "unused")
        scores = {"covered": 3.0, "valid": 2.4, "unused": 1.0}
        optimizer = RibLayoutOptimizer(
            SimpleNamespace(candidate_efficiency=lambda rib, thickness, result: scores[rib.name]),
            cfg,
        )
        inspected, chosen = optimizer._select_addition_candidates(
            [unused, valid, covered], [existing], SimpleNamespace(), limit=2
        )
        self.assertEqual(
            [rib.name for rib in inspected], ["covered", "valid", "unused"]
        )
        self.assertEqual([rib.name for rib in chosen], ["valid"])

    def test_partial_overlap_retains_higher_factor_candidate(self):
        cfg = load_case(1, quick=True)
        shorter = Rib((15.0, 0.0), (25.0, 0.0), 2.0, "shorter")
        longer = Rib((0.0, 0.0), (20.0, 0.0), 2.0, "longer")
        scores = {"shorter": 3.0, "longer": 2.4}
        optimizer = RibLayoutOptimizer(
            SimpleNamespace(candidate_efficiency=lambda rib, thickness, result: scores[rib.name]),
            cfg,
        )
        selected, chosen = optimizer._select_addition_candidates(
            [shorter, longer], [], SimpleNamespace(), limit=2
        )
        self.assertEqual([rib.name for rib in selected], ["shorter", "longer"])
        self.assertEqual([rib.name for rib in chosen], ["shorter"])

    def test_candidate_efficiency_is_frozen_energy_per_added_volume(self):
        rib = Rib((0.0, 0.0), (10.0, 0.0), 2.0, "candidate")
        for model_class in (StiffenedPlateModel, ShellStiffenedPlateModel):
            with self.subTest(model=model_class.__name__):
                model = object.__new__(model_class)
                model.candidate_stiffness_energy = (
                    lambda candidate, thickness, result: 24.0
                )
                factor = model.candidate_efficiency(
                    rib, 0.3, SimpleNamespace()
                )
                self.assertEqual(factor, 4.0)

    def test_direct_candidate_ranking_scores_every_candidate(self):
        cfg = load_case(1, quick=True)
        candidates = [
            Rib((0.0, 0.0), (10.0, 0.0), 1.0, "C1"),
            Rib((0.0, 5.0), (10.0, 5.0), 1.0, "C2"),
            Rib((0.0, 10.0), (10.0, 10.0), 1.0, "C3"),
        ]
        scores = {"C1": 10.0, "C2": 8.5, "C3": 100.0}
        scored_names = []

        def efficiency(rib, thickness, result):
            scored_names.append(rib.name)
            return scores[rib.name]

        optimizer = RibLayoutOptimizer(
            SimpleNamespace(candidate_efficiency=efficiency), cfg
        )
        inspected, chosen = optimizer._select_addition_candidates(
            candidates, [], SimpleNamespace(), limit=2
        )
        self.assertCountEqual(scored_names, ["C1", "C2", "C3"])
        self.assertEqual([rib.name for rib in inspected], ["C3", "C1"])
        self.assertEqual([rib.name for rib in chosen], ["C3"])
        self.assertFalse(any("shortlist" in line for line in optimizer.log))

    def test_pair_overlap_uses_length_only_for_equal_factor_tie(self):
        cfg = load_case(1, quick=True)
        shorter = Rib((0.0, 0.0), (10.0, 0.0), 2.0, "shorter")
        longer = Rib((0.0, 0.0), (20.0, 0.0), 2.0, "longer")
        optimizer = RibLayoutOptimizer(
            SimpleNamespace(candidate_efficiency=lambda *args: 2.0), cfg
        )
        _, chosen = optimizer._select_addition_candidates(
            [shorter, longer], [], SimpleNamespace(), limit=2
        )
        self.assertEqual([rib.name for rib in chosen], ["longer"])

    def test_full_coverage_prefers_longer_rib_above_global_factor_floor(self):
        cfg = load_case(1, quick=True)
        shorter = Rib((0.0, 0.0), (10.0, 0.0), 2.0, "shorter")
        longer = Rib((0.0, 0.0), (20.0, 0.0), 2.0, "longer")
        scores = {"shorter": 10.0, "longer": 8.0}
        optimizer = RibLayoutOptimizer(
            SimpleNamespace(
                candidate_efficiency=lambda rib, thickness, result: scores[rib.name]
            ),
            cfg,
        )
        _, chosen = optimizer._select_addition_candidates(
            [shorter, longer], [], SimpleNamespace(), limit=2
        )
        self.assertEqual([rib.name for rib in chosen], ["longer"])
        self.assertTrue(any(
            "preferred for length" in line
            and "completely covers shorter rib" in line
            and "regardless of score difference=20.000%" in line
            for line in optimizer.log
        ))

    def test_example1_front_candidates_select_covering_c2(self):
        cfg = load_case(1, quick=True)
        candidates = [
            Rib((0.0, 0.0), (10.0, 0.0), 2.0, "C1"),
            Rib((0.0, 0.0), (20.0, 0.0), 2.0, "C2"),
            Rib((10.0, 0.0), (20.0, 0.0), 2.0, "C9"),
        ]
        scores = {
            "C1": 0.003310326,
            "C2": 0.002798377,
            "C9": 0.002594398,
        }
        optimizer = RibLayoutOptimizer(
            SimpleNamespace(
                candidate_efficiency=lambda rib, thickness, result: scores[rib.name]
            ),
            cfg,
        )
        _, chosen = optimizer._select_addition_candidates(
            candidates, [], SimpleNamespace(), limit=2
        )
        self.assertEqual([rib.name for rib in chosen], ["C2"])

    def test_direct_stiffness_per_volume_ranking_applies_full_coverage_rule(self):
        cfg = load_case(1, quick=True)
        candidates = [
            Rib((0.0, 0.0), (10.0, 0.0), 2.0, "C1"),
            Rib((0.0, 0.0), (20.0, 0.0), 2.0, "C2"),
            Rib((10.0, 0.0), (20.0, 0.0), 2.0, "C9"),
        ]
        scores = {
            "C1": 6.134203,
            "C2": 6.132447,
            "C9": 4.086556,
        }
        model = SimpleNamespace(candidate_efficiency=(
            lambda rib, thickness, result: scores[rib.name]
        ))
        optimizer = RibLayoutOptimizer(model, cfg)
        _, chosen = optimizer._select_addition_candidates(
            candidates, [], SimpleNamespace(), limit=2
        )
        self.assertEqual([rib.name for rib in chosen], ["C2"])

    def test_length_preference_does_not_apply_to_noncollinear_candidates(self):
        cfg = load_case(1, quick=True)
        shorter = Rib((0.0, 0.0), (10.0, 0.0), 2.0, "shorter")
        longer = Rib((0.0, 5.0), (20.0, 5.0), 2.0, "longer")
        scores = {"shorter": 10.0, "longer": 9.6}
        optimizer = RibLayoutOptimizer(
            SimpleNamespace(
                candidate_efficiency=lambda rib, thickness, result: scores[rib.name]
            ),
            cfg,
        )
        _, chosen = optimizer._select_addition_candidates(
            [shorter, longer], [], SimpleNamespace(), limit=1
        )
        self.assertEqual([rib.name for rib in chosen], ["shorter"])

    def test_long_candidate_replaces_fully_covered_existing_short_rib(self):
        cfg = load_case(1, quick=True)
        short = Rib((0.0, 0.0), (10.0, 0.0), 2.0, "short")
        other = Rib((0.0, 10.0), (10.0, 20.0), 2.0, "other")
        long = Rib((0.0, 0.0), (20.0, 0.0), 2.0, "long")
        weak = Rib((0.0, 5.0), (20.0, 5.0), 2.0, "weak")
        scores = {"long": 10.0, "weak": 5.0}
        optimizer = RibLayoutOptimizer(
            SimpleNamespace(
                candidate_efficiency=lambda rib, thickness, result: scores[rib.name]
            ),
            cfg,
        )
        optimizer.filter_until_stable = lambda ribs, values, result: (
            ribs, values, result, 0
        )
        sizing_starts = []

        def sizing(active_ribs, initial=None, maxiter=None):
            sizing_starts.append(
                ([rib.name for rib in active_ribs], np.asarray(initial).copy())
            )
            return np.asarray(initial).copy(), SimpleNamespace(compliance=50.0)

        optimizer.size = sizing
        final_ribs, final_t, _ = optimizer.adapt(
            [short, other], np.array([0.3, 0.4]),
            SimpleNamespace(compliance=100.0), [weak, long],
        )
        self.assertEqual([rib.name for rib in final_ribs], ["other", "long"])
        self.assertTrue(np.allclose(final_t, [0.4, cfg["rib"]["initial"]]))
        self.assertEqual(sizing_starts[0][0], ["other", "long"])
        self.assertTrue(np.allclose(sizing_starts[0][1], [0.4, 0.2]))
        self.assertTrue(any(
            "removed fully covered existing ribs=['short']" in line
            for line in optimizer.log
        ))

    def test_covered_top_candidates_are_skipped_and_lower_candidate_backfills(self):
        cfg = load_case(1, quick=True)
        existing_a = Rib((0.0, 0.0), (20.0, 0.0), 2.0, "existing_a")
        existing_b = Rib((0.0, 10.0), (20.0, 10.0), 2.0, "existing_b")
        covered_a = Rib((5.0, 0.0), (15.0, 0.0), 2.0, "covered_a")
        covered_b = Rib((5.0, 10.0), (15.0, 10.0), 2.0, "covered_b")
        lower_ranked = Rib((0.0, 5.0), (20.0, 5.0), 2.0, "lower_ranked")
        scores = {"covered_a": 3.0, "covered_b": 2.8, "lower_ranked": 2.2}
        optimizer = RibLayoutOptimizer(
            SimpleNamespace(candidate_efficiency=lambda rib, thickness, result: scores[rib.name]),
            cfg,
        )
        inspected, chosen = optimizer._select_addition_candidates(
            [lower_ranked, covered_b, covered_a],
            [existing_a, existing_b], SimpleNamespace(), limit=2,
        )
        self.assertEqual(
            [rib.name for rib in inspected], ["covered_a", "covered_b", "lower_ranked"]
        )
        self.assertEqual([rib.name for rib in chosen], ["lower_ranked"])

    def test_factor_floor_uses_maximum_uncovered_eligible_candidate(self):
        cfg = load_case(1, quick=True)
        existing = Rib((0.0, 0.0), (20.0, 0.0), 2.0, "existing")
        covered_maximum = Rib((5.0, 0.0), (15.0, 0.0), 2.0, "covered_max")
        first_selected = Rib((0.0, 5.0), (20.0, 5.0), 2.0, "selected")
        below_global_floor = Rib((0.0, 10.0), (20.0, 10.0), 2.0, "below")
        scores = {"covered_max": 10.0, "selected": 8.0, "below": 6.5}
        optimizer = RibLayoutOptimizer(
            SimpleNamespace(
                candidate_efficiency=lambda rib, thickness, result: scores[rib.name]
            ),
            cfg,
        )
        inspected, chosen = optimizer._select_addition_candidates(
            [below_global_floor, first_selected, covered_maximum],
            [existing],
            SimpleNamespace(),
            limit=2,
        )
        self.assertEqual(
            [rib.name for rib in inspected], ["covered_max", "selected", "below"]
        )
        self.assertEqual([rib.name for rib in chosen], ["selected", "below"])
        self.assertTrue(any(
            "eligible=2 of 3, maximum=8" in line
            for line in optimizer.log
        ))

    def test_candidate_covered_by_collinear_active_union_is_skipped(self):
        cfg = load_case(1, quick=True)
        left_half = Rib((0.0, 20.0), (10.0, 10.0), 2.0, "D6-")
        right_half = Rib((10.0, 10.0), (20.0, 0.0), 2.0, "D4-")
        combined = Rib((0.0, 20.0), (20.0, 0.0), 2.0, "C31")
        valid = Rib((10.0, 0.0), (10.0, 20.0), 2.0, "C11")
        scores = {"C31": 10.0, "C11": 8.0}
        optimizer = RibLayoutOptimizer(
            SimpleNamespace(
                candidate_efficiency=lambda rib, thickness, result: scores[rib.name]
            ),
            cfg,
        )
        inspected, chosen = optimizer._select_addition_candidates(
            [valid, combined], [left_half, right_half], SimpleNamespace(), limit=1
        )
        self.assertEqual([rib.name for rib in inspected], ["C31", "C11"])
        self.assertEqual([rib.name for rib in chosen], ["C11"])
        self.assertTrue(any(
            "C31 is fully covered by union of existing ribs" in line
            and "D6-" in line and "D4-" in line
            for line in optimizer.log
        ))

    def test_gap_in_collinear_active_union_does_not_cover_candidate(self):
        cfg = load_case(1, quick=True)
        left = Rib((0.0, 0.0), (9.0, 0.0), 2.0, "left")
        right = Rib((11.0, 0.0), (20.0, 0.0), 2.0, "right")
        candidate = Rib((0.0, 0.0), (20.0, 0.0), 2.0, "candidate")
        optimizer = RibLayoutOptimizer(
            SimpleNamespace(candidate_efficiency=lambda *args: 1.0), cfg
        )
        _, chosen = optimizer._select_addition_candidates(
            [candidate], [left, right], SimpleNamespace(), limit=1
        )
        self.assertEqual([rib.name for rib in chosen], ["candidate"])

    def test_member_addition_terminates_only_when_all_candidates_are_covered(self):
        cfg = load_case(1, quick=True)
        existing_a = Rib((0.0, 0.0), (20.0, 0.0), 2.0, "existing_a")
        existing_b = Rib((0.0, 10.0), (20.0, 10.0), 2.0, "existing_b")
        covered_a = Rib((5.0, 0.0), (15.0, 0.0), 2.0, "covered_a")
        covered_b = Rib((5.0, 10.0), (15.0, 10.0), 2.0, "covered_b")
        scores = {"covered_a": 3.0, "covered_b": 2.0}
        optimizer = RibLayoutOptimizer(
            SimpleNamespace(candidate_efficiency=lambda rib, thickness, result: scores[rib.name]),
            cfg,
        )
        optimizer.filter_until_stable = lambda ribs, values, result: (
            ribs, values, result, 0
        )
        optimizer.size = lambda *args, **kwargs: self.fail(
            "sizing must not run when every candidate is covered"
        )
        initial_t = np.array([0.2, 0.2])
        initial_result = SimpleNamespace(compliance=100.0)
        final_ribs, final_t, final_result = optimizer.adapt(
            [existing_a, existing_b], initial_t, initial_result,
            [covered_b, covered_a],
        )
        self.assertEqual(final_ribs, [existing_a, existing_b])
        self.assertTrue(np.array_equal(final_t, initial_t))
        self.assertIs(final_result, initial_result)
        self.assertTrue(any(
            "no valid uncovered candidate remains" in line for line in optimizer.log
        ))

    def test_candidate_at_seventy_percent_of_list_maximum_is_accepted(self):
        cfg = load_case(1, quick=True)
        first = Rib((0.0, 0.0), (10.0, 0.0), 2.0, "first")
        second = Rib((0.0, 5.0), (10.0, 5.0), 2.0, "second")
        third = Rib((0.0, 10.0), (10.0, 10.0), 2.0, "third")
        scores = {"first": 10.0, "second": 7.0, "third": 6.0}
        optimizer = RibLayoutOptimizer(
            SimpleNamespace(candidate_efficiency=lambda rib, thickness, result: scores[rib.name]),
            cfg,
        )
        inspected, chosen = optimizer._select_addition_candidates(
            [third, second, first], [], SimpleNamespace(), limit=2
        )
        self.assertEqual([rib.name for rib in inspected], ["first", "second"])
        self.assertEqual([rib.name for rib in chosen], ["first", "second"])

    def test_second_member_above_seventy_percent_is_accepted(self):
        cfg = load_case(1, quick=True)
        first = Rib((0.0, 0.0), (10.0, 0.0), 2.0, "first")
        second = Rib((0.0, 5.0), (10.0, 5.0), 2.0, "second")
        scores = {"first": 10.0, "second": 7.01}
        optimizer = RibLayoutOptimizer(
            SimpleNamespace(candidate_efficiency=lambda rib, thickness, result: scores[rib.name]),
            cfg,
        )
        _, chosen = optimizer._select_addition_candidates(
            [second, first], [], SimpleNamespace(), limit=2
        )
        self.assertEqual([rib.name for rib in chosen], ["first", "second"])

    def test_collinear_coverage_is_directional(self):
        existing = Rib((0, 0), (10, 0), 1)
        self.assertTrue(collinear_covered(Rib((0, 0), (10, 0), 1), existing))
        self.assertTrue(collinear_covered(Rib((2, 0), (8, 0), 1), existing))
        self.assertFalse(collinear_covered(Rib((5, 0), (12, 0), 1), existing))
        self.assertFalse(collinear_covered(Rib((5, -1), (5, 1), 1), existing))

    def test_example_one_rib_is_half_ground_diagonal(self):
        cfg = load_case(1, quick=True)
        rib = initial_ribs(cfg)[0]
        ground_diagonal = float(np.linalg.norm(cfg["domain"]))
        self.assertAlmostEqual(rib.length, ground_diagonal / 2.0, places=10)

    def test_collinear_overlap(self):
        a = Rib((0, 0), (10, 0), 1)
        self.assertTrue(collinear_overlap(a, Rib((5, 0), (12, 0), 1)))
        self.assertFalse(collinear_overlap(a, Rib((5, -1), (5, 1), 1)))
        self.assertFalse(collinear_overlap(a, Rib((5, 0), (5, 0), 1)))

    def test_invalid_geometry_freezes_only_causative_rib_positions(self):
        current = [
            Rib((0.0, 0.0), (10.0, 0.0), 2.0, "A"),
            Rib((0.0, 2.0), (10.0, 2.0), 2.0, "B"),
            Rib((0.0, 4.0), (10.0, 4.0), 2.0, "C"),
        ]
        # Only A moves onto stationary B; C becomes newly too short.
        candidate = [
            Rib((0.0, 2.0), (10.0, 2.0), 2.0, "A"),
            current[1],
            Rib((0.0, 4.0), (1.0, 4.0), 2.0, "C"),
        ]
        reasons = geometry_move_freeze_reasons(current, candidate, 2.0)
        self.assertEqual(set(reasons), {0, 2})
        self.assertIn("overlap:B", reasons[0])
        self.assertEqual(reasons[2], {"short"})

        # If both ribs move into a new overlap and neither is uniquely
        # causative, both positions are frozen.
        jointly_moved = [
            Rib((0.0, 1.0), (10.0, 1.0), 2.0, "A"),
            Rib((0.0, 1.0), (10.0, 1.0), 2.0, "B"),
            current[2],
        ]
        joint_reasons = geometry_move_freeze_reasons(
            current, jointly_moved, 2.0
        )
        self.assertEqual(set(joint_reasons), {0, 1})

    def test_sizing_respects_volume_and_improves(self):
        cfg = load_case(1, quick=True)
        cfg["mesh"] = [6, 6]
        cfg["algorithm"]["geometry_sweeps"] = 0
        model = build_model(cfg)
        ribs = initial_ribs(cfg)
        opt = RibLayoutOptimizer(model, cfg)
        start = opt._feasible_start(ribs, None)
        before = opt.analyze(ribs, start).compliance
        x, result = opt.size(ribs, maxiter=10)
        self.assertLessEqual(opt.volume(ribs, x), cfg["volume_bound"] * (1 + 1e-7))
        self.assertLessEqual(result.compliance, before * (1 + 1e-5))

    def test_sizing_uses_shared_eq7_dual_solver_without_geometry_variables(self):
        cfg = load_case(1, quick=True)

        class SizingModel:
            @staticmethod
            def analyze(ribs, thicknesses):
                return SimpleNamespace(compliance=1.0)

            @staticmethod
            def compliance_gradient(ribs, thicknesses, result):
                return -np.ones(len(ribs))

        optimizer = RibLayoutOptimizer(SizingModel(), cfg)
        ribs = initial_ribs(cfg)[:3]
        with patch(
            "rib_layout_algorithms.optimization.solve_geometry_convex_subproblem",
            wraps=solve_geometry_convex_subproblem,
        ) as shared_solver:
            optimizer.size(ribs, maxiter=1)
        self.assertEqual(shared_solver.call_count, 1)
        arguments = shared_solver.call_args.args
        self.assertEqual(len(arguments[2]), 0)  # geometry gradient
        self.assertEqual(len(arguments[8]), 0)  # coordinate scale

    def test_rationalization_inner_optimizer_has_no_fea_callbacks(self):
        cfg = load_case(1, quick=True)
        cfg["algorithm"]["rationalization_beta"] = 10.0
        cfg["algorithm"]["rationalization_max_iterations"] = 20
        cfg["algorithm"]["rationalization_geometry_iterations"] = 0

        class SensitivityModel:
            width, height = cfg["domain"]
            dx = width/cfg["mesh"][0]
            dy = height/cfg["mesh"][1]

            def analyze(self, ribs, thicknesses):
                return SimpleNamespace(compliance=1.0)

            @staticmethod
            def compliance_gradient(ribs, thicknesses, result):
                return -np.ones(len(ribs))

            @staticmethod
            def geometry_gradient(ribs, thicknesses, result, step):
                return np.zeros((len(ribs), 4))

        optimizer = RibLayoutOptimizer(SensitivityModel(), cfg)
        ribs = initial_ribs(cfg)[:4]
        thicknesses = optimizer._feasible_start(ribs, None)
        result = SimpleNamespace(compliance=1.0)
        seen_bounds = []

        def retain_current_dual_design(**kwargs):
            # The inner solve is FEA-free. Returning its current design still
            # makes each SCA outer iteration perform exactly one new analysis.
            seen_bounds.append(list(zip(kwargs["lower"], kwargs["upper"])))
            return SimpleNamespace(
                x=np.asarray(kwargs["current"], float).copy(),
                success=True,
                status=0,
                iterations=1,
                message="test dual solution",
            )

        with patch(
            "rib_layout_algorithms.optimization.solve_rationalization_convex_subproblem",
            side_effect=retain_current_dual_design,
        ):
            optimizer.rationalize(ribs, thicknesses, result, relaxation=0.05)
        # beta=1,...,10 needs ten steps; two consecutive convergence checks at
        # beta=10 make outer iteration 11 the earliest possible termination.
        self.assertEqual(optimizer.analysis_count, 11)
        self.assertEqual(len(seen_bounds), 11)
        beta_values = [
            float(line.split("beta=", 1)[1].split(",", 1)[0])
            for line in optimizer.log
            if line.startswith("rationalization SCA outer completed")
        ]
        self.assertEqual(beta_values, [*map(float, range(1, 11)), 10.0])
        iteration_events = [
            event for event in optimizer.rationalization_history
            if event.get("event") == "eq18_outer_iteration"
        ]
        self.assertEqual(len(iteration_events), 11)
        self.assertEqual(
            [event["beta"] for event in iteration_events],
            [*map(float, range(1, 11)), 10.0],
        )
        self.assertTrue(all(
            len(event["thicknesses"]) == len(ribs)
            and event["objective"] >= 0.0
            and event["compliance"] > 0.0
            for event in iteration_events
        ))
        self.assertTrue(any(
            "produced no thin or short/light rib" in line
            for line in optimizer.log
        ))
        # First endpoint is at x=0. With dx=1, the requested coordinate move
        # half-width is 0.5*1.0*(2*dx)=1.0 and is clipped at the boundary.
        self.assertTrue(np.allclose(seen_bounds[0][len(ribs)], (0.0, 1.0)))

    def test_eq18_accepts_point_one_percent_compliance_redundancy(self):
        cfg = load_case(1, quick=True)
        cfg["algorithm"]["rationalization_max_iterations"] = 1

        class RedundancyModel:
            width, height = cfg["domain"]
            dx = width/cfg["mesh"][0]
            dy = height/cfg["mesh"][1]

            @staticmethod
            def analyze(ribs, thicknesses):
                return SimpleNamespace(compliance=1.0005)

            @staticmethod
            def compliance_gradient(ribs, thicknesses, result):
                return -np.ones(len(ribs))

            @staticmethod
            def geometry_gradient(ribs, thicknesses, result, step):
                return np.zeros((len(ribs), 4))

        optimizer = RibLayoutOptimizer(RedundancyModel(), cfg)
        ribs = initial_ribs(cfg)[:4]
        thicknesses = np.full(4, 0.2)

        def small_count_descent(**kwargs):
            candidate = np.asarray(kwargs["current"]).copy()
            candidate[:len(ribs)] *= 0.999
            return SimpleNamespace(
                x=candidate,
                success=True,
                status=0,
                iterations=1,
                message="test dual solution",
            )

        with patch(
            "rib_layout_algorithms.optimization.solve_rationalization_convex_subproblem",
            side_effect=small_count_descent,
        ):
            _, _, result, _ = optimizer._solve_rationalization_eq18(
                ribs, thicknesses, SimpleNamespace(compliance=0.9), 1.0, 0.2
            )
        self.assertEqual(optimizer.analysis_count, 1)
        self.assertEqual(result.compliance, 1.0005)
        self.assertFalse(any(
            "verification rejected" in line for line in optimizer.log
        ))

    def test_rationalization_fixed_tref_batch_deletion_is_accepted(self):
        cfg = load_case(2, quick=True)
        cfg["algorithm"]["rationalization_geometry_iterations"] = 0

        class ThresholdModel:
            width, height = cfg["domain"]
            dx = width/cfg["mesh"][0]
            dy = height/cfg["mesh"][1]

            @staticmethod
            def analyze(ribs, thicknesses):
                return SimpleNamespace(compliance=1.0)

        optimizer = RibLayoutOptimizer(ThresholdModel(), cfg)
        ribs = [
            Rib((0.0, float(i)), (10.0, float(i)), 2.0, f"R{i}")
            for i in range(4)
        ]
        thicknesses = np.array([0.2, 0.2, 0.2, 0.2])
        calls = []

        def eq18(active_ribs, active_t, result, cref, tref, coordinate_bounds=None):
            calls.append((len(active_ribs), tref))
            if coordinate_bounds is None:
                coordinate_bounds = np.array([
                    [[0.0, cfg["domain"][0]], [0.0, cfg["domain"][1]],
                     [0.0, cfg["domain"][0]], [0.0, cfg["domain"][1]]]
                    for _ in active_ribs
                ])
            projected_t = np.array([0.001, 0.3, 0.3, 0.3])
            return (
                list(active_ribs), projected_t,
                SimpleNamespace(compliance=1.0), coordinate_bounds,
            )

        optimizer._solve_rationalization_eq18 = eq18
        geometry_counts = []
        optimizer.optimize_geometry = lambda active_ribs, active_t, result, bounds, max_iterations, **kwargs: (
            geometry_counts.append(len(active_ribs)) or list(active_ribs),
            np.asarray(active_t).copy(),
            SimpleNamespace(compliance=1.05),
        )
        final_ribs, _, final_result = optimizer.rationalize(
            ribs, thicknesses, SimpleNamespace(compliance=1.0), relaxation=0.05
        )
        self.assertEqual([call[0] for call in calls], [4])
        self.assertTrue(np.allclose([call[1] for call in calls], [0.2]))
        self.assertEqual(geometry_counts, [3])
        self.assertEqual(len(final_ribs), 3)
        self.assertEqual(final_result.compliance, 1.05)
        self.assertTrue(any("rationalization accepted" in line for line in optimizer.log))

    def test_rationalization_deletes_a_complete_mirror_group(self):
        cfg = load_case(2, quick=True)
        cfg["algorithm"]["rationalization_geometry_iterations"] = 0

        class SymmetricModel:
            width, height = cfg["domain"]
            dx = width/cfg["mesh"][0]
            dy = height/cfg["mesh"][1]

        optimizer = RibLayoutOptimizer(SymmetricModel(), cfg)
        ribs = [
            Rib((0.0, 0.0), (10.0, 10.0), 2.0, "left"),
            Rib((40.0, 0.0), (30.0, 10.0), 2.0, "right"),
            Rib((20.0, 0.0), (20.0, 10.0), 2.0, "center_a"),
            Rib((20.0, 10.0), (20.0, 20.0), 2.0, "center_b"),
        ]
        thicknesses = np.full(4, 0.2)

        def eq18(active_ribs, active_t, result, cref, tref, coordinate_bounds=None):
            bounds = np.array([
                [[0.0, 40.0], [0.0, 20.0], [0.0, 40.0], [0.0, 20.0]]
                for _ in active_ribs
            ])
            return (
                list(active_ribs), np.array([0.001, 0.0015, 0.3, 0.3]),
                SimpleNamespace(compliance=1.0), bounds,
            )

        optimizer._solve_rationalization_eq18 = eq18
        optimizer.optimize_geometry = (
            lambda active_ribs, active_t, result, bounds, max_iterations, **kwargs: (
                list(active_ribs), np.asarray(active_t).copy(),
                SimpleNamespace(compliance=1.04),
            )
        )
        final_ribs, _, _ = optimizer.rationalize(
            ribs, thicknesses, SimpleNamespace(compliance=1.0), relaxation=0.05
        )
        self.assertEqual(
            [rib.name for rib in final_ribs], ["center_a", "center_b"]
        )
        attempt = next(
            event for event in optimizer.rationalization_history
            if event.get("event") == "filtering_attempt"
        )
        self.assertEqual(set(attempt["removed_names"]), {"left", "right"})

    def test_rationalization_uses_fixed_thickest_seed_batches_without_symmetry(self):
        cfg = load_case(2, quick=True)
        cfg["mirror_symmetry"] = []
        cfg["algorithm"]["rationalization_geometry_iterations"] = 0

        class ThresholdModel:
            width, height = cfg["domain"]
            dx = width/cfg["mesh"][0]
            dy = height/cfg["mesh"][1]

            @staticmethod
            def analyze(ribs, thicknesses):
                return SimpleNamespace(compliance=1.0)

        optimizer = RibLayoutOptimizer(ThresholdModel(), cfg)
        ribs = [
            Rib((0.0, float(i)), (10.0, float(i)), 2.0, f"R{i}")
            for i in range(9)
        ]
        thicknesses = np.full(9, 0.2)
        eq18_calls = []
        saved_t = np.array([
            0.10, 0.12, 0.12, 0.08, 0.06, 0.04, 0.02, 0.3, 0.4,
        ])

        def eq18(active_ribs, active_t, result, cref, tref, coordinate_bounds=None):
            eq18_calls.append(tref)
            bounds = np.array([
                [[0.0, cfg["domain"][0]], [0.0, cfg["domain"][1]],
                 [0.0, cfg["domain"][0]], [0.0, cfg["domain"][1]]]
                for _ in active_ribs
            ])
            return list(active_ribs), saved_t.copy(), result, bounds

        optimizer._solve_rationalization_eq18 = eq18
        geometry_inputs = []
        geometry_current_inputs = []
        geometry_move_steps = []
        compliances = iter([1.10, 1.10, 1.04])

        def geometry(active_ribs, active_t, result, bounds, max_iterations, **kwargs):
            geometry_inputs.append(np.asarray(active_t).copy())
            geometry_current_inputs.append(result)
            geometry_move_steps.append(kwargs["initial_move_step"])
            return (
                list(active_ribs), np.asarray(active_t).copy(),
                SimpleNamespace(compliance=next(compliances)),
            )

        optimizer.optimize_geometry = geometry
        final_ribs, _, final_result = optimizer.rationalize(
            ribs, thicknesses, SimpleNamespace(compliance=1.0), relaxation=0.05
        )
        self.assertTrue(np.allclose(eq18_calls, [0.2]))
        self.assertEqual([len(values) for values in geometry_inputs], [2, 5, 8])
        self.assertTrue(np.allclose(geometry_inputs[0], [0.3, 0.4]))
        self.assertTrue(np.allclose(
            geometry_inputs[1], [0.10, 0.12, 0.12, 0.3, 0.4]
        ))
        self.assertTrue(np.allclose(
            geometry_inputs[2], [
                0.10, 0.12, 0.12, 0.08, 0.06, 0.04, 0.3, 0.4,
            ]
        ))
        self.assertEqual(geometry_current_inputs, [None, None, None])
        self.assertTrue(np.allclose(geometry_move_steps, [0.05, 0.05, 0.05]))
        self.assertEqual(
            [rib.name for rib in final_ribs],
            ["R0", "R1", "R2", "R3", "R4", "R5", "R7", "R8"],
        )
        self.assertEqual(final_result.compliance, 1.04)
        self.assertTrue(any(
            "seed_names=['R1', 'R2', 'R0']" in line for line in optimizer.log
        ))
        restorations = [
            event for event in optimizer.rationalization_history
            if event.get("event") == "restoration_after_failed_validation"
        ]
        self.assertEqual(len(restorations), 2)
        first, second = restorations
        self.assertEqual(
            first["strategy"],
            "fixed_initial_deleted_seed_count_thickest_first",
        )
        self.assertEqual([first["round"], second["round"]], [1, 2])
        self.assertEqual([first["n_rrib"], second["n_rrib"]], [7, 7])
        self.assertEqual([first["seed_target"], second["seed_target"]], [3, 3])
        self.assertFalse(first["mirror_completion"])
        self.assertEqual(
            first["ranked_currently_deleted_names"],
            ["R1", "R2", "R0", "R3", "R4", "R5", "R6"],
        )
        self.assertEqual(first["seed_names"], ["R1", "R2", "R0"])
        self.assertEqual(first["seed_thicknesses"], [0.12, 0.12, 0.10])
        self.assertEqual(first["restored_names"], ["R0", "R1", "R2"])
        self.assertEqual(
            second["ranked_currently_deleted_names"],
            ["R3", "R4", "R5", "R6"],
        )
        self.assertEqual(second["seed_names"], ["R3", "R4", "R5"])
        self.assertEqual(second["restored_names"], ["R3", "R4", "R5"])
        validation_events = [
            event for event in optimizer.rationalization_history
            if event.get("event") == "post_filter_geometry"
        ]
        self.assertTrue(all(
            event["acceptance_limit"] == 1.05 for event in validation_events
        ))
        filtering_events = [
            event for event in optimizer.rationalization_history
            if event.get("event") == "filtering_attempt"
        ]
        self.assertTrue(all(
            event["threshold"] == 0.2 for event in filtering_events
        ))

    def test_rationalization_fixed_seeds_complete_configured_mirror_groups(self):
        cfg = load_case(2, quick=True)
        cfg["algorithm"]["rationalization_geometry_iterations"] = 0

        class ThresholdModel:
            width, height = cfg["domain"]
            dx = width/cfg["mesh"][0]
            dy = height/cfg["mesh"][1]

            @staticmethod
            def analyze(ribs, thicknesses):
                return SimpleNamespace(compliance=1.0)

        optimizer = RibLayoutOptimizer(ThresholdModel(), cfg)
        ribs = [
            Rib((0.0, 0.0), (10.0, 10.0), 2.0, "A_left"),
            Rib((40.0, 0.0), (30.0, 10.0), 2.0, "A_right"),
            Rib((0.0, 20.0), (10.0, 10.0), 2.0, "B_left"),
            Rib((40.0, 20.0), (30.0, 10.0), 2.0, "B_right"),
            Rib((0.0, 5.0), (10.0, 15.0), 2.0, "C_left"),
            Rib((40.0, 5.0), (30.0, 15.0), 2.0, "C_right"),
            Rib((20.0, 0.0), (20.0, 10.0), 2.0, "center_a"),
            Rib((20.0, 10.0), (20.0, 20.0), 2.0, "center_b"),
        ]
        thicknesses = np.full(8, 0.2)
        saved_t = np.array([
            0.01, 0.009, 0.19, 0.011, 0.18, 0.012, 0.3, 0.4,
        ])

        def eq18(active_ribs, active_t, result, cref, tref, coordinate_bounds=None):
            bounds = np.array([
                [[0.0, 40.0], [0.0, 20.0], [0.0, 40.0], [0.0, 20.0]]
                for _ in active_ribs
            ])
            return list(active_ribs), saved_t.copy(), result, bounds

        optimizer._solve_rationalization_eq18 = eq18
        geometry_names = []
        compliances = iter([1.10, 1.04])

        def geometry(active_ribs, active_t, result, bounds, max_iterations, **kwargs):
            geometry_names.append([rib.name for rib in active_ribs])
            return (
                list(active_ribs), np.asarray(active_t).copy(),
                SimpleNamespace(compliance=next(compliances)),
            )

        optimizer.optimize_geometry = geometry
        final_ribs, _, final_result = optimizer.rationalize(
            ribs, thicknesses, SimpleNamespace(compliance=1.0), relaxation=0.05
        )

        self.assertEqual(geometry_names[0], ["center_a", "center_b"])
        self.assertEqual(
            geometry_names[1],
            [
                "B_left", "B_right", "C_left", "C_right",
                "center_a", "center_b",
            ],
        )
        self.assertEqual(
            [rib.name for rib in final_ribs], geometry_names[1]
        )
        self.assertEqual(final_result.compliance, 1.04)
        restoration = next(
            event for event in optimizer.rationalization_history
            if event.get("event") == "restoration_after_failed_validation"
        )
        self.assertTrue(restoration["mirror_completion"])
        self.assertEqual(restoration["mirror_axes"], ["x"])
        self.assertEqual(restoration["n_rrib"], 6)
        self.assertEqual(restoration["seed_target"], 2)
        self.assertEqual(restoration["seed_names"], ["B_left", "C_left"])
        self.assertEqual(
            restoration["restored_names"],
            ["B_left", "B_right", "C_left", "C_right"],
        )
        self.assertEqual(restoration["actual_restored_count"], 4)

    def test_rationalization_recovery_seed_target_has_minimum_one(self):
        cfg = load_case(2, quick=True)
        cfg["mirror_symmetry"] = []
        cfg["algorithm"]["rationalization_geometry_iterations"] = 0

        class ThresholdModel:
            width, height = cfg["domain"]
            dx = width/cfg["mesh"][0]
            dy = height/cfg["mesh"][1]

        optimizer = RibLayoutOptimizer(ThresholdModel(), cfg)
        ribs = [
            Rib((0.0, float(i)), (10.0, float(i)), 2.0, f"R{i}")
            for i in range(4)
        ]
        thicknesses = np.full(4, 0.2)
        original_result = SimpleNamespace(compliance=1.0)

        def eq18(active_ribs, active_t, result, cref, tref, coordinate_bounds=None):
            bounds = np.array([
                [[0.0, cfg["domain"][0]], [0.0, cfg["domain"][1]],
                 [0.0, cfg["domain"][0]], [0.0, cfg["domain"][1]]]
                for _ in active_ribs
            ])
            return (
                list(active_ribs), np.array([0.01, 0.3, 0.3, 0.3]),
                result, bounds,
            )

        optimizer._solve_rationalization_eq18 = eq18
        optimizer.optimize_geometry = (
            lambda active_ribs, active_t, result, bounds, max_iterations, **kwargs: (
                list(active_ribs), np.asarray(active_t).copy(),
                SimpleNamespace(compliance=1.10),
            )
        )
        final_ribs, final_t, final_result = optimizer.rationalize(
            ribs, thicknesses, original_result, relaxation=0.05
        )

        self.assertEqual(final_ribs, ribs)
        self.assertTrue(np.array_equal(final_t, thicknesses))
        self.assertIs(final_result, original_result)
        restoration = next(
            event for event in optimizer.rationalization_history
            if event.get("event") == "restoration_after_failed_validation"
        )
        self.assertEqual(restoration["n_rrib"], 1)
        self.assertEqual(restoration["seed_target"], 1)
        self.assertEqual(restoration["seed_names"], ["R0"])
        self.assertEqual(restoration["actual_restored_count"], 1)
        self.assertFalse(restoration["mirror_completion"])

    def test_rationalization_restores_geometry_design_if_all_deletions_fail(self):
        cfg = load_case(2, quick=True)
        cfg["algorithm"]["rationalization_geometry_iterations"] = 0

        class ThresholdModel:
            width, height = cfg["domain"]
            dx = width/cfg["mesh"][0]
            dy = height/cfg["mesh"][1]

            @staticmethod
            def analyze(ribs, thicknesses):
                return SimpleNamespace(compliance=1.0)

        optimizer = RibLayoutOptimizer(ThresholdModel(), cfg)
        ribs = [
            Rib((0.0, float(i)), (10.0, float(i)), 2.0, f"R{i}")
            for i in range(4)
        ]
        thicknesses = np.full(4, 0.2)
        original_result = SimpleNamespace(compliance=1.0)

        def eq18(active_ribs, active_t, result, cref, tref, coordinate_bounds=None):
            bounds = np.array([
                [[0.0, cfg["domain"][0]], [0.0, cfg["domain"][1]],
                 [0.0, cfg["domain"][0]], [0.0, cfg["domain"][1]]]
                for _ in active_ribs
            ])
            return (
                list(active_ribs), np.array([0.001, 0.1, 0.1, 0.1]),
                SimpleNamespace(compliance=1.0), bounds,
            )

        optimizer._solve_rationalization_eq18 = eq18
        optimizer.optimize_geometry = lambda active_ribs, active_t, result, bounds, max_iterations, **kwargs: (
            list(active_ribs), np.asarray(active_t).copy(),
            SimpleNamespace(compliance=1.10),
        )
        final_ribs, final_t, final_result = optimizer.rationalize(
            ribs, thicknesses, original_result, relaxation=0.05
        )
        self.assertEqual(final_ribs, ribs)
        self.assertTrue(np.array_equal(final_t, thicknesses))
        self.assertIs(final_result, original_result)
        self.assertTrue(any(
            "restored pre-rationalization geometry result" in line
            for line in optimizer.log
        ))
        restorations = [
            event for event in optimizer.rationalization_history
            if event.get("event") == "restoration_after_failed_validation"
        ]
        self.assertEqual(len(restorations), 2)
        self.assertEqual(
            restorations[-1]["remaining_removed_indices"], []
        )

    def test_smooth_member_count_matches_paper_projection_and_gradient(self):
        threshold, beta = 0.002, 10.0
        thicknesses = np.array([threshold, 0.5 * threshold, 1.5 * threshold])
        count = smooth_member_count(thicknesses, threshold, beta)
        expected = np.sum(
            0.5 * (1.0 + np.tanh(beta * (thicknesses / threshold - 1.0)))
        )
        self.assertAlmostEqual(count, expected, places=12)
        gradient = smooth_member_count_gradient(thicknesses, threshold, beta)
        step = 1.0e-9
        finite_difference = np.empty_like(thicknesses)
        for index in range(len(thicknesses)):
            plus, minus = thicknesses.copy(), thicknesses.copy()
            plus[index] += step
            minus[index] -= step
            finite_difference[index] = (
                smooth_member_count(plus, threshold, beta)
                - smooth_member_count(minus, threshold, beta)
            ) / (2.0 * step)
        self.assertTrue(np.allclose(gradient, finite_difference, rtol=1.0e-5, atol=1.0e-6))


if __name__ == "__main__":
    unittest.main()
