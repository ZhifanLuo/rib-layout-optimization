from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np

from rib_layout_core import load_case
from rib_layout_algorithms.density_solid import (
    ExtrudedSolidModel, interface_transform, rib_footprint_occupancy,
    mkl_local_thread_setter,scoped_mkl_solver_threads,validate_solver_threads)
from rib_layout_algorithms.density_shell import ParallelLayerModel
from tools import run_extruded_density_comparison as cli


def config(symmetry=()):
    cfg = load_case(5)
    cfg.update(domain=[6.,4.], mesh=[3,2], wall_thickness=.4,
               volume_bound=6., mirror_symmetry=list(symmetry),
               supports={"type":"edge", "edge":"right"},
               load_cases=[{"weight":1., "forces":[{"point":[0.,4.], "value":[0.,-1.,-2.]}]},
                           {"weight":3., "forces":[{"point":[0.,0.], "value":[1.,0.,-1.]}]}])
    cfg["rib"]["height"] = 2.
    return cfg


def model(**kwargs):
    return ExtrudedSolidModel(config(), vertical_elements=2, linear_solver="superlu", **kwargs)


class ExtrudedSolidTests(unittest.TestCase):
    def test_density_models_reject_nonphysical_isotropic_material_before_assembly(self):
        for constructor in (ParallelLayerModel, ExtrudedSolidModel):
            for key, value in (("E", 0.), ("E", -70000.), ("E", np.inf),
                               ("E", np.nan), ("nu", -1.), ("nu", .5),
                               ("nu", .6), ("nu", np.inf), ("nu", np.nan)):
                with self.subTest(model=constructor.__name__, key=key, value=value):
                    cfg = config()
                    cfg["material"][key] = value
                    with self.assertRaisesRegex(ValueError, "isotropic material"):
                        constructor(cfg, linear_solver="superlu")

    def test_solid_inherits_zero_start_guard_and_stable_load_weights(self):
        cfg = config()
        cfg["load_cases"][0]["weight"] = 1e308
        cfg["load_cases"][1]["weight"] = 1e308
        solid = ExtrudedSolidModel(cfg, vertical_elements=2, linear_solver="superlu")
        np.testing.assert_array_equal(solid.base.load_weights, [.5, .5])
        with self.assertRaisesRegex(ValueError, "nonzero initial density"):
            solid.optimize(initial_raw=np.zeros(solid.nelem))
        response = solid.analyze(np.full(solid.nelem, solid.volume_fraction))
        self.assertGreater(response["compliance"], 0.)
        self.assertTrue(np.all(np.isfinite(response["gradient"])))

    def test_interface_offset_maps_rigid_rotation_at_physical_surface(self):
        transform = interface_transform(.2)
        xy = np.array([[0.,0.],[2.,0.],[2.,2.],[0.,2.]])
        for omega in np.eye(3):
            base = np.concatenate([np.r_[np.cross(omega,[x,y,0.]),omega] for x,y in xy])
            upper = np.concatenate([np.cross(omega,[x,y,1.1]) for x,y in xy])
            expected = np.concatenate([np.cross(omega,[x,y,.2]) for x,y in xy]+[upper])
            np.testing.assert_allclose(transform@np.r_[base,upper], expected, atol=1e-15)

    def hybrid_field(self, m, gradient=None, omega=None, translation=None):
        field = np.zeros(m.ndof)
        gradient = np.zeros((3,3)) if gradient is None else gradient
        omega = np.zeros(3) if omega is None else omega
        translation = np.zeros(3) if translation is None else translation
        for node in range(m.base.nnode):
            x,y = m.base.node_xy(node)
            # For general affine checks use zero rotations and exact interface
            # translation, sufficient to exercise the constrained brick field.
            base_translation = (gradient@np.array([x,y,m.z_bottom]) + translation
                                + np.cross(omega,[x,y,0.]))
            field[6*node:6*node+6] = np.r_[base_translation,omega]
            for iz in range(m.nz):
                xyz = np.array([x,y,m.z_bottom+(iz+1)*m.dz])
                start = m.base_ndof+3*(iz*m.base.nnode+node)
                field[start:start+3] = gradient@xyz+translation+np.cross(omega,xyz)
        return field

    def test_assembled_solid_affine_energy_matches_three_dimensional_elasticity(self):
        m = model()
        gradient = np.array([[.002,.003,-.001],[.001,-.002,.002],[.003,.001,.001]])
        field = self.hybrid_field(m,gradient=gradient)
        added = m.stiffness(np.ones(m.nelem))-m.base_stiffness
        strain = .5*(gradient+gradient.T)
        E,nu = m.base.E,m.base.nu
        lam,mu = E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))
        analytic = m.capacity*(lam*np.trace(strain)**2+2*mu*np.sum(strain*strain))
        self.assertAlmostEqual(float(field@added@field),analytic,places=10)

    def test_added_solid_preserves_rigid_modes_and_upper_nodes_are_independent(self):
        m = model()
        added = m.stiffness(np.ones(m.nelem))-m.base_stiffness
        scale = float(np.linalg.norm(added.data))
        for direction in np.eye(3):
            for field in (self.hybrid_field(m,translation=direction),self.hybrid_field(m,omega=direction)):
                self.assertLess(np.linalg.norm(added@field)/(scale*np.linalg.norm(field)),1e-13)
        self.assertEqual(m.ndof,m.base_ndof+3*m.nz*m.base.nnode)
        first = m.element_groups[0]["edofs"]
        second = m.element_groups[1]["edofs"]
        np.testing.assert_array_equal(first[:,24:],second[:,:12])
        self.assertTrue(np.all(first[:,24:]>=m.base_ndof))
        self.assertTrue(np.all(second[:,12:]>=m.base_ndof+3*m.base.nnode))
        local = np.zeros(m.ndof); local[-3] = .001
        self.assertGreater(float(local@added@local),0.)

    def test_all_variable_multiload_symmetric_compliance_and_volume_derivatives(self):
        cfg = config(("x","y"))
        m = ExtrudedSolidModel(cfg,vertical_elements=3,filter_radius=2.8,linear_solver="superlu")
        raw = np.linspace(.2,.7,m.nelem)
        response = m.analyze(raw); step = 1e-6
        finite,volume = [],[]
        for index in range(m.nelem):
            d = np.zeros(m.nelem); d[index] = step
            plus,minus = m.analyze(raw+d),m.analyze(raw-d)
            finite.append((plus["compliance"]-minus["compliance"])/(2*step))
            volume.append((plus["volume"]-minus["volume"])/(2*step))
        np.testing.assert_allclose(response["gradient"],finite,rtol=2e-5,atol=1e-10)
        np.testing.assert_allclose(m.volume_gradient(),volume,rtol=1e-8)
        self.assertAlmostEqual(response["compliance"],np.array([.25,.75])@response["load_compliances"],places=14)

    def test_physical_geometry_volume_loads_and_displacement_semantics(self):
        m = model()
        self.assertEqual(m.z_top,m.config["rib"]["height"])
        self.assertEqual(m.z_bottom,m.base_thickness/2)
        response = m.evaluate_physical(np.full(m.nelem,.3))
        self.assertAlmostEqual(response["volume"],.3*6*4*(2-.4/2))
        self.assertEqual(response["displacement"].shape,(m.ndof,2))
        self.assertEqual(response["base_displacement"].shape,(m.base_ndof,2))
        np.testing.assert_array_equal(response["base_displacement"],response["displacement"][:m.base_ndof])
        np.testing.assert_array_equal(m.fixed_dofs,m.base.fixed_dofs)
        for force,base_force in zip(m.load_vectors,m.base.load_vectors):
            np.testing.assert_array_equal(force[:m.base_ndof],base_force)
            self.assertEqual(np.count_nonzero(force[m.base_ndof:]),0)
        self.assertLessEqual(response["relative_residual"],1e-7)

    def test_geometric_occupancy_is_unfiltered_and_not_penalized(self):
        a = model(penal=3.,filter_radius=2.8)
        b = model(penal=1.,filter_radius=.5)
        occupancy = np.array([.1,.2,.3,.4,.5,.6])
        one,two = a.evaluate_occupancy(occupancy),b.evaluate_occupancy(occupancy)
        self.assertAlmostEqual(one["compliance"],two["compliance"],places=13)
        np.testing.assert_array_equal(one["physical_density"],occupancy)
        self.assertGreater(a.evaluate_physical(occupancy)["compliance"],one["compliance"])
        self.assertAlmostEqual(one["volume"],a.cell_volume*occupancy.sum())
        self.assertIn("no SIMP or filter",one["material_representation"])

    def test_positive_void_floor_is_required_and_void_response_remains_finite(self):
        for floor in (0.,-1.,1.,np.nan):
            with self.subTest(floor=floor),self.assertRaises(ValueError):
                model(emin_ratio=floor)
        m = model(emin_ratio=1e-10)
        response = m.evaluate_physical(np.zeros(m.nelem))
        self.assertTrue(np.isfinite(response["compliance"]))
        self.assertEqual(response["volume"],0.)
        self.assertLessEqual(response["relative_residual"],1e-7)
        for nz in (0,-1,1.5,True,np.nan,np.inf):
            with self.subTest(nz=nz),self.assertRaises(ValueError):
                ExtrudedSolidModel(config(),vertical_elements=nz)

    def test_inherited_oc_and_stop_history_obey_physical_budget(self):
        m = model()
        result = m.optimize(max_iterations=4)
        self.assertEqual(result["analysis_count"],result["iterations"]+1)
        self.assertEqual(result["initialization"]["type"],"uniform_at_budget")
        self.assertAlmostEqual(result["initial"]["volume"],m.budget)
        for row in result["history"]:
            self.assertIn("raw_change",row)
            self.assertLessEqual(row["volume"],m.budget*(1+1e-10))
            self.assertLessEqual(row["relative_residual"],1e-7)

    def test_rectangle_union_has_flat_ends_and_no_double_counting(self):
        cfg = config(); cfg.update(domain=[6.,4.],mesh=[6,4])
        rib = dict(p0=[1.,2.],p1=[3.,2.],height=2.,thickness=2.)
        one = rib_footprint_occupancy(cfg,[rib],samples=4).reshape(4,6)
        two = rib_footprint_occupancy(cfg,[rib,rib],samples=4)
        self.assertEqual(one.sum(),4.)
        np.testing.assert_array_equal(one.ravel(),two)
        self.assertEqual(one[:,0].sum(),0.)
        self.assertEqual(one[:,3:].sum(),0.)


class ExtrudedSolidCliTests(unittest.TestCase):
    def test_planning_is_read_only_and_execution_saves_continuous_hybrid_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)/"new"
            with redirect_stdout(io.StringIO()) as stdout:
                cli.main(["--cases","5","--output",str(out),"--vertical-elements","2"])
            self.assertFalse(out.exists())
            plan = json.loads(stdout.getvalue())
            self.assertEqual(plan["cases"][0]["options"]["emin_ratio"],1e-9)
            self.assertEqual(plan["cases"][0]["options"]["vertical_elements"],2)
            self.assertEqual(plan["cases"][0]["options"]["solver_threads"],1)
            with mock.patch.object(cli,"load_case",return_value=config()), \
                    mock.patch.object(ExtrudedSolidModel,"binary_design",side_effect=AssertionError("binary called")), \
                    redirect_stdout(io.StringIO()):
                cli.main(["--cases","5","--output",str(out),"--vertical-elements","2",
                          "--max-iterations","3","--no-images","--execute"])
            folder = out/"example_5"
            result = json.loads((folder/"results.json").read_text())
            self.assertEqual(result["initialization"]["type"],"uniform_at_budget")
            self.assertEqual(result["total_analyses"],result["iterations"]+1)
            self.assertEqual(result["model"]["mesh"],[3,2,2])
            self.assertFalse(any("binary" in name or "threshold" in name for name in result))
            self.assertFalse(any("binary" in p.name or "threshold" in p.name for p in folder.iterdir()))
            self.assertTrue((folder/"base_displacement.npy").is_file())
            self.assertEqual(np.load(folder/"displacement.npy").shape[0],result["model"]["hybrid_dofs"])
            manifest = json.loads((out/"manifest.json").read_text())
            self.assertTrue(manifest["source_unchanged"])


class SolidSolverThreadTests(unittest.TestCase):
    def test_thread_count_requires_a_positive_integer_and_available_pardiso(self):
        for value in (0,-1,1.5,True,np.nan,np.inf):
            with self.subTest(value=value),self.assertRaises(ValueError):
                validate_solver_threads(value)
        self.assertEqual(validate_solver_threads(np.int64(2)),2)
        with self.assertRaisesRegex(ValueError,"requires MKL Pardiso"):
            model(solver_threads=2)
        base = SimpleNamespace(_pardiso_spsolve=lambda *args:None,
                               _pardiso_solver=SimpleNamespace(libmkl=SimpleNamespace()))
        with self.assertRaisesRegex(RuntimeError,"thread-local thread setter"):
            mkl_local_thread_setter(base,2)
        self.assertIsNone(mkl_local_thread_setter(base,1))

    def test_scoped_thread_setting_restores_previous_value_on_success_and_exception(self):
        state = [0]; calls = []
        def change(value):
            previous = state[0]; state[0] = value; calls.append(value); return previous
        for fail in (False,True):
            state[0] = 4
            try:
                with scoped_mkl_solver_threads(change,2):
                    self.assertEqual(state[0],2)
                    if fail:
                        raise RuntimeError("test exception")
            except RuntimeError:
                self.assertTrue(fail)
            self.assertEqual(state[0],4)
        self.assertEqual(calls,[2,4,2,4])

    def test_complete_original_equilibrium_fallback_stays_inside_thread_scope(self):
        from scipy.sparse import diags
        m = model(); state = [4]; calls = []
        def change(value):
            previous = state[0]; state[0] = value; return previous
        m._mkl_thread_setter = change; m.solver_threads = 2
        matrix = diags([1e-12,1e12]).tocsc(); rhs = np.array([[1.],[2.]])
        def limited_precision(system,force):
            self.assertEqual(state[0],2)
            calls.append(system.copy())
            result = np.linalg.solve(system.toarray(),force)
            return result*(1+2e-7) if len(calls)<=2 else result
        with mock.patch.object(m.base,"_solve_global",side_effect=limited_precision):
            solved,details = m._equilibrium_solve(matrix,rhs)
        self.assertEqual(details["method"],"equilibrated_fallback")
        self.assertLess(details["relative_residual"],1e-7)
        np.testing.assert_allclose(matrix@solved,rhs,rtol=1e-14)
        self.assertEqual(state[0],4)
        self.assertEqual(details["requested_solver_threads"],2)
        with mock.patch.object(ParallelLayerModel,"_equilibrium_solve",side_effect=RuntimeError("failed solve")), \
                self.assertRaisesRegex(RuntimeError,"failed solve"):
            m._equilibrium_solve(matrix,rhs)
        self.assertEqual(state[0],4)

    def test_real_mkl_one_vs_two_thread_response_agrees_and_restores_control(self):
        one = ExtrudedSolidModel(config(),vertical_elements=2,linear_solver="auto",solver_threads=1)
        if one._mkl_thread_setter is None:
            self.skipTest("optional MKL/Pardiso local thread control unavailable")
        two = ExtrudedSolidModel(config(),vertical_elements=2,linear_solver="auto",solver_threads=2)
        setter = one._mkl_thread_setter; previous = int(setter(0))
        try:
            rho = np.linspace(.2,.7,one.nelem)
            a,b = one.evaluate_physical(rho),two.evaluate_physical(rho)
            self.assertLessEqual(a["relative_residual"],1e-7)
            self.assertLessEqual(b["relative_residual"],1e-7)
            np.testing.assert_allclose(a["load_compliances"],b["load_compliances"],rtol=1e-9)
            np.testing.assert_array_equal(one.stiffness(rho).toarray(),two.stiffness(rho).toarray())
            self.assertEqual(int(setter(0)),0)
        finally:
            setter(previous)


if __name__ == "__main__":
    unittest.main()
