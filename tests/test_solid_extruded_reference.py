import unittest
import numpy as np

from rib_layout_core import load_case
from rib_layout_algorithms.solid_extruded_reference import (
    ConformingExtrudedSolidModel,prism6_stiffness)


def config():
    cfg = load_case(5)
    cfg.update(domain=[6.,4.],mesh=[3,2],wall_thickness=.4,volume_bound=6.,
               supports={"type":"edge","edge":"right"},
               load_cases=[{"weight":1.,"forces":[{"point":[0.,4.],"value":[1.,0.,-2.]}]},
                           {"weight":3.,"forces":[{"point":[0.,0.],"value":[0.,1.,-1.]}]}])
    cfg["rib"]["height"] = 2.
    return cfg


class PrismElementTests(unittest.TestCase):
    xy = np.array([[.2,.3],[2.3,.5],[.7,1.9]])
    depth,E,nu = 1.7,70000.,.3

    def test_skew_prism_general_affine_energy_matches_3d_elasticity(self):
        k = prism6_stiffness(self.xy,self.depth,self.E,self.nu)
        gradient = np.array([[.002,.003,-.001],[.001,-.002,.002],[.003,.001,.001]])
        xyz = np.vstack((np.c_[self.xy,np.zeros(3)],np.c_[self.xy,np.full(3,self.depth)]))
        q = (xyz@gradient.T).ravel()
        strain = .5*(gradient+gradient.T)
        lam = self.E*self.nu/((1+self.nu)*(1-2*self.nu))
        mu = self.E/(2*(1+self.nu))
        area = np.linalg.det(np.stack((self.xy[1]-self.xy[0],self.xy[2]-self.xy[0])))/2
        analytic = area*self.depth*(lam*np.trace(strain)**2+2*mu*np.sum(strain**2))
        self.assertAlmostEqual(float(q@k@q),analytic,places=11)

    def test_full_prism_has_exactly_six_rigid_modes_and_no_hourglass_modes(self):
        k = prism6_stiffness(self.xy,self.depth,self.E,self.nu)
        xyz = np.vstack((np.c_[self.xy,np.zeros(3)],np.c_[self.xy,np.full(3,self.depth)]))
        scale = np.linalg.norm(k)
        for direction in np.eye(3):
            for q in (np.tile(direction,6),np.cross(direction,xyz).ravel()):
                self.assertLess(np.linalg.norm(k@q)/(scale*np.linalg.norm(q)),1e-13)
        values = np.linalg.eigvalsh(k)
        self.assertEqual(np.count_nonzero(values<values[-1]*1e-10),6)
        self.assertGreater(values[6],values[-1]*1e-5)

    def test_orientation_degeneracy_and_material_validation(self):
        for xy,depth,E,nu in ((self.xy[::-1],self.depth,self.E,self.nu),
                               (np.zeros((3,2)),self.depth,self.E,self.nu),
                               (self.xy,0.,self.E,self.nu),(self.xy,self.depth,-1.,self.nu),
                               (self.xy,self.depth,self.E,.5)):
            with self.subTest(depth=depth,E=E,nu=nu),self.assertRaises(ValueError):
                prism6_stiffness(xy,depth,E,nu)


class ConformingPrismModelTests(unittest.TestCase):
    nodes = np.array([[.2,.4],[4.9,.3],[3.2,3.7]])
    triangles = np.array([[0,1,2]])

    def make_model(self):
        return ConformingExtrudedSolidModel(config(),self.nodes,self.triangles,
                                            vertical_elements=2,linear_solver="superlu")

    def hybrid_field(self,m,gradient=None,omega=None,translation=None):
        gradient = np.zeros((3,3)) if gradient is None else gradient
        omega = np.zeros(3) if omega is None else omega
        translation = np.zeros(3) if translation is None else translation
        q = np.zeros(m.ndof)
        for node in range(m.base.nnode):
            x,y = m.base.node_xy(node)
            q[6*node:6*node+6] = np.r_[gradient@[x,y,m.z_bottom]+translation+np.cross(omega,[x,y,0.]),omega]
        for iz in range(m.nz):
            for node,(x,y) in enumerate(m.nodes_xy):
                z = m.z_bottom+(iz+1)*m.dz
                start = m.base_ndof+3*(iz*m.footprint_node_count+node)
                q[start:start+3] = gradient@[x,y,z]+translation+np.cross(omega,[x,y,z])
        return q

    def test_nonmatching_interface_preserves_all_rigid_modes(self):
        m = self.make_model(); scale = np.linalg.norm(m.added_stiffness.data)
        xyz = np.vstack([np.c_[m.nodes_xy,np.full(len(m.nodes_xy),m.z_bottom+iz*m.dz)]
                         for iz in range(m.nz+1)])
        for direction in np.eye(3):
            for q,expected in ((self.hybrid_field(m,translation=direction),np.tile(direction,len(xyz))),
                               (self.hybrid_field(m,omega=direction),np.cross(direction,xyz).ravel())):
                np.testing.assert_allclose(m.constraint@q,expected,atol=2e-15)
                self.assertLess(np.linalg.norm(m.added_stiffness@q)/(scale*np.linalg.norm(q)),1e-13)

    def test_nonmatching_constraint_and_global_affine_solid_energy(self):
        m = self.make_model()
        gradient = np.array([[.002,.003,-.001],[.001,-.002,.002],[.003,.001,.001]])
        q = self.hybrid_field(m,gradient=gradient)
        strain = (gradient+gradient.T)/2
        E,nu = m.base.E,m.base.nu
        expected = m.volume*(E*nu/((1+nu)*(1-2*nu))*np.trace(strain)**2
                             +E/(1+nu)*np.sum(strain**2))
        self.assertAlmostEqual(float(q@m.added_stiffness@q),expected,places=9)
        self.assertEqual(m.z_top,2.)
        self.assertEqual(m.z_bottom,.2)
        self.assertAlmostEqual(m.volume,m.footprint_area*1.8)

    def test_response_has_full_material_volume_and_original_base_load_support_semantics(self):
        m = self.make_model(); response = m.analyze()
        self.assertLessEqual(response["relative_residual"],1e-7)
        self.assertAlmostEqual(response["compliance"],np.array([.25,.75])@response["load_compliances"],places=14)
        self.assertEqual(response["volume"],m.volume)
        np.testing.assert_array_equal(m.fixed_dofs,m.base.fixed_dofs)
        np.testing.assert_array_equal(response["base_displacement"],response["displacement"][:m.base_ndof])
        self.assertEqual(np.count_nonzero(response["displacement"][m.fixed_dofs]),0)
        for full,base in zip(m.load_vectors,m.base.load_vectors):
            np.testing.assert_array_equal(full[:m.base_ndof],base)
            self.assertEqual(np.count_nonzero(full[m.base_ndof:]),0)
        matrix = m.stiffness[m.free_dofs][:,m.free_dofs].toarray()
        self.assertGreater(np.linalg.eigvalsh(matrix)[0],0.)

    def test_geometry_contract_rejects_singular_or_duplicated_meshes(self):
        cases = [(self.nodes,self.triangles[:,::-1]),
                 (np.vstack((self.nodes,[[1.,1.]])),self.triangles),
                 (self.nodes,np.vstack((self.triangles,self.triangles))),
                 (self.nodes,self.triangles.astype(float)),
                 (np.array([[-.1,.4],[4.9,.3],[3.2,3.7]]),self.triangles)]
        for nodes,triangles in cases:
            with self.subTest(nodes=nodes.tolist(),triangles=triangles.tolist()),self.assertRaises(ValueError):
                ConformingExtrudedSolidModel(config(),nodes,triangles,vertical_elements=2)

    def test_solver_thread_validation_and_default_metadata(self):
        m = self.make_model()
        self.assertEqual(m.metadata()["solver_threads"],1)
        with self.assertRaisesRegex(ValueError,"requires MKL Pardiso"):
            ConformingExtrudedSolidModel(config(),self.nodes,self.triangles,
                                         vertical_elements=2,linear_solver="superlu",solver_threads=2)

    def test_real_mkl_one_vs_two_threads_preserves_conforming_response(self):
        one = ConformingExtrudedSolidModel(config(),self.nodes,self.triangles,
                                           vertical_elements=2,linear_solver="auto",solver_threads=1)
        if one._mkl_thread_setter is None:
            self.skipTest("optional MKL/Pardiso local thread control unavailable")
        two = ConformingExtrudedSolidModel(config(),self.nodes,self.triangles,
                                           vertical_elements=2,linear_solver="auto",solver_threads=2)
        a,b = one.analyze(),two.analyze()
        self.assertLessEqual(a["relative_residual"],1e-7)
        self.assertLessEqual(b["relative_residual"],1e-7)
        np.testing.assert_allclose(a["load_compliances"],b["load_compliances"],rtol=1e-9)
        np.testing.assert_array_equal(one.stiffness.toarray(),two.stiffness.toarray())


if __name__ == "__main__":
    unittest.main()
