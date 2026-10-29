"""Analytic skew-element patches, independent of the assembly implementation."""
import unittest

import numpy as np

from rib_layout_algorithms.shell import shell_q4_stiffness


class AffineShellPatchTests(unittest.TestCase):
    xyz = np.array([[0., 0., 0.], [2., 0., 0.],
                    [3., 1., 0.], [1., 1., 0.]])
    E, nu, thickness, area = 70000., .3, .1, 2.

    def membrane_displacement(self, strain, spin=0.):
        exx, eyy, gamma = strain
        gradient = np.array([[exx, gamma/2-spin], [gamma/2+spin, eyy]])
        return np.concatenate([np.r_[gradient@p[:2], 0., 0., 0., spin] for p in self.xyz])

    def membrane_energy(self, strain):
        plane = self.E/(1-self.nu**2)*np.array([
            [1., self.nu, 0.], [self.nu, 1., 0.], [0., 0., (1-self.nu)/2]])
        return self.area*self.thickness*(strain@plane@strain)

    def test_skew_uniaxial_extension_matches_analytic_energy(self):
        strain = np.array([.002, 0., 0.])
        q = self.membrane_displacement(strain)
        k = shell_q4_stiffness(self.xyz, self.thickness, self.E, self.nu)
        self.assertAlmostEqual(float(q@k@q), self.membrane_energy(strain), places=12)

    def test_skew_constant_strain_components_and_in_plane_spin(self):
        k = shell_q4_stiffness(self.xyz, self.thickness, self.E, self.nu)
        for strain in (np.array([0., -.001, 0.]), np.array([0., 0., .003]),
                       np.array([.002, -.001, .003])):
            for spin in (0., .004):
                with self.subTest(strain=strain.tolist(), spin=spin):
                    q = self.membrane_displacement(strain, spin)
                    self.assertAlmostEqual(float(q@k@q), self.membrane_energy(strain), places=12)

    def test_skew_membrane_patch_is_objective_under_spatial_rotation(self):
        axis = np.array([1., 2., 3.]); axis /= np.linalg.norm(axis)
        cross = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
        angle = .83
        rotation = np.eye(3)+np.sin(angle)*cross+(1-np.cos(angle))*(cross@cross)
        xyz = self.xyz@rotation.T+np.array([3., -2., 5.])
        strain = np.array([.002, -.001, .003])
        q_local = self.membrane_displacement(strain, .004).reshape(4, 6)
        q = np.concatenate([np.r_[rotation@row[:3], rotation@row[3:]] for row in q_local])
        k = shell_q4_stiffness(xyz, self.thickness, self.E, self.nu)
        self.assertAlmostEqual(float(q@k@q), self.membrane_energy(strain), places=12)

    def test_skew_shell_preserves_all_rigid_body_modes(self):
        k = shell_q4_stiffness(self.xyz, self.thickness, self.E, self.nu)
        for index, direction in enumerate(np.eye(3)):
            translation = np.concatenate([np.r_[direction, np.zeros(3)] for p in self.xyz])
            rotation = np.concatenate([np.r_[np.cross(direction, p), direction] for p in self.xyz])
            for name, q in (("translation", translation), ("rotation", rotation)):
                with self.subTest(mode=name, direction=index):
                    scale = max(float(np.linalg.norm(k)*np.linalg.norm(q)), 1.)
                    self.assertLess(float(np.linalg.norm(k@q))/scale, 1e-12)


if __name__ == "__main__":
    unittest.main()
