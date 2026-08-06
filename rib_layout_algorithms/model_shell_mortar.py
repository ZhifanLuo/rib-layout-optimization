"""Q4 shell model with mortar-style rib-root coupling."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from scipy.sparse import csc_matrix, coo_matrix

from .model import Rib
from .model_shell import ShellStiffenedPlateModel


class MortarQ4ShellStiffenedPlateModel(ShellStiffenedPlateModel):
    """Q4 shell/rib model using an L2 projection at every rib root.

    The vertical Q4 rib strip is retained as an independent component.  Its
    bottom-edge degrees of freedom are coupled to the ground-shell trace by
    ``M_root T = M_cross`` rather than by pointwise interpolation.  The root
    mesh is fixed in physical coordinates and is therefore independent of the
    ground-shell grid.  The coupling is compatible with the production
    thickness and endpoint sensitivities because the root mesh is fixed in
    physical coordinates.  It remains an optional model selected by
    configuration; the default shell model is unchanged unless
    ``analysis_model`` is set to ``mortar_shell``.
    """

    def __init__(
        self, *args, root_segments: int = 40,
        mortar_quadrature_order: int = 4, **kwargs,
    ) -> None:
        if int(root_segments) < 1:
            raise ValueError("root_segments must be positive")
        if int(mortar_quadrature_order) < 2:
            raise ValueError("mortar_quadrature_order must be at least two")
        self.root_segments = int(root_segments)
        self.mortar_quadrature_order = int(mortar_quadrature_order)
        self._mortar_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        super().__init__(*args, **kwargs)

    def rib_bottom_points(self, rib: Rib) -> np.ndarray:
        p0 = np.asarray(rib.p0, float)
        p1 = np.asarray(rib.p1, float)
        parameters = np.linspace(0.0, 1.0, self.root_segments+1)
        return (1.0-parameters[:, None])*p0+parameters[:, None]*p1

    def _rib_bottom_points_and_shape_derivatives(
        self, rib: Rib, side: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return smooth endpoint derivatives for the fixed physical root mesh."""
        del side
        p0 = np.asarray(rib.p0, float)
        p1 = np.asarray(rib.p1, float)
        parameters = np.linspace(0.0, 1.0, self.root_segments + 1)
        bottom = (1.0-parameters[:, None])*p0+parameters[:, None]*p1
        derivatives = np.zeros((len(parameters), 2, 4), float)
        derivatives[:, 0, 0] = 1.0-parameters
        derivatives[:, 1, 1] = 1.0-parameters
        derivatives[:, 0, 2] = parameters
        derivatives[:, 1, 3] = parameters
        return bottom, derivatives

    def _mortar_breaks(self, rib: Rib) -> list[float]:
        """Split quadrature intervals at ground-shell grid crossings."""
        p0 = np.asarray(rib.p0, float)
        p1 = np.asarray(rib.p1, float)
        delta = p1-p0
        values = [0.0, 1.0]
        if abs(delta[0]) > 1.0e-14:
            values.extend(
                (ix*self.dx-p0[0])/delta[0]
                for ix in range(1, self.nx)
                if 1.0e-12 < (ix*self.dx-p0[0])/delta[0] < 1.0-1.0e-12
            )
        if abs(delta[1]) > 1.0e-14:
            values.extend(
                (iy*self.dy-p0[1])/delta[1]
                for iy in range(1, self.ny)
                if 1.0e-12 < (iy*self.dy-p0[1])/delta[1] < 1.0-1.0e-12
            )
        return sorted(set(float(value) for value in values))

    def _mortar_projection(
        self, rib: Rib,
    ) -> tuple[np.ndarray, np.ndarray]:
        key = (self._rib_numeric_geometry_key(rib), self.root_segments)
        cached = self._mortar_cache.get(key)
        if cached is not None:
            self._mortar_cache.move_to_end(key)
            return cached
        root_count = self.root_segments+1
        root_mass = np.zeros((root_count, root_count))
        cross_terms: dict[tuple[int, int], float] = {}
        ground_nodes: set[int] = set()
        p0 = np.asarray(rib.p0, float)
        p1 = np.asarray(rib.p1, float)
        delta = p1-p0
        length = float(np.linalg.norm(delta))
        gauss, weights = np.polynomial.legendre.leggauss(
            self.mortar_quadrature_order
        )
        grid_breaks = self._mortar_breaks(rib)
        for root_index in range(self.root_segments):
            root_left = root_index/self.root_segments
            root_right = (root_index+1)/self.root_segments
            sub_breaks = [root_left, root_right]
            sub_breaks.extend(
                value for value in grid_breaks
                if root_left < value < root_right
            )
            sub_breaks = sorted(set(sub_breaks))
            for left, right in zip(sub_breaks[:-1], sub_breaks[1:]):
                midpoint = 0.5*(left+right)
                half = 0.5*(right-left)
                for point, weight in zip(gauss, weights):
                    parameter = midpoint+half*point
                    physical = p0+parameter*delta
                    nodes, shape = self.interpolation(physical)
                    root_fraction = (
                        parameter-root_left)/(root_right-root_left)
                    root_shape = np.array(
                        [1.0-root_fraction, root_fraction], float,
                    )
                    jacobian = length*half
                    integration_weight = float(weight*jacobian)
                    root_nodes = (root_index, root_index+1)
                    root_mass[np.ix_(root_nodes, root_nodes)] += (
                        integration_weight*np.outer(root_shape, root_shape)
                    )
                    for node, value in zip(nodes, shape):
                        ground_nodes.add(int(node))
                        for local_root, root_value in zip(root_nodes, root_shape):
                            cross_terms[(local_root, int(node))] = (
                                cross_terms.get((local_root, int(node)), 0.0)
                                + integration_weight*root_value*float(value)
                            )
        ordered_ground = np.array(sorted(ground_nodes), int)
        lookup = {int(node): index for index, node in enumerate(ordered_ground)}
        cross = np.zeros((root_count, len(ordered_ground)))
        for (root_node, ground_node), value in cross_terms.items():
            cross[root_node, lookup[ground_node]] = value
        projection = np.linalg.solve(root_mass, cross)
        result = (ordered_ground, projection)
        self._mortar_cache[key] = result
        while len(self._mortar_cache) > self.rib_cache_max_entries:
            self._mortar_cache.popitem(last=False)
        return result

    def rib_stiffness(self, rib: Rib, thickness: float) -> csc_matrix:
        key = (
            "mortar-l2", self._rib_numeric_state_key(rib, thickness),
            self.root_segments, self.mortar_quadrature_order,
        )
        cached = self._rib_cache.get(key)
        if cached is not None:
            self._rib_cache.move_to_end(key)
            return cached
        bottom, condensed = self._rib_condensed(rib, float(thickness))
        ground_nodes, projection = self._mortar_projection(rib)
        root_count = len(bottom)
        ground_count = len(ground_nodes)
        mapping = np.zeros((6*root_count, 6*ground_count))
        identity = np.eye(6)
        for root_node in range(root_count):
            for ground_node in range(ground_count):
                value = projection[root_node, ground_node]
                if abs(value) > 1.0e-15:
                    mapping[
                        6*root_node:6*root_node+6,
                        6*ground_node:6*ground_node+6,
                    ] += value*identity
        ground_stiffness = mapping.T@condensed@mapping
        dofs = np.concatenate([
            np.arange(6*int(node), 6*int(node)+6)
            for node in ground_nodes
        ])
        rows, cols = np.meshgrid(dofs, dofs, indexing="ij")
        mask = np.abs(ground_stiffness) > 1.0e-18
        matrix = coo_matrix(
            (ground_stiffness[mask], (rows[mask], cols[mask])),
            shape=(self.ndof, self.ndof),
        ).tocsc()
        self._rib_cache[key] = matrix
        while len(self._rib_cache) > self.rib_cache_max_entries:
            self._rib_cache.popitem(last=False)
        return matrix
