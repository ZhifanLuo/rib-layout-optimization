"""Reference shell model with consistent embedded line-beam ribs."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from scipy.sparse import csc_matrix, coo_matrix

from .frame import local_axes, rigid_offset_transform
from .model import AnalysisResult, Rib
from .model_shell import ShellStiffenedPlateModel
from .model_shell_local import LocallyRefinedShellStiffenedPlateModel
from .shell import shape_q4


class EmbeddedLineBeamShellReferenceModel(ShellStiffenedPlateModel):
    """Fixed-layout shell reference with direct line-stiffener coupling.

    A rib is represented by an eccentric rectangular Euler--Bernoulli line
    beam. Its axial, bending, and torsional strains are evaluated from the
    existing shell displacement/rotation field and integrated along the
    physical rib line. The rib contributes stiffness only on existing shell
    DOFs; no mesh-dependent root constraint nodes are introduced.

    This model is response-only. It is intended to verify frozen layouts and
    is not enabled in the production optimizer without a separate geometry
    sensitivity derivation.
    """

    @staticmethod
    def _beam_section(rib: Rib, thickness: float) -> tuple[float, float, float, float]:
        t = float(thickness)
        h = float(rib.height)
        if t <= 0.0 or h <= 0.0:
            raise ValueError("rib thickness and height must be positive")
        return t*h, t*h**3/12.0, h*t**3/12.0, h*t**3/3.0

    def _interpolation_with_line_derivative(
        self, point: np.ndarray, tangent: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = float(np.clip(point[0], 0.0, self.width))
        y = float(np.clip(point[1], 0.0, self.height))
        if hasattr(self, "x_coords") and hasattr(self, "y_coords"):
            ix = min(max(int(np.searchsorted(self.x_coords, x, side="right")-1), 0), self.nx-1)
            iy = min(max(int(np.searchsorted(self.y_coords, y, side="right")-1), 0), self.ny-1)
            left, right = self.x_coords[ix], self.x_coords[ix+1]
            bottom, top = self.y_coords[iy], self.y_coords[iy+1]
            cell_dx, cell_dy = right-left, top-bottom
        else:
            ix = min(int(x/self.dx), self.nx-1)
            iy = min(int(y/self.dy), self.ny-1)
            left, bottom = ix*self.dx, iy*self.dy
            cell_dx, cell_dy = self.dx, self.dy
        # shell.shape_q4 uses the natural coordinate range [-1, 1], whereas
        # the physical-cell fractions below are in [0, 1].
        xi = 2.0*(x-left)/cell_dx-1.0
        eta = 2.0*(y-bottom)/cell_dy-1.0
        nodes = np.array([
            self.node(ix, iy), self.node(ix+1, iy),
            self.node(ix+1, iy+1), self.node(ix, iy+1),
        ], int)
        shape, derivatives = shape_q4(xi, eta)
        dshape_xy = np.column_stack((
            2.0*derivatives[:, 0]/cell_dx,
            2.0*derivatives[:, 1]/cell_dy,
        ))
        return nodes, shape, dshape_xy@np.asarray(tangent, float)

    def _line_strain_rows(
        self, rib: Rib, point: np.ndarray, tangent: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        nodes, shape, dshape_line = self._interpolation_with_line_derivative(
            point, tangent,
        )
        q0 = np.r_[np.asarray(rib.p0, float), float(rib.height)/2.0]
        q1 = np.r_[np.asarray(rib.p1, float), float(rib.height)/2.0]
        rotation, _ = local_axes(q0, q1)
        rows = [np.zeros(6*len(nodes)) for _ in range(4)]
        offset = rigid_offset_transform(
            np.array([0.0, 0.0, float(rib.height)/2.0])
        )
        for index, derivative in enumerate(dshape_line):
            start = 6*index
            translation_derivative = rotation@(derivative*offset[:3, :])
            rotation_derivative = rotation@(derivative*offset[3:, :])
            rows[0][start:start+6] = translation_derivative[0]
            rows[1][start:start+6] = rotation_derivative[1]
            rows[2][start:start+6] = rotation_derivative[2]
            rows[3][start:start+6] = rotation_derivative[0]
        return nodes, shape, rows[0], rows[1], rows[2], rows[3]

    def rib_stiffness(self, rib: Rib, thickness: float) -> csc_matrix:
        key = (
            "embedded-line", self._rib_numeric_state_key(rib, thickness),
            int(self.interface_subdivisions_per_cell),
        )
        if not hasattr(self, "_embedded_line_cache"):
            self._embedded_line_cache = OrderedDict()
        cached = self._embedded_line_cache.get(key)
        if cached is not None:
            self._embedded_line_cache.move_to_end(key)
            return cached
        area, iy, iz, torsion = self._beam_section(rib, thickness)
        shear_modulus = self.E/(2.0*(1.0+self.nu))
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        points = self.rib_bottom_points(rib)
        for p0, p1 in zip(points[:-1], points[1:]):
            delta = np.asarray(p1)-np.asarray(p0)
            length = float(np.linalg.norm(delta))
            if length <= 1.0e-14:
                continue
            tangent = delta/length
            midpoint = 0.5*(np.asarray(p0)+np.asarray(p1))
            half_length = 0.5*length
            for gauss_point in (-1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0)):
                point = midpoint+half_length*gauss_point*tangent
                node_array, _, axial, bend_y, bend_z, twist = (
                    self._line_strain_rows(rib, point, tangent)
                )
                matrix = (
                    self.E*area*np.outer(axial, axial)
                    + self.E*iy*np.outer(bend_y, bend_y)
                    + self.E*iz*np.outer(bend_z, bend_z)
                    + shear_modulus*torsion*np.outer(twist, twist)
                )*half_length
                dofs = np.concatenate([
                    np.arange(6*int(node), 6*int(node)+6)
                    for node in node_array
                ])
                self._append_dense(rows, cols, data, dofs, matrix)
        result = coo_matrix(
            (data, (rows, cols)), shape=(self.ndof, self.ndof)
        ).tocsc()
        self._embedded_line_cache[key] = result
        while len(self._embedded_line_cache) > self.rib_cache_max_entries:
            self._embedded_line_cache.popitem(last=False)
        return result

    def stiffness(self, ribs, thicknesses) -> csc_matrix:
        matrix = self.base_stiffness.copy()
        for rib, thickness in zip(ribs, thicknesses):
            matrix = matrix+self.rib_stiffness(rib, float(thickness))
        return matrix.tocsc()

    def analyze(self, ribs, thicknesses) -> AnalysisResult:
        matrix = self.stiffness(ribs, thicknesses)
        free_matrix = matrix[self.free_dofs][:, self.free_dofs].tocsc()
        rhs = np.column_stack([
            load[self.free_dofs] for load in self.load_vectors
        ])
        free_solution = self._solve_global(free_matrix, rhs)
        if free_solution.ndim == 1:
            free_solution = free_solution[:, None]
        displacements: list[np.ndarray] = []
        compliances: list[float] = []
        for column, load in enumerate(self.load_vectors):
            displacement = np.zeros(self.ndof)
            displacement[self.free_dofs] = free_solution[:, column]
            displacements.append(displacement)
            compliances.append(float(load @ displacement))
        return AnalysisResult(
            float(self.load_weights @ compliances),
            displacements,
            compliances,
        )

    def compliance_gradient(self, *args, **kwargs):
        raise NotImplementedError("reference embedded-line model is response-only")

    def geometry_gradient(self, *args, **kwargs):
        raise NotImplementedError("reference embedded-line model is response-only")


class LocallyRefinedEmbeddedLineBeamShellReferenceModel(
    EmbeddedLineBeamShellReferenceModel
):
    """Embedded-line reference model on the local nonuniform shell grid."""

    node_xy = LocallyRefinedShellStiffenedPlateModel.node_xy
    nearest_node = LocallyRefinedShellStiffenedPlateModel.nearest_node
    interpolation = LocallyRefinedShellStiffenedPlateModel.interpolation
    _nodes_in_patch = LocallyRefinedShellStiffenedPlateModel._nodes_in_patch
    rib_bottom_points = LocallyRefinedShellStiffenedPlateModel.rib_bottom_points
    _build_axis = staticmethod(LocallyRefinedShellStiffenedPlateModel._build_axis)
    _refinement_intervals = staticmethod(
        LocallyRefinedShellStiffenedPlateModel._refinement_intervals
    )

    def __init__(
        self, width, height, nx, ny, wall_thickness, E, nu, loads, supports,
        refinement_ribs=(), refinement_factor: int = 2,
        interface_subdivisions_per_cell: int = 4, **kwargs,
    ):
        self._base_dx = float(width)/int(nx)
        self._base_dy = float(height)/int(ny)
        self.x_coords = self._build_axis(
            float(width), int(nx), self._refinement_intervals(
                loads, supports, refinement_ribs, 0,
            ), refinement_factor,
        )
        self.y_coords = self._build_axis(
            float(height), int(ny), self._refinement_intervals(
                loads, supports, refinement_ribs, 1,
            ), refinement_factor,
        )
        super().__init__(
            width, height, len(self.x_coords)-1, len(self.y_coords)-1,
            wall_thickness, E, nu, loads, supports,
            interface_subdivisions_per_cell=interface_subdivisions_per_cell,
            **kwargs,
        )
        self._base_dx = float(width)/int(nx)
        self._base_dy = float(height)/int(ny)
