"""Column-constrained 3-D SIMP above a fixed base shell.

Only the bottom solid face follows the shell's top-surface displacement.
Every higher solid plane has independent Cartesian displacement unknowns.
The design constraint shares density, never displacement, through a column.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
from numbers import Integral
import numpy as np
from scipy.sparse import block_diag, coo_matrix, csc_matrix

from .density_shell import ParallelLayerModel
from .solid_reference import _hex8_stiffness


def validate_solver_threads(value: int) -> int:
    if isinstance(value,bool) or not isinstance(value,Integral) or value < 1:
        raise ValueError("solver_threads must be a positive integer")
    return int(value)


def mkl_local_thread_setter(base, threads: int):
    """Resolve thread-local control from the already loaded Pardiso MKL only."""
    if base._pardiso_spsolve is None or base._pardiso_solver is None:
        if threads > 1:
            raise ValueError("solver_threads>1 requires MKL Pardiso; SuperLU is not controlled by this option")
        return None
    library = getattr(base._pardiso_solver,"libmkl",None)
    setter = None
    if library is not None:
        for name in ("MKL_Set_Num_Threads_Local","mkl_set_num_threads_local"):
            setter = getattr(library,name,None)
            if setter is not None:
                break
    if setter is None:
        if threads > 1:
            raise RuntimeError("loaded Pardiso MKL does not expose a thread-local thread setter")
        return None
    setter.argtypes = [ctypes.c_int]
    setter.restype = ctypes.c_int
    return setter


@contextmanager
def scoped_mkl_solver_threads(setter, threads: int):
    """Restore the previous calling-thread MKL setting, including on errors."""
    if setter is None:
        yield
        return
    previous = int(setter(threads))
    try:
        yield
    finally:
        setter(previous)


def equilibrium_with_solver_threads(model, matrix, rhs):
    # The audited solve and every equilibration/refinement call remain inside
    # the same scope. Matrices, residual threshold and solver fallback remain
    # unchanged. This knob applies only to MKL/Pardiso, not SuperLU/OpenBLAS.
    with scoped_mkl_solver_threads(model._mkl_thread_setter,model.solver_threads):
        solution,details = ParallelLayerModel._equilibrium_solve(model,matrix,rhs)
    details = dict(details,requested_solver_threads=model.solver_threads,
                   solver_thread_control=("MKL calling-thread local; original SuperLU fallback remains"
                                          if model._mkl_thread_setter is not None else "none; default single-thread policy"))
    return solution,details


def interface_transform(offset: float) -> np.ndarray:
    """Map four base6 plus four upper3 DOFs to a bottom brick's24 DOFs."""
    if not np.isfinite(offset):
        raise ValueError("interface offset must be finite")
    transform = np.zeros((24, 36))
    for node in range(4):
        transform[3*node:3*node+3, 6*node:6*node+3] = np.eye(3)
        transform[3*node, 6*node+4] = offset
        transform[3*node+1, 6*node+3] = -offset
    transform[12:, 24:] = np.eye(12)
    return transform


def rib_footprint_occupancy(config: dict, ribs, *, samples: int = 12) -> np.ndarray:
    """Sample union of flat-ended rectangular ribs, with widths in actual mm.

    This is an explicit geometric quadrature approximation, not a SIMP density
    or a thresholded optimized layout. Crossings are counted once. Increasing
    samples and the planform grid is necessary to establish geometry accuracy.
    All ribs must reach the configured common top height.
    """
    if isinstance(samples, bool) or int(samples) != samples or samples < 2:
        raise ValueError("samples must be an integer >=2")
    nx, ny = map(int, config["mesh"])
    width, height = map(float, config["domain"])
    expected_height = float(config["rib"]["height"])
    rectangles = []
    for rib in ribs:
        get = rib.get if isinstance(rib, dict) else lambda name: getattr(rib, name)
        a, b = np.asarray(get("p0"), float), np.asarray(get("p1"), float)
        thickness, top = float(get("thickness")), float(get("height"))
        vector = b-a; length = float(np.linalg.norm(vector))
        if (a.shape != (2,) or b.shape != (2,) or not np.all(np.isfinite(np.r_[a,b,thickness,top]))
                or length <= 0 or thickness < 0 or not np.isclose(top, expected_height, rtol=0, atol=1e-12)):
            raise ValueError("finite positive-length, nonnegative-width, common-height ribs required")
        if thickness:
            rectangles.append((a, vector/length, length, thickness))
    offsets = (np.arange(int(samples))+.5)/samples
    occupied = np.zeros((ny, nx))
    # Vectorize all planform cells; bound temporary memory by one quadrature row.
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
    for sy in offsets:
        for sx in offsets:
            x, y = (xx+sx)*width/nx, (yy+sy)*height/ny
            hit = np.zeros((ny, nx), bool)
            for a, direction, length, thickness in rectangles:
                px, py = x-a[0], y-a[1]
                longitudinal = px*direction[0]+py*direction[1]
                transverse = -px*direction[1]+py*direction[0]
                hit |= ((longitudinal >= 0) & (longitudinal <= length)
                        & (np.abs(transverse) <= thickness/2))
            occupied += hit
    return (occupied/(samples*samples)).ravel()


class ExtrudedSolidModel(ParallelLayerModel):
    """Fixed shell + bonded constant-height 3-D solid column design domain."""

    def __init__(self, config: dict, *, vertical_elements: int = 8,
                 penal: float = 3., emin_ratio: float = 1e-9,
                 filter_radius: float | None = None, linear_solver: str = "auto",
                 solver_threads: int = 1):
        self.solver_threads = validate_solver_threads(solver_threads)
        if (isinstance(vertical_elements, bool) or not np.isfinite(vertical_elements)
                or int(vertical_elements) != vertical_elements
                or vertical_elements < 1):
            raise ValueError("vertical_elements must be a positive integer")
        if not np.isfinite(emin_ratio) or not 0 < emin_ratio < 1:
            raise ValueError("solid void DOFs require an explicit positive Emin/E0 in (0,1)")
        # Reuse exactly the audited controls, column filter, symmetry and base.
        super().__init__(config, penal=penal, emin_ratio=emin_ratio,
                         filter_radius=filter_radius, linear_solver=linear_solver)
        self._mkl_thread_setter = mkl_local_thread_setter(self.base,self.solver_threads)
        self.nz = int(vertical_elements)
        self.z_bottom = self.base_thickness/2
        self.z_top = float(config["rib"]["height"])
        self.dz = self.layer_thickness/self.nz
        self.base_ndof = self.base.ndof
        self.ndof = self.base_ndof+3*self.base.nnode*self.nz
        self.fixed_dofs = self.base.fixed_dofs.copy()
        self.free_dofs = np.setdiff1d(np.arange(self.ndof), self.fixed_dofs)
        self.load_vectors = [np.pad(f, (0, self.ndof-self.base_ndof)) for f in self.base.load_vectors]
        self.load_weights = self.base.load_weights.copy()
        self.base_stiffness = block_diag((self.base.base_stiffness,
                                         csc_matrix((self.ndof-self.base_ndof,)*2)), format="csc")
        # Release obsolete parallel-shell assembly arrays before constructing3-D.
        del self.rows, self.cols, self.edofs, self.layer_element, self.transform
        self.solid_element = _hex8_stiffness(self.dx, self.dy, self.dz, self.base.E, self.base.nu)
        self.interface_transform = interface_transform(self.z_bottom)
        self.interface_element = self.interface_transform.T@self.solid_element@self.interface_transform
        xy_nodes = np.array([[self.base.node(ix, iy), self.base.node(ix+1, iy),
                              self.base.node(ix+1, iy+1), self.base.node(ix, iy+1)]
                             for iy in range(self.ny) for ix in range(self.nx)], dtype=np.int64)
        base_dofs = (6*xy_nodes[:,:,None]+np.arange(6)).reshape(self.nelem,24)
        groups = []
        for iz in range(self.nz):
            upper = (self.base_ndof+3*(iz*self.base.nnode+xy_nodes[:,:,None])
                     +np.arange(3)).reshape(self.nelem,12)
            if iz == 0:
                edofs = np.concatenate((base_dofs, upper), axis=1)
                kernel = self.interface_element
            else:
                lower = (self.base_ndof+3*((iz-1)*self.base.nnode+xy_nodes[:,:,None])
                         +np.arange(3)).reshape(self.nelem,12)
                edofs = np.concatenate((lower, upper), axis=1)
                kernel = self.solid_element
            nonzero = np.flatnonzero(kernel.ravel() != 0)
            size = kernel.shape[0]
            rows = edofs[:,nonzero//size].ravel()
            cols = edofs[:,nonzero%size].ravel()
            groups.append(dict(edofs=edofs, kernel=kernel, rows=rows, cols=cols,
                               values=kernel.ravel()[nonzero]))
        self.element_groups = tuple(groups)

    _equilibrium_solve = equilibrium_with_solver_threads

    def _stiffness_scale(self, scale: np.ndarray):
        stiffness = self.base_stiffness.copy()
        for group in self.element_groups:
            data = (scale[:,None]*group["values"][None,:]).ravel()
            stiffness += coo_matrix((data, (group["rows"],group["cols"])),
                                    shape=(self.ndof,self.ndof)).tocsc()
        return stiffness

    def stiffness(self, physical: np.ndarray):
        rho = self._density(physical)
        return self._stiffness_scale(self.emin_ratio+(1-self.emin_ratio)*rho**self.penal)

    def occupancy_stiffness(self, occupancy: np.ndarray):
        """Linear occupied-volume mixture for geometric footprint quadrature."""
        area = self._density(occupancy)
        return self._stiffness_scale(self.emin_ratio+(1-self.emin_ratio)*area)

    def _response(self, values: np.ndarray, *, occupancy: bool) -> dict:
        rho = self._density(values)
        stiffness = self.occupancy_stiffness(rho) if occupancy else self.stiffness(rho)
        forces = np.column_stack(self.load_vectors)
        free = self.free_dofs
        solved, details = self._equilibrium_solve(stiffness[free][:,free].tocsc(), forces[free])
        displacement = np.zeros_like(forces); displacement[free] = solved
        compliances = np.sum(forces*displacement, axis=0)
        objective = float(self.load_weights@compliances)
        if not np.isfinite(objective) or objective < 0:
            raise RuntimeError("invalid hybrid compliance")
        column_energy = np.zeros(self.nelem)
        for group in self.element_groups:
            local_u = displacement[group["edofs"]]
            column_energy += np.einsum("eim,ij,ejm,m->e", local_u, group["kernel"],
                                       local_u, self.load_weights, optimize=True)
        derivative = np.full(self.nelem, 1-self.emin_ratio) if occupancy else (
            self.penal*(1-self.emin_ratio)*rho**(self.penal-1))
        base_displacement = displacement[:self.base_ndof].copy()
        self.analysis_count += 1
        return dict(compliance=objective, load_compliances=compliances,
                    load_weights=self.load_weights.copy(), displacement=displacement,
                    base_displacement=base_displacement, physical_density=rho.copy(),
                    physical_gradient=-derivative*column_energy,
                    volume=float(self.cell_volume*rho.sum()),
                    max_downward_displacement=np.maximum(0., -base_displacement[2::6].min(axis=0)),
                    relative_residual=float(details["relative_residual"]), linear_solution=details,
                    material_representation=("linear geometric area occupancy; no SIMP or filter"
                                             if occupancy else "column-shared SIMP physical density"))

    def evaluate_physical(self, physical: np.ndarray) -> dict:
        return self._response(physical, occupancy=False)

    def evaluate_occupancy(self, occupancy: np.ndarray) -> dict:
        """Analyze fixed physical rib footprint without filtering or penalizing.

        Fractional cells are a declared homogenized geometric quadrature, not
        exact cut-cell integration. Their physical volume is reported rather
        than silently adjusted to the nominal rib-volume budget.
        """
        return self._response(occupancy, occupancy=True)

    def metadata(self) -> dict:
        return dict(model="fixed base shell plus column-constrained extruded 3-D solid",
                    solid_element="trilinear Hex8, full2x2x2 integration",
                    layer_thickness=self.layer_thickness, layer_bottom_z=self.z_bottom,
                    layer_top_z=self.z_top, vertical_elements=self.nz,
                    mesh=[self.nx,self.ny,self.nz], design_variables=self.nelem,
                    solid_elements=self.nelem*self.nz, hybrid_dofs=self.ndof,
                    base_shell_dofs=self.base_ndof,
                    fixed_base_thickness=self.base_thickness,
                    fixed_base_volume=self.width*self.height*self.base_thickness,
                    design_volume_budget=self.budget, solid_design_capacity=self.capacity,
                    target_volume_fraction=self.volume_fraction, penal=self.penal,
                    emin_ratio=self.emin_ratio, filter_radius=self.filter_radius,
                    solver_threads=self.solver_threads,
                    solver_thread_control="MKL/Pardiso calling-thread local only; original automatic SuperLU fallback unchanged",
                    filter="row-normalized2-D physical-density filter with exact adjoint",
                    mirror_symmetry=list(self.symmetry),
                    layer_kinematics="bottom solid face tied to shell top displacement; upper3-D nodes independent",
                    manufacturing_constraints="constant density along z; equal-height straight extrusion; no draft angle",
                    support_semantics="original base shell constraints only; solid side/top surfaces free",
                    fixed_footprint_model="linear geometric area occupancy with union volume; no density filter or SIMP",
                    comparison_limit="same hybrid model needed for response comparison; resolve mesh, footprint and Emin effects")
