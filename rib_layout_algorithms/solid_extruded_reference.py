"""Response-only conforming triangular-prism extrusion above the base shell.

The caller supplies a nonoverlapping conforming CCW triangular mesh of the
actual material footprint. No density mixture, SIMP floor or mesher is used.
Bottom solid translations are tied nodewise to the Q4 shell top surface;
the resulting nonmatching trace approximation must be refined separately.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np
from scipy.sparse import block_diag, coo_matrix, csc_matrix

from .density_solid import (equilibrium_with_solver_threads,mkl_local_thread_setter,
                            validate_solver_threads)
from .model_shell import ShellStiffenedPlateModel
from .solid_reference import _elasticity_matrix


def prism6_stiffness(nodes_xy: np.ndarray, depth: float, E: float, nu: float) -> np.ndarray:
    """Full3-triangle-point x2-line-point stiffness of a straight6-node wedge.

    Node order is CCW bottom0,1,2 then corresponding top0,1,2. Positive height
    and positive signed triangle area are required. Weights integrate the
    unit natural triangle (area1/2) and line[-1,1] exactly for B.T D B.
    """
    xy = np.asarray(nodes_xy, float)
    if xy.shape != (3,2) or not np.isfinite(xy).all() or not np.isfinite(depth) or depth <= 0:
        raise ValueError("finite3x2 triangle and positive prism depth required")
    if not np.isfinite(E) or E <= 0 or not np.isfinite(nu) or not -1 < nu < .5:
        raise ValueError("positive E and -1<nu<0.5 required")
    area_twice = np.linalg.det(np.stack((xy[1]-xy[0],xy[2]-xy[0])))
    if area_twice <= 0:
        raise ValueError("prism triangle must be nondegenerate and counterclockwise")
    local_xy = xy-xy[0]
    xyz = np.vstack((np.c_[local_xy,np.zeros(3)],np.c_[local_xy,np.full(3,depth)]))
    constitutive = _elasticity_matrix(E,nu)
    triangle_points = ((1/6,1/6),(2/3,1/6),(1/6,2/3))
    dL = np.array([[-1.,-1.],[1.,0.],[0.,1.]])
    stiffness = np.zeros((18,18))
    for r,s in triangle_points:
        L = np.array([1-r-s,r,s])
        for zeta in (-1/np.sqrt(3),1/np.sqrt(3)):
            natural = np.empty((6,3))
            natural[:3,:2],natural[3:,:2] = dL*(1-zeta)/2,dL*(1+zeta)/2
            natural[:3,2],natural[3:,2] = -L/2,L/2
            jacobian = natural.T@xyz
            determinant = float(np.linalg.det(jacobian))
            if determinant <= 0:
                raise ValueError("nonpositive prism Jacobian")
            gradient = natural@np.linalg.inv(jacobian).T
            B = np.zeros((6,18))
            for node,(dx,dy,dz) in enumerate(gradient):
                col = 3*node
                B[0,col],B[1,col+1],B[2,col+2] = dx,dy,dz
                B[3,col],B[3,col+1] = dy,dx
                B[4,col+1],B[4,col+2] = dz,dy
                B[5,col],B[5,col+2] = dz,dx
            stiffness += B.T@constitutive@B*(determinant/6)
    return stiffness


class ConformingExtrudedSolidModel:
    """Fixed physical footprint extrusion; response-only, no design variables."""

    _equilibrium_solve = equilibrium_with_solver_threads

    def __init__(self, config: dict, footprint_nodes_xy: np.ndarray,
                 footprint_triangles: np.ndarray, *, vertical_elements: int = 8,
                 linear_solver: str = "auto",solver_threads: int = 1):
        self.solver_threads = validate_solver_threads(solver_threads)
        self.config = deepcopy(config)
        if (isinstance(vertical_elements,bool) or not np.isfinite(vertical_elements)
                or int(vertical_elements) != vertical_elements or vertical_elements < 1):
            raise ValueError("vertical_elements must be a positive integer")
        self.nz = int(vertical_elements)
        self.nodes_xy = np.asarray(footprint_nodes_xy,float).copy()
        raw_triangles = np.asarray(footprint_triangles)
        if (self.nodes_xy.ndim != 2 or self.nodes_xy.shape[1] != 2 or len(self.nodes_xy) < 3
                or not np.isfinite(self.nodes_xy).all()):
            raise ValueError("finite Nx2 footprint coordinates with N>=3 required")
        if (raw_triangles.ndim != 2 or raw_triangles.shape[1] != 3 or not len(raw_triangles)
                or not np.issubdtype(raw_triangles.dtype,np.integer)):
            raise ValueError("integer Mx3 footprint connectivity with M>=1 required")
        self.triangles = raw_triangles.astype(np.int64,copy=True)
        if self.triangles.min() < 0 or self.triangles.max() >= len(self.nodes_xy):
            raise ValueError("footprint connectivity outside node range")
        if len(np.unique(self.triangles)) != len(self.nodes_xy):
            raise ValueError("every footprint node must belong to a material triangle")
        if len(np.unique(self.nodes_xy,axis=0)) != len(self.nodes_xy):
            raise ValueError("coincident footprint nodes must be welded by the caller")
        if len(np.unique(np.sort(self.triangles,axis=1),axis=0)) != len(self.triangles):
            raise ValueError("duplicate material triangles are not allowed")
        width,height = map(float,config["domain"])
        if (np.any(self.nodes_xy < -1e-12) or np.any(self.nodes_xy[:,0] > width+1e-12)
                or np.any(self.nodes_xy[:,1] > height+1e-12)):
            raise ValueError("footprint must lie inside the base planform")
        # Repair only caller geometric clipping roundoff at an exact boundary.
        self.nodes_xy = np.clip(self.nodes_xy,[0.,0.],[width,height])
        triangle_xy = self.nodes_xy[self.triangles]
        edges1,edges2 = triangle_xy[:,1]-triangle_xy[:,0],triangle_xy[:,2]-triangle_xy[:,0]
        twice_area = edges1[:,0]*edges2[:,1]-edges1[:,1]*edges2[:,0]
        if not np.isfinite(twice_area).all() or np.any(twice_area <= 0):
            raise ValueError("all footprint triangles must be nondegenerate and counterclockwise")
        self.triangle_areas = twice_area/2
        self.footprint_area = float(self.triangle_areas.sum())
        self.z_bottom = float(config["wall_thickness"])/2
        self.z_top = float(config["rib"]["height"])
        self.extrusion_depth = self.z_top-self.z_bottom
        if not np.isfinite(self.extrusion_depth) or self.z_bottom <= 0 or self.extrusion_depth <= 0:
            raise ValueError("positive base thickness and extrusion depth required")
        self.volume = self.footprint_area*self.extrusion_depth
        self.dz = self.extrusion_depth/self.nz
        self.footprint_node_count = len(self.nodes_xy)
        material = config["material"]
        self.base = ShellStiffenedPlateModel(width=width,height=height,nx=int(config["mesh"][0]),
            ny=int(config["mesh"][1]),wall_thickness=float(config["wall_thickness"]),
            E=float(material["E"]),nu=float(material["nu"]),loads=config["load_cases"],
            supports=config["supports"],linear_solver=linear_solver)
        self._mkl_thread_setter = mkl_local_thread_setter(self.base,self.solver_threads)
        self.base_ndof = self.base.ndof
        n = self.footprint_node_count
        self.ndof = self.base_ndof+3*n*self.nz
        self.fixed_dofs = self.base.fixed_dofs.copy()
        self.free_dofs = np.setdiff1d(np.arange(self.ndof),self.fixed_dofs)
        self.load_vectors = [np.pad(f,(0,self.ndof-self.base_ndof)) for f in self.base.load_vectors]
        self.load_weights = self.base.load_weights.copy()
        self.constraint = self._constraint_matrix()
        self.element_kernels = np.array([prism6_stiffness(xy,self.dz,self.base.E,self.base.nu)
                                         for xy in triangle_xy])
        physical_dofs = 3*n*(self.nz+1)
        solid_stiffness = csc_matrix((physical_dofs,physical_dofs))
        for iz in range(self.nz):
            nodes = np.concatenate((self.triangles+iz*n,self.triangles+(iz+1)*n),axis=1)
            edofs = (3*nodes[:,:,None]+np.arange(3)).reshape(len(nodes),18)
            rows = np.repeat(edofs,18,axis=1).ravel()
            cols = np.tile(edofs,(1,18)).ravel()
            solid_stiffness += coo_matrix((self.element_kernels.ravel(),(rows,cols)),
                                          shape=(physical_dofs,physical_dofs)).tocsc()
        self.added_stiffness = (self.constraint.T@solid_stiffness@self.constraint).tocsc()
        self.base_stiffness = block_diag((self.base.base_stiffness,
                                         csc_matrix((self.ndof-self.base_ndof,)*2)),format="csc")
        self.stiffness = self.base_stiffness+self.added_stiffness

    def _constraint_matrix(self):
        n = self.footprint_node_count
        rows,cols,data = [],[],[]
        for index,point in enumerate(self.nodes_xy):
            nodes,weights = self.base.interpolation(point)
            for node,weight in zip(nodes,weights):
                if weight == 0:
                    continue
                for solid_component,base_component,factor in (
                        (0,0,1.),(0,4,self.z_bottom),(1,1,1.),(1,3,-self.z_bottom),(2,2,1.)):
                    rows.append(3*index+solid_component)
                    cols.append(6*int(node)+base_component)
                    data.append(weight*factor)
        upper_count = 3*n*self.nz
        rows.extend((3*n+np.arange(upper_count)).tolist())
        cols.extend((self.base_ndof+np.arange(upper_count)).tolist())
        data.extend(np.ones(upper_count).tolist())
        return coo_matrix((data,(rows,cols)),shape=(3*n*(self.nz+1),self.ndof)).tocsc()

    def analyze(self) -> dict:
        forces = np.column_stack(self.load_vectors)
        free = self.free_dofs
        solved,details = self._equilibrium_solve(self.stiffness[free][:,free].tocsc(),forces[free])
        displacement = np.zeros_like(forces); displacement[free] = solved
        compliances = np.sum(forces*displacement,axis=0)
        objective = float(self.load_weights@compliances)
        if not np.isfinite(objective) or objective < 0:
            raise RuntimeError("invalid conforming prism compliance")
        base_displacement = displacement[:self.base_ndof].copy()
        return dict(compliance=objective,load_compliances=compliances,load_weights=self.load_weights.copy(),
                    displacement=displacement,base_displacement=base_displacement,
                    volume=self.volume,footprint_area=self.footprint_area,
                    relative_residual=float(details["relative_residual"]),linear_solution=details,
                    max_downward_displacement=np.maximum(0.,-base_displacement[2::6].min(axis=0)),
                    material_representation="conforming full-material triangular-prism extrusion; no SIMP or void floor")

    def metadata(self) -> dict:
        return dict(model="fixed base shell plus conforming equal-height prism solid footprint",
                    solid_element="linear6-node wedge, full3-point triangle x2-point line integration",
                    base_mesh=list(self.config["mesh"]),footprint_nodes=self.footprint_node_count,
                    footprint_triangles=len(self.triangles),vertical_elements=self.nz,
                    solid_elements=len(self.triangles)*self.nz,hybrid_dofs=self.ndof,
                    base_shell_dofs=self.base_ndof,footprint_area=self.footprint_area,
                    solver_threads=self.solver_threads,
                    solver_thread_control="MKL/Pardiso calling-thread local only; original automatic SuperLU fallback unchanged",
                    layer_bottom_z=self.z_bottom,layer_top_z=self.z_top,
                    extrusion_depth=self.extrusion_depth,actual_added_volume=self.volume,
                    interface="nodewise Q4 interpolation of shell top displacement; upper3-D nodes independent",
                    support_semantics="original base shell constraints only; solid side/top surfaces free",
                    material="full solid footprint only; no void floor, SIMP or density filter",
                    caller_mesh_contract="conforming nonoverlapping CCW triangles exactly covering intended material union",
                    accuracy_limit="resolve wedge bending and nonmatching Q4/triangle interface trace by mesh refinement")
