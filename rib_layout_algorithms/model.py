"""Internal finite-element surrogate for a thin wall with explicit ribs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, eye
from scipy.sparse.linalg import splu

from .frame import frame_global_stiffness, rigid_offset_transform


@dataclass(frozen=True)
class Rib:
    p0: tuple[float, float]
    p1: tuple[float, float]
    height: float
    name: str = ""
    segments: int = 10

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.subtract(self.p1, self.p0)))

    @property
    def key(self) -> tuple:
        a, b = sorted((tuple(np.round(self.p0, 10)), tuple(np.round(self.p1, 10))))
        return a, b, round(self.height, 10), self.segments


@dataclass
class AnalysisResult:
    compliance: float
    displacements: list[np.ndarray]
    load_compliances: list[float]


def equation_15_deformation_factor(
    response_at: Callable[[np.ndarray, Sequence[float]], np.ndarray],
    load_weights: Sequence[float],
    rib: Rib,
    result: AnalysisResult,
    omega_displacement: float = 1.0,
    omega_rotation: float = 1.0,
) -> float:
    """Evaluate manuscript Eq. (15) on the planar ground surface.

    The ground normal is ``+Z``. Thus, for the in-plane axial unit vector
    ``a=(ax,ay,0)``, the transverse vector ``b=n x a`` is
    ``(-ay,ax,0)``. Translational and rotational endpoint responses use the
    first and last three shell DOFs, respectively.
    """
    length = float(rib.length)
    if length <= 0.0:
        return -np.inf
    wd = float(omega_displacement)
    wt = float(omega_rotation)
    if wd < 0.0 or wt < 0.0 or wd+wt <= 0.0:
        raise ValueError(
            "Eq. (15) weights must be nonnegative and not both zero"
        )
    direction_2d = (np.asarray(rib.p1, float)-np.asarray(rib.p0, float))/length
    axial = np.array([direction_2d[0], direction_2d[1], 0.0])
    transverse = np.array([-direction_2d[1], direction_2d[0], 0.0])
    weights = np.asarray(load_weights, float)
    if len(weights) != len(result.displacements):
        raise ValueError("load weights and displacement cases must have equal length")
    factor = 0.0
    for weight, displacement in zip(weights, result.displacements):
        response_0 = np.asarray(response_at(displacement, rib.p0), float)
        response_1 = np.asarray(response_at(displacement, rib.p1), float)
        if response_0.size < 6 or response_1.size < 6:
            raise ValueError("Eq. (15) requires six shell DOFs at each endpoint")
        relative = response_1[:6]-response_0[:6]
        axial_measure = abs(float(axial@relative[:3]))/length
        rotation_measure = (
            float(rib.height)*abs(float(transverse@relative[3:6]))/length
        )
        factor += float(weight)*(wd*axial_measure+wt*rotation_measure)
    return float(factor)


def endpoint_energy_density_factor(
    response_at: Callable[[np.ndarray, Sequence[float]], np.ndarray],
    load_weights: Sequence[float],
    elastic_modulus: float,
    rib: Rib,
    result: AnalysisResult,
) -> float:
    """Return a fast endpoint strain-energy-density surrogate.

    The axial endpoint strain and transverse rotation gradient are squared and
    weighted by their membrane and bending stiffness scales.  The measure is
    used only to form a small candidate shortlist; final ranking uses the
    fixed-volume net compliance benefit.
    """
    length = float(rib.length)
    if length <= 0.0:
        return -np.inf
    direction_2d = (np.asarray(rib.p1, float)-np.asarray(rib.p0, float))/length
    axial = np.array([direction_2d[0], direction_2d[1], 0.0])
    transverse = np.array([-direction_2d[1], direction_2d[0], 0.0])
    factor = 0.0
    for weight, displacement in zip(load_weights, result.displacements):
        response_0 = np.asarray(response_at(displacement, rib.p0), float)
        response_1 = np.asarray(response_at(displacement, rib.p1), float)
        relative = response_1[:6]-response_0[:6]
        axial_strain = float(axial@relative[:3])/length
        rotation_gradient = float(transverse@relative[3:6])/length
        density = float(elastic_modulus)*(
            axial_strain**2 + float(rib.height)**2*rotation_gradient**2/12.0
        )
        factor += float(weight)*density
    return float(factor)


class StiffenedPlateModel:
    """Structured wall grid with eccentric 3-D frame rib stiffeners.

    Six degrees of freedom are retained at every wall node.  The wall is an
    open, reproducible grillage approximation to a thin shell.  A rib is a
    rectangular eccentric beam distributed through the requested number of
    segments and bilinearly tied to the wall grid.
    """

    def __init__(
        self,
        width: float,
        height: float,
        nx: int,
        ny: int,
        wall_thickness: float,
        E: float,
        nu: float,
        loads: Sequence[dict],
        supports: dict,
        diagonal_scale: float = 0.12,
        deformation_factor_method: str = "fixed_volume_net_benefit",
    ) -> None:
        self.width, self.height = float(width), float(height)
        self.nx, self.ny = int(nx), int(ny)
        self.wall_thickness = float(wall_thickness)
        self.E, self.nu = float(E), float(nu)
        self.dx, self.dy = self.width / self.nx, self.height / self.ny
        self.nnode = (self.nx + 1) * (self.ny + 1)
        self.ndof = 6 * self.nnode
        self._rib_cache: dict[tuple, tuple[csc_matrix, csc_matrix]] = {}
        self.base_stiffness = self._build_wall(diagonal_scale).tocsc()
        self.fixed_dofs = self._build_supports(supports)
        self.free_dofs = np.setdiff1d(np.arange(self.ndof), self.fixed_dofs)
        self.load_vectors, self.load_weights = self._build_loads(loads)
        if not self.load_vectors:
            raise ValueError("at least one load case is required")
        self.deformation_factor_method = str(deformation_factor_method).lower()
        if self.deformation_factor_method not in {
            "equation_15", "stiffness_per_volume", "fixed_volume_net_benefit"
        }:
            raise ValueError(
                "deformation_factor_method must be equation_15, "
                "stiffness_per_volume, or fixed_volume_net_benefit"
            )

    def node(self, ix: int, iy: int) -> int:
        return iy * (self.nx + 1) + ix

    def node_xy(self, node: int) -> np.ndarray:
        iy, ix = divmod(int(node), self.nx + 1)
        return np.array([ix * self.dx, iy * self.dy], float)

    def nearest_node(self, point: Sequence[float]) -> int:
        x, y = point
        ix = int(np.clip(round(x / self.dx), 0, self.nx))
        iy = int(np.clip(round(y / self.dy), 0, self.ny))
        return self.node(ix, iy)

    @staticmethod
    def _append_dense(rows: list[int], cols: list[int], data: list[float], dofs: np.ndarray, k: np.ndarray) -> None:
        rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
        mask = np.abs(k) > 1.0e-16
        rows.extend(rr[mask].ravel().tolist())
        cols.extend(cc[mask].ravel().tolist())
        data.extend(k[mask].ravel().tolist())

    def _wall_member(self, p0: np.ndarray, p1: np.ndarray, tributary: float, axial_scale: float = 1.0) -> np.ndarray:
        t = self.wall_thickness
        A = max(t * tributary * axial_scale, 1.0e-18)
        # Correct strip bending about an in-plane axis; suppress artificial
        # in-plane frame bending because membrane shear comes from diagonals.
        Iy = tributary * t**3 / 12.0
        Iz = t * tributary**3 * 1.0e-7 / 12.0
        J = max(Iy * 0.15, 1.0e-24)
        p03 = np.r_[p0, 0.0]
        p13 = np.r_[p1, 0.0]
        return frame_global_stiffness(p03, p13, self.E, self.nu, A, Iy, Iz, J)

    def _build_wall(self, diagonal_scale: float) -> csc_matrix:
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        def add_edge(n0: int, n1: int, tributary: float, scale: float = 1.0) -> None:
            p0, p1 = self.node_xy(n0), self.node_xy(n1)
            k = self._wall_member(p0, p1, tributary, scale)
            dofs = np.r_[np.arange(6 * n0, 6 * n0 + 6), np.arange(6 * n1, 6 * n1 + 6)]
            self._append_dense(rows, cols, data, dofs, k)

        for iy in range(self.ny + 1):
            tributary = self.dy * (0.5 if iy in (0, self.ny) else 1.0)
            for ix in range(self.nx):
                add_edge(self.node(ix, iy), self.node(ix + 1, iy), tributary)
        for ix in range(self.nx + 1):
            tributary = self.dx * (0.5 if ix in (0, self.nx) else 1.0)
            for iy in range(self.ny):
                add_edge(self.node(ix, iy), self.node(ix, iy + 1), tributary)
        if diagonal_scale > 0.0:
            trib = min(self.dx, self.dy)
            for iy in range(self.ny):
                for ix in range(self.nx):
                    add_edge(self.node(ix, iy), self.node(ix + 1, iy + 1), trib, diagonal_scale)
                    add_edge(self.node(ix + 1, iy), self.node(ix, iy + 1), trib, diagonal_scale)

        K = coo_matrix((data, (rows, cols)), shape=(self.ndof, self.ndof)).tocsc()
        scale = max(float(np.max(np.abs(K.diagonal()))), 1.0)
        return K + eye(self.ndof, format="csc") * (scale * 1.0e-11)

    def _nodes_from_selector(self, selector: dict) -> list[int]:
        kind = selector.get("type", "points")
        if kind == "points":
            return sorted({self.nearest_node(p) for p in selector["points"]})
        if kind == "edge":
            edge = selector["edge"]
            if edge == "right":
                return [self.node(self.nx, iy) for iy in range(self.ny + 1)]
            if edge == "left":
                return [self.node(0, iy) for iy in range(self.ny + 1)]
            if edge == "top":
                return [self.node(ix, self.ny) for ix in range(self.nx + 1)]
            if edge == "bottom":
                return [self.node(ix, 0) for ix in range(self.nx + 1)]
        raise ValueError(f"unknown support selector: {selector}")

    def _build_supports(self, supports: dict) -> np.ndarray:
        nodes = self._nodes_from_selector(supports)
        components = supports.get("components", [0, 1, 2, 3, 4, 5])
        return np.array(sorted(6 * n + int(c) for n in nodes for c in components), dtype=int)

    def _build_loads(self, load_cases: Sequence[dict]) -> tuple[list[np.ndarray], np.ndarray]:
        vectors: list[np.ndarray] = []
        weights: list[float] = []
        for case in load_cases:
            f = np.zeros(self.ndof, float)
            for item in case["forces"]:
                node = self.nearest_node(item["point"])
                f[6 * node : 6 * node + 3] += np.asarray(item["value"], float)
            vectors.append(f)
            weights.append(float(case.get("weight", 1.0)))
        weights_arr = np.asarray(weights, float)
        weights_arr /= weights_arr.sum()
        return vectors, weights_arr

    def interpolation(self, point: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        x = float(np.clip(point[0], 0.0, self.width))
        y = float(np.clip(point[1], 0.0, self.height))
        ix = min(int(x / self.dx), self.nx - 1)
        iy = min(int(y / self.dy), self.ny - 1)
        xi = (x - ix * self.dx) / self.dx
        eta = (y - iy * self.dy) / self.dy
        nodes = np.array(
            [self.node(ix, iy), self.node(ix + 1, iy), self.node(ix + 1, iy + 1), self.node(ix, iy + 1)],
            int,
        )
        weights = np.array([(1-xi)*(1-eta), xi*(1-eta), xi*eta, (1-xi)*eta], float)
        return nodes, weights

    def _segment_condensed(self, p0: np.ndarray, p1: np.ndarray, height: float, cubic: bool) -> tuple[np.ndarray, np.ndarray]:
        node_sets, weights = zip(self.interpolation(p0), self.interpolation(p1))
        nodes = np.unique(np.r_[node_sets[0], node_sets[1]])
        lookup = {int(node): i for i, node in enumerate(nodes)}
        B = np.zeros((12, 6 * len(nodes)), float)
        offset = rigid_offset_transform(np.array([0.0, 0.0, height / 2.0]))
        for endpoint in range(2):
            endpoint_nodes = node_sets[endpoint]
            endpoint_weights = weights[endpoint]
            for node, weight in zip(endpoint_nodes, endpoint_weights):
                j = 6 * lookup[int(node)]
                B[6*endpoint:6*endpoint+6, j:j+6] += float(weight) * offset
        if cubic:
            A, Iy, Iz, J = 0.0, 0.0, height / 12.0, height / 3.0
        else:
            A, Iy, Iz, J = height, height**3 / 12.0, 0.0, 0.0
        q0, q1 = np.r_[p0, height / 2.0], np.r_[p1, height / 2.0]
        k = frame_global_stiffness(q0, q1, self.E, self.nu, A, Iy, Iz, J)
        dofs = np.concatenate([np.arange(6*n, 6*n+6) for n in nodes])
        return dofs, B.T @ k @ B

    def rib_components(self, rib: Rib) -> tuple[csc_matrix, csc_matrix]:
        cached = self._rib_cache.get(rib.key)
        if cached is not None:
            return cached
        rows1: list[int] = []; cols1: list[int] = []; data1: list[float] = []
        rows3: list[int] = []; cols3: list[int] = []; data3: list[float] = []
        p0, p1 = np.asarray(rib.p0, float), np.asarray(rib.p1, float)
        for s in range(rib.segments):
            a, b = s / rib.segments, (s + 1) / rib.segments
            q0, q1 = (1-a)*p0+a*p1, (1-b)*p0+b*p1
            dofs, k1 = self._segment_condensed(q0, q1, rib.height, False)
            _, k3 = self._segment_condensed(q0, q1, rib.height, True)
            self._append_dense(rows1, cols1, data1, dofs, k1)
            self._append_dense(rows3, cols3, data3, dofs, k3)
        K1 = coo_matrix((data1, (rows1, cols1)), shape=(self.ndof, self.ndof)).tocsc()
        K3 = coo_matrix((data3, (rows3, cols3)), shape=(self.ndof, self.ndof)).tocsc()
        self._rib_cache[rib.key] = (K1, K3)
        return K1, K3

    def stiffness(self, ribs: Sequence[Rib], thicknesses: Sequence[float]) -> csc_matrix:
        K = self.base_stiffness.copy()
        for rib, t in zip(ribs, thicknesses):
            K1, K3 = self.rib_components(rib)
            K = K + float(t) * K1 + float(t)**3 * K3
        return K

    def analyze(self, ribs: Sequence[Rib], thicknesses: Sequence[float]) -> AnalysisResult:
        K = self.stiffness(ribs, thicknesses)
        Kff = K[self.free_dofs][:, self.free_dofs].tocsc()
        solver = splu(Kff)
        displacements: list[np.ndarray] = []
        compliances: list[float] = []
        for f in self.load_vectors:
            u = np.zeros(self.ndof, float)
            u[self.free_dofs] = solver.solve(f[self.free_dofs])
            displacements.append(u)
            compliances.append(float(f @ u))
        total = float(np.dot(self.load_weights, compliances))
        return AnalysisResult(total, displacements, compliances)

    def compliance_gradient(self, ribs: Sequence[Rib], thicknesses: Sequence[float], result: AnalysisResult) -> np.ndarray:
        gradient = np.zeros(len(ribs), float)
        for e, (rib, t) in enumerate(zip(ribs, thicknesses)):
            K1, K3 = self.rib_components(rib)
            derivative = K1 + 3.0 * float(t)**2 * K3
            gradient[e] = -sum(
                w * float(u @ (derivative @ u))
                for w, u in zip(self.load_weights, result.displacements)
            )
        return gradient

    def candidate_efficiency(self, rib: Rib, thickness: float, result: AnalysisResult) -> float:
        """Return frozen-field stiffness contribution per added rib volume."""
        volume = rib.length * rib.height * float(thickness)
        if volume <= 0.0:
            return -np.inf
        return float(self.candidate_stiffness_energy(rib, thickness, result)/volume)

    def candidate_stiffness_energy(
        self, rib: Rib, thickness: float, result: AnalysisResult
    ) -> float:
        """Return the candidate's weighted frozen-field stiffness energy."""
        K1, K3 = self.rib_components(rib)
        matrix = float(thickness) * K1 + float(thickness)**3 * K3
        energy = sum(
            weight * float(u @ (matrix @ u))
            for weight, u in zip(self.load_weights, result.displacements)
        )
        return float(energy)

    def endpoint_energy_density(self, rib: Rib, result: AnalysisResult) -> float:
        """Return the fast endpoint surrogate used before net-benefit ranking."""
        return endpoint_energy_density_factor(
            self.response_at, self.load_weights, self.E, rib, result
        )

    def response_at(self, displacement: np.ndarray, point: Sequence[float]) -> np.ndarray:
        nodes, weights = self.interpolation(point)
        return sum(float(w) * displacement[6*n:6*n+6] for n, w in zip(nodes, weights))

    def deformation_factor(
        self, rib: Rib, thickness: float, result: AnalysisResult
    ) -> float:
        """Evaluate the configured candidate-ranking measure."""
        if self.deformation_factor_method == "stiffness_per_volume":
            return self.deformation_factor_stiffness_per_volume(
                rib, thickness, result
            )
        if self.deformation_factor_method == "fixed_volume_net_benefit":
            return self.endpoint_energy_density(rib, result)
        return self.deformation_factor_equation_15(rib, result)

    def deformation_factor_stiffness_per_volume(
        self, rib: Rib, thickness: float, result: AnalysisResult
    ) -> float:
        """Preserved pre-Eq.-(15) frozen-energy/volume ranking measure."""
        return self.candidate_efficiency(rib, thickness, result)

    def deformation_factor_equation_15(
        self, rib: Rib, result: AnalysisResult
    ) -> float:
        """Return the relative-displacement/rotation factor in paper Eq. (15)."""
        return equation_15_deformation_factor(
            self.response_at, self.load_weights, rib, result
        )
