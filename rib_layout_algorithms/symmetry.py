"""Mirror-symmetry utilities for discrete and continuous rib-layout design."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .model import Rib


def mirror_axes(config: dict) -> tuple[str, ...]:
    """Return validated coordinate-reflection axes (``x`` and/or ``y``)."""
    axes = tuple(str(axis).lower() for axis in config.get("mirror_symmetry", ()))
    if len(set(axes)) != len(axes) or any(axis not in {"x", "y"} for axis in axes):
        raise ValueError("mirror_symmetry must contain unique axes 'x' and/or 'y'")
    return axes


def mirror_rib(rib: Rib, axis: str, width: float, height: float) -> Rib:
    """Reflect a rib about the corresponding domain centerline."""
    axis = str(axis).lower()
    if axis == "x":
        transform = lambda point: (float(width-point[0]), float(point[1]))
    elif axis == "y":
        transform = lambda point: (float(point[0]), float(height-point[1]))
    else:
        raise ValueError("mirror axis must be 'x' or 'y'")
    return Rib(
        transform(rib.p0), transform(rib.p1), rib.height,
        rib.name, rib.segments,
    )


def _canonical_endpoints(rib: Rib) -> np.ndarray:
    endpoints = sorted((tuple(rib.p0), tuple(rib.p1)))
    return np.asarray(endpoints, float)


def rib_distance(first: Rib, second: Rib) -> float:
    """Return the maximum endpoint mismatch after orientation normalization."""
    if first.segments != second.segments or not np.isclose(first.height, second.height):
        return np.inf
    return float(np.max(np.abs(
        _canonical_endpoints(first)-_canonical_endpoints(second)
    )))


def mirror_partner_index(
    ribs: Sequence[Rib],
    index: int,
    axis: str,
    width: float,
    height: float,
    tolerance: float,
) -> int | None:
    """Locate one reflected partner, allowing small optimization drift."""
    target = mirror_rib(ribs[index], axis, width, height)
    distances = np.asarray([rib_distance(target, rib) for rib in ribs], float)
    partner = int(np.argmin(distances))
    return partner if distances[partner] <= tolerance else None


def mirror_groups(
    ribs: Sequence[Rib],
    axes: Sequence[str],
    width: float,
    height: float,
    tolerance: float | None = None,
) -> list[list[int]]:
    """Partition the available ribs into connected mirror orbits."""
    if tolerance is None:
        tolerance = 1.0e-7*max(float(width), float(height), 1.0)
    parent = list(range(len(ribs)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        a, b = find(first), find(second)
        if a != b:
            parent[max(a, b)] = min(a, b)

    exact_lookup = {rib.key: index for index, rib in enumerate(ribs)}
    for index in range(len(ribs)):
        for axis in axes:
            target = mirror_rib(ribs[index], axis, width, height)
            partner = exact_lookup.get(target.key)
            if partner is None:
                partner = mirror_partner_index(
                    ribs, index, axis, width, height, float(tolerance)
                )
            if partner is not None:
                union(index, partner)
    grouped: dict[int, list[int]] = {}
    for index in range(len(ribs)):
        grouped.setdefault(find(index), []).append(index)
    return [grouped[key] for key in sorted(grouped)]


def missing_mirror_partners(
    ribs: Sequence[Rib],
    axes: Sequence[str],
    width: float,
    height: float,
    tolerance: float | None = None,
) -> list[tuple[str, str]]:
    """List ``(rib name, axis)`` pairs whose reflected rib is absent."""
    if tolerance is None:
        tolerance = 1.0e-7*max(float(width), float(height), 1.0)
    missing: list[tuple[str, str]] = []
    exact_keys = {rib.key for rib in ribs}
    for index, rib in enumerate(ribs):
        for axis in axes:
            target = mirror_rib(rib, axis, width, height)
            if target.key in exact_keys:
                continue
            if mirror_partner_index(
                ribs, index, axis, width, height, float(tolerance)
            ) is None:
                missing.append((rib.name, axis))
    return missing


@dataclass
class MirrorVariableMap:
    """Reduced thickness/coordinate variables for mirror-related ribs."""

    rib_groups: list[list[int]]
    thickness_group: np.ndarray
    coordinate_variable: np.ndarray
    coordinate_sign: np.ndarray
    coordinate_offset: np.ndarray
    coordinate_fixed: np.ndarray
    coordinate_representatives: list[int]

    @property
    def thickness_count(self) -> int:
        return len(self.rib_groups)

    @property
    def coordinate_count(self) -> int:
        return len(self.coordinate_representatives)

    def reduce_thicknesses(self, values: Sequence[float]) -> np.ndarray:
        full = np.asarray(values, float)
        return np.asarray([
            float(np.mean(full[group])) for group in self.rib_groups
        ])

    def expand_thicknesses(self, reduced: Sequence[float]) -> np.ndarray:
        return np.asarray(reduced, float)[self.thickness_group]

    def reduce_thickness_gradient(self, gradient: Sequence[float]) -> np.ndarray:
        full = np.asarray(gradient, float)
        return np.asarray([float(np.sum(full[group])) for group in self.rib_groups])

    def reduce_coordinates(self, coordinates: Sequence[float]) -> np.ndarray:
        full = np.asarray(coordinates, float)
        reduced = np.empty(self.coordinate_count, float)
        for variable in range(self.coordinate_count):
            nodes = np.flatnonzero(self.coordinate_variable == variable)
            reduced[variable] = float(np.mean(
                self.coordinate_sign[nodes]
                * (full[nodes]-self.coordinate_offset[nodes])
            ))
        return reduced

    def expand_coordinates(self, reduced: Sequence[float]) -> np.ndarray:
        values = np.asarray(self.coordinate_fixed, float).copy()
        moving = self.coordinate_variable >= 0
        variables = np.asarray(reduced, float)
        values[moving] = (
            self.coordinate_sign[moving]*variables[self.coordinate_variable[moving]]
            + self.coordinate_offset[moving]
        )
        return values

    def reduce_coordinate_gradient(self, gradient: Sequence[float]) -> np.ndarray:
        full = np.asarray(gradient, float)
        reduced = np.zeros(self.coordinate_count, float)
        moving = self.coordinate_variable >= 0
        np.add.at(
            reduced,
            self.coordinate_variable[moving],
            full[moving]*self.coordinate_sign[moving],
        )
        return reduced

    def reduce_coordinate_bounds(self, bounds: np.ndarray) -> np.ndarray:
        full = np.asarray(bounds, float).reshape(-1, 2)
        reduced = np.empty((self.coordinate_count, 2), float)
        for variable in range(self.coordinate_count):
            nodes = np.flatnonzero(self.coordinate_variable == variable)
            lower, upper = -np.inf, np.inf
            for node in nodes:
                sign = self.coordinate_sign[node]
                offset = self.coordinate_offset[node]
                if sign > 0.0:
                    node_lower = full[node, 0]-offset
                    node_upper = full[node, 1]-offset
                else:
                    node_lower = offset-full[node, 1]
                    node_upper = offset-full[node, 0]
                lower = max(lower, node_lower)
                upper = min(upper, node_upper)
            if lower > upper+1.0e-12:
                raise ValueError("mirror coordinate bounds have an empty intersection")
            reduced[variable] = [lower, upper]
        return reduced

    def reduce_coordinate_scale(self, scale: Sequence[float]) -> np.ndarray:
        full = np.asarray(scale, float)
        return np.asarray([
            float(np.min(full[self.coordinate_variable == variable]))
            for variable in range(self.coordinate_count)
        ])

    def reduce_coordinate_quadratic_scale(
        self, scale: Sequence[float]
    ) -> np.ndarray:
        """Preserve a sum of full squared normalized coordinate changes."""
        full = np.asarray(scale, float)
        return np.asarray([
            float(1.0/np.sqrt(np.sum(
                1.0/full[self.coordinate_variable == variable]**2
            )))
            for variable in range(self.coordinate_count)
        ])


def build_mirror_variable_map(
    ribs: Sequence[Rib],
    axes: Sequence[str],
    width: float,
    height: float,
    tolerance: float | None = None,
) -> MirrorVariableMap:
    """Construct exact affine mirror relations for all continuous variables."""
    if tolerance is None:
        tolerance = 1.0e-7*max(float(width), float(height), 1.0)
    groups = mirror_groups(ribs, axes, width, height, tolerance)
    thickness_group = np.empty(len(ribs), int)
    for group_index, group in enumerate(groups):
        thickness_group[group] = group_index

    count = 4*len(ribs)
    graph: list[list[tuple[int, float, float]]] = [[] for _ in range(count)]

    def add_relation(source: int, target: int, sign: float, offset: float) -> None:
        # target = sign*source + offset
        graph[source].append((target, sign, offset))
        graph[target].append((source, sign, -sign*offset))

    for rib_index, rib in enumerate(ribs):
        for axis in axes:
            partner = mirror_partner_index(
                ribs, rib_index, axis, width, height, float(tolerance)
            )
            if partner is None:
                continue
            reflected = mirror_rib(rib, axis, width, height)
            reflected_points = np.asarray([reflected.p0, reflected.p1], float)
            partner_points = np.asarray([ribs[partner].p0, ribs[partner].p1], float)
            same_error = float(np.max(np.abs(reflected_points-partner_points)))
            swapped_error = float(np.max(np.abs(reflected_points[::-1]-partner_points)))
            swapped = swapped_error < same_error
            for endpoint in range(2):
                target_endpoint = 1-endpoint if swapped else endpoint
                for coordinate in range(2):
                    source_node = 4*rib_index+2*endpoint+coordinate
                    target_node = 4*partner+2*target_endpoint+coordinate
                    reflected_coordinate = (
                        (axis == "x" and coordinate == 0)
                        or (axis == "y" and coordinate == 1)
                    )
                    sign = -1.0 if reflected_coordinate else 1.0
                    offset = (
                        width if axis == "x" and coordinate == 0
                        else height if axis == "y" and coordinate == 1
                        else 0.0
                    )
                    add_relation(source_node, target_node, sign, float(offset))

    coordinate_variable = np.full(count, -2, int)
    coordinate_sign = np.zeros(count, float)
    coordinate_offset = np.zeros(count, float)
    coordinate_fixed = np.full(count, np.nan, float)
    representatives: list[int] = []
    for start in range(count):
        if coordinate_variable[start] != -2:
            continue
        component: list[int] = []
        signs = {start: 1.0}
        offsets = {start: 0.0}
        stack = [start]
        fixed_root: list[float] = []
        while stack:
            source = stack.pop()
            component.append(source)
            for target, edge_sign, edge_offset in graph[source]:
                predicted_sign = edge_sign*signs[source]
                predicted_offset = edge_sign*offsets[source]+edge_offset
                if target not in signs:
                    signs[target] = predicted_sign
                    offsets[target] = predicted_offset
                    stack.append(target)
                    continue
                coefficient = signs[target]-predicted_sign
                constant = predicted_offset-offsets[target]
                if abs(coefficient) > 1.0e-12:
                    fixed_root.append(constant/coefficient)
                elif abs(constant) > float(tolerance):
                    raise ValueError("inconsistent mirror coordinate relations")
        if fixed_root:
            root_value = float(np.mean(fixed_root))
            if np.ptp(fixed_root) > float(tolerance):
                raise ValueError("inconsistent fixed mirror coordinate")
            for node in component:
                coordinate_variable[node] = -1
                coordinate_sign[node] = signs[node]
                coordinate_offset[node] = offsets[node]
                coordinate_fixed[node] = signs[node]*root_value+offsets[node]
        else:
            variable = len(representatives)
            representatives.append(start)
            for node in component:
                coordinate_variable[node] = variable
                coordinate_sign[node] = signs[node]
                coordinate_offset[node] = offsets[node]

    return MirrorVariableMap(
        groups,
        thickness_group,
        coordinate_variable,
        coordinate_sign,
        coordinate_offset,
        coordinate_fixed,
        representatives,
    )
