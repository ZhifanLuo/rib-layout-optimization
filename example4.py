#!/usr/bin/env python
"""Example IV: 200 x 100 plate under two load cases."""

from __future__ import annotations

from rib_layout_core import run_example_cli


# Everything below is specific to Example IV.
CASE_CONFIG = {
    "number": 4,
    "name": "force_couple_and_transverse_load",
    # Finite-element model dimensions (mm), formal mesh, and optional quick mesh.
    "domain": [200.0, 100.0],
    # Mid-height reflection reverses the force couple but preserves compliance.
    "mirror_symmetry": ["y"],
    "mesh": [80, 40],
    "quick_mesh": [32, 16],
    "quick_rib_segments": 4,
    "wall_thickness": 1.0,
    "initial_rib_cell_size": 25.0,
    # AlSi10MnMg structural casting alloy: N and mm units (E = 70 GPa).
    "material": {
        "name": "AlSi10MnMg structural casting aluminium alloy",
        "E": 70_000.0,
        "nu": 0.33,
    },
    # Load case 1: equal and opposite transverse forces form the applied couple.
    # Load case 2: the same two points and magnitudes, both directed along -Y.
    # Equal weights give the two load cases equal importance in the objective.
    "load_cases": [
        {
            "weight": 1.0,
            "forces": [
                {"point": [0.0, 100.0], "value": [0.0, 0.0, 100.0]},
                {"point": [0.0, 0.0], "value": [0.0, 0.0, -100.0]},
            ],
        },
        {
            "weight": 1.0,
            "forces": [
                {"point": [0.0, 100.0], "value": [0.0, -100.0, 0.0]},
                {"point": [0.0, 0.0], "value": [0.0, -100.0, 0.0]},
            ],
        },
    ],
    "supports": {"type": "edge", "edge": "right"},
    # Rib design limits and material-volume constraint.
    "rib": {"height": 10.0, "initial": 0.5, "upper": 3.0},
    "volume_bound": 11_310.0,
    "rationalization_relaxation": 0.05,
    # Repeat rationalization from the first rationalized design to demonstrate
    # that the same simplification step can be applied successively.
    "further_rationalization_relaxation": 0.05,
    "algorithm": {
        "filter_tolerance": 0.01,
        "sizing_max_iterations": 60,
        "active_set_max_iterations": 6,
    },
}


# Example-specific presentation and output selection.
OUTPUT_OPTIONS = {
    "directory_name": "example_4",
    "roman": "IV",
    "title": "Example IV — two load cases",
    "description": "200x100 plate, force couple plus negative-Y load case, rib height 10 mm",
    "detail_profile": "",
    "write_stage_images": True,
}


def main(argv=None) -> int:
    return run_example_cli(CASE_CONFIG, OUTPUT_OPTIONS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
