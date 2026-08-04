#!/usr/bin/env python
"""Example II: 40 x 20 plate with a downward lower-midspan load."""

from __future__ import annotations

from rib_layout_core import run_example_cli


# Everything below is specific to Example II.
CASE_CONFIG = {
    "number": 2,
    "name": "lower_midspan_load",
    # Finite-element model dimensions (mm) and mesh.
    "domain": [40.0, 20.0],
    # Reflection of x about the vertical centerline x=20 mm.
    "mirror_symmetry": ["x"],
    "mesh": [40, 20],
    "wall_thickness": 0.01,
    "initial_rib_cell_size": 10.0,
    # AlSi10MnMg structural casting alloy: N and mm units (E = 70 GPa).
    "material": {
        "name": "AlSi10MnMg structural casting aluminium alloy",
        "E": 70_000.0,
        "nu": 0.33,
    },
    # Load and support boundary conditions.
    "load_cases": [{
        "weight": 1.0,
        "forces": [{
            "point": [20.0, 0.0],
            "value": [0.0, -100.0, 0.0],
        }],
    }],
    "supports": {
        "type": "points",
        "points": [[0.0, 0.0], [40.0, 0.0]],
    },
    # Rib design limits and material-volume constraint.
    "rib": {"height": 2.0, "initial": 0.2, "upper": 2.0},
    "volume_bound": 90.0,
    "rationalization_relaxation": 0.05,
    "algorithm": {},
}


# Example-specific presentation and output selection.
OUTPUT_OPTIONS = {
    "directory_name": "example_2",
    "roman": "II",
    "title": "Example II — lower-midspan load",
    "description": "40x20 plate, lower-midspan load, rib height 2 mm",
    "detail_profile": "",
    "write_stage_images": True,
}


def main(argv=None) -> int:
    return run_example_cli(CASE_CONFIG, OUTPUT_OPTIONS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
