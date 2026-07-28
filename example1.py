#!/usr/bin/env python
"""Example I: 20 x 20 plate with a downward corner load."""

from __future__ import annotations

from rib_layout_core import run_example_cli


# Everything below is specific to Example I. Shared algorithm settings are in
# rib_layout_core.py; runtime/thread settings are in rib_layout_env.py.
CASE_CONFIG = {
    "number": 1,
    "name": "corner_load",
    # Finite-element model dimensions (mm) and mesh.
    "domain": [20.0, 20.0],
    "mirror_symmetry": [],
    "mesh": [20, 20],
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
        "points": [[0.0, 0.0], [0.0, 20.0]],
    },
    # Rib design limits and material-volume constraint.
    "rib": {"height": 2.0, "initial": 0.2, "upper": 2.0},
    "volume_bound": 45.0,
    "rationalization_relaxation": 0.05,
    "algorithm": {"active_set_max_iterations": 8},
}


# Example-specific presentation and output selection.
OUTPUT_OPTIONS = {
    "directory_name": "example_1",
    "roman": "I",
    "title": "Example I — corner load",
    "description": "20x20 plate, corner load, rib height 2 mm",
    "detail_profile": "",
    "write_stage_images": True,
}


def main(argv=None) -> int:
    return run_example_cli(CASE_CONFIG, OUTPUT_OPTIONS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
