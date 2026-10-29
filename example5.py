#!/usr/bin/env python
"""Example V: right-clamped plate under uniform transverse pressure."""

from __future__ import annotations

from rib_layout_core import run_example_cli

CASE_CONFIG = {
    "number": 5,
    "name": "uniform_transverse_pressure",
    "domain": [100.0, 50.0],
    "mesh": [80, 40],
    "mirror_symmetry": ["y"],
    "wall_thickness": 1.0,
    "initial_rib_cell_size": 25.0,
    "material": {
        "name": "AlSi10MnMg structural casting aluminium alloy",
        "E": 70_000.0, "nu": 0.33,
    },
    # Fixed small-deflection scale for all revision comparisons: -0.5 N total.
    # Backup-solver pilot motivated reduction from the proposed 0.01 N/mm².
    "load_cases": [{"weight": 1.0, "forces": [],
                    "pressure": {"type": "uniform", "value": -0.0001}}],
    # Clamp base-plate DOFs only; rib-top nodes retain production free-top BCs.
    "supports": {"type": "edge", "edge": "right"},
    # Rib top is 5 mm above the base midsurface (4.5 mm above its upper face).
    "rib": {"height": 5.0, "initial": 0.5, "upper": 3.0},
    "volume_bound": 1500.0,
    "rationalization_relaxation": 0.05,
    "algorithm": {"active_set_max_iterations": 10, "sizing_max_iterations": 100},
}

OUTPUT_OPTIONS = {
    "directory_name": "example_5", "roman": "V",
    "title": "Example V — uniform transverse pressure",
    "description": "100x50 mm plate, right clamped, uniform pressure 0.0001 N/mm²",
    "detail_profile": "", "write_stage_images": True,
}


def main(argv=None) -> int:
    return run_example_cli(CASE_CONFIG, OUTPUT_OPTIONS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
