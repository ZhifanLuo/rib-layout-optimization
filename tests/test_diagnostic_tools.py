from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from rib_layout_algorithms.model import Rib
from tools.run_full_pool_baseline import build_parser as full_pool_parser
from tools.run_geometry_restart_diagnostic import (
    LEGACY_HISTORY_FIELDS,
    build_parser as restart_parser,
    legacy_single_restart_payload,
    write_legacy_history,
)
from tools.run_topology_lifting_diagnostic import (
    build_parser as lifting_parser,
    validate_saved_compliance,
    validate_source_metadata,
    lifted_design,
)


class DiagnosticToolTests(unittest.TestCase):
    def test_restart_parser_defaults_are_deterministic(self):
        args = restart_parser().parse_args([
            "--case", "3", "--source", "results.json", "--output", "audit",
        ])
        self.assertEqual(args.restarts, 1)
        self.assertEqual(args.multistarts, 0)
        self.assertEqual(args.seed, 20260803)
        self.assertAlmostEqual(args.thickness_perturbation, 0.10)

    def test_full_pool_parser_accepts_multiple_cases(self):
        args = full_pool_parser().parse_args([
            "--case", "1", "2", "--output", "audit", "--quick",
        ])
        self.assertEqual(args.case, [1, 2])
        self.assertTrue(args.quick)

    def test_lifting_reuses_reduced_state_and_adds_deleted_at_lower_bound(self):
        full = [
            Rib((0.0, 0.0), (1.0, 0.0), 2.0, "A"),
            Rib((0.0, 1.0), (1.0, 1.0), 2.0, "B"),
        ]
        moved_a = Rib((0.0, 0.1), (1.0, 0.1), 2.0, "A")
        ribs, thicknesses, names = lifted_design(
            full, np.array([0.2, 0.3]), [moved_a], np.array([0.4]), 1.0e-6
        )
        self.assertEqual([rib.name for rib in ribs], ["A", "B"])
        self.assertEqual(ribs[0], moved_a)
        self.assertTrue(np.allclose(thicknesses, [0.4, 1.0e-6]))
        self.assertEqual(names, ["B"])

    def test_lifting_parser_exposes_stage_selection(self):
        args = lifting_parser().parse_args([
            "--case", "4", "--source", "results.json", "--output", "audit",
            "--reduced-stage", "further_rationalized",
        ])
        self.assertEqual(args.full_stage, "geometry")
        self.assertEqual(args.reduced_stage, "further_rationalized")
        self.assertEqual(args.source_compliance_tolerance, 1.0e-6)

    def test_lifting_rejects_incompatible_metadata_and_compliance(self):
        cfg = {"mesh": [40, 20]}
        source = {"example": 2, "quick": False, "mesh_used": [40, 20]}
        validate_source_metadata(source, 2, cfg)
        with self.assertRaisesRegex(ValueError, "case mismatch"):
            validate_source_metadata({**source, "example": 3}, 2, cfg)
        with self.assertRaisesRegex(ValueError, "mesh mismatch"):
            validate_source_metadata(
                {**source, "mesh_used": [20, 10]}, 2, cfg
            )
        self.assertAlmostEqual(
            validate_saved_compliance("geometry", 10.0, 10.000001, 1.0e-6),
            1.0e-7,
        )
        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            validate_saved_compliance("geometry", 10.0, 10.1, 1.0e-6)

    def test_single_restart_payload_and_csv_retain_legacy_schema(self):
        record = {
            "initial_compliance": 2.0,
            "geometry_analysis_count": 3,
            "outer_records": 2,
            "initial_volume": 1.0,
            "final_volume": 1.0,
            "final_compliance": 1.5,
            "relative_compliance_change": -0.25,
            "initial_ribs": [],
            "final_ribs": [],
            "iteration_history": [],
            "log": [],
        }
        payload = legacy_single_restart_payload(
            case=1, source=Path("source.json"), stage_name="geometry",
            mesh=[20, 20], rib_count=1, saved_compliance=2.0, record=record,
        )
        self.assertEqual(payload["restart_compliance"], 2.0)
        self.assertEqual(payload["restart_initialization_fea"], 1)
        self.assertEqual(payload["geometry_outer_fea"], 3)
        self.assertEqual(payload["outer_iterations"], 2)

        rib = Rib((0.0, 0.0), (1.0, 0.0), 2.0, "A")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"legacy.csv"
            write_legacy_history(
                path, [], [rib], np.array([0.2]), 2.0, 0.4, 1.0, 0.001
            )
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
        self.assertEqual(rows[0], [*LEGACY_HISTORY_FIELDS, "t_A"])
        self.assertEqual(rows[1][0], "0")

    def test_diagnostic_help_supports_module_and_direct_invocation(self):
        code_root = Path(__file__).resolve().parents[1]
        modules = (
            "run_geometry_restart_diagnostic",
            "run_topology_lifting_diagnostic",
            "run_full_pool_baseline",
        )
        for name in modules:
            commands = (
                [sys.executable, "-B", "-m", f"tools.{name}", "--help"],
                [sys.executable, "-B", str(code_root/"tools"/f"{name}.py"), "--help"],
            )
            for command in commands:
                completed = subprocess.run(
                    command, cwd=code_root, capture_output=True, text=True,
                    timeout=30,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("usage:", completed.stdout.lower())


if __name__ == "__main__":
    unittest.main()
