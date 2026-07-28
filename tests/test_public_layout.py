from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rib_layout_core import load_case, make_case_config
from rib_layout_output import save_optimization_results
from rib_layout_algorithms.model import Rib
from rib_layout_algorithms.optimization import OptimizationRun, Stage


class PublicLayoutTests(unittest.TestCase):
    def test_example_modules_match_the_validated_case_configurations(self):
        for number in range(1, 5):
            with self.subTest(example=number):
                module = importlib.import_module(f"example{number}")
                self.assertEqual(
                    make_case_config(module.CASE_CONFIG),
                    load_case(number),
                )
                self.assertEqual(
                    module.OUTPUT_OPTIONS["directory_name"],
                    f"example_{number}",
                )
                self.assertEqual(module.OUTPUT_OPTIONS["detail_profile"], "")
                self.assertTrue(module.OUTPUT_OPTIONS["write_stage_images"])

    def test_all_executable_cases_use_alsi10mnmg_structural_casting_material(self):
        expected = {
            "name": "AlSi10MnMg structural casting aluminium alloy",
            "E": 70_000.0,
            "nu": 0.33,
        }
        for number in range(1, 5):
            with self.subTest(example=number):
                module = importlib.import_module(f"example{number}")
                self.assertEqual(module.CASE_CONFIG["material"], expected)
                self.assertEqual(load_case(number)["material"], expected)

    def test_quick_mesh_is_applied_only_when_the_case_defines_one(self):
        example1 = importlib.import_module("example1")
        example3 = importlib.import_module("example3")
        self.assertEqual(
            make_case_config(example1.CASE_CONFIG, quick=True)["mesh"],
            [20, 20],
        )
        quick3 = make_case_config(example3.CASE_CONFIG, quick=True)
        self.assertEqual(quick3["mesh"], [32, 16])
        self.assertEqual(quick3["rib"]["segments"], 4)

    def test_output_manager_writes_checkpoint_and_summaries(self):
        rib = Rib((0.0, 0.0), (1.0, 0.0), 1.0, "R1", 1)
        stage = Stage(
            "initial_sizing", [rib], np.array([0.2]), 1.5, 2, "test"
        )
        run = OptimizationRun(stages=[stage], log=["complete"])
        optimizer = SimpleNamespace(
            analysis_count=2,
            rationalization_history=[],
            volume=lambda ribs, thicknesses: float(sum(
                item.length*item.height*value
                for item, value in zip(ribs, thicknesses)
            )),
        )
        artifacts = SimpleNamespace(
            config={
                "number": 1,
                "name": "test",
                "mesh": [1, 1],
                "linear_solver": "auto",
                "linear_solver_threads": 1,
                "algorithm": {"sensitivity_workers": 1},
            },
            optimizer=optimizer,
            run=run,
            elapsed_seconds=0.1,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = save_optimization_results(
                artifacts,
                Path(directory),
                {
                    "directory_name": "example_1",
                    "description": "test case",
                    "write_stage_images": False,
                },
                quick=True,
            )
            self.assertTrue((output / "checkpoint_results.json").is_file())
            self.assertTrue((output / "results.json").is_file())
            self.assertTrue((output / "summary.csv").is_file())
            self.assertTrue((output / "case_summary.csv").is_file())

    def test_common_output_removes_stale_case_specific_artifacts(self):
        rib = Rib((0.0, 0.0), (1.0, 0.0), 1.0, "R1", 1)
        stage = Stage("geometry", [rib], np.array([0.2]), 1.0, 1, "test")
        artifacts = SimpleNamespace(
            config={
                "number": 4,
                "name": "test",
                "mesh": [1, 1],
                "linear_solver": "auto",
                "linear_solver_threads": 1,
                "algorithm": {"sensitivity_workers": 1},
            },
            optimizer=SimpleNamespace(
                analysis_count=1,
                rationalization_history=[],
                volume=lambda ribs, values: 0.2,
            ),
            run=OptimizationRun(stages=[stage], log=[]),
            elapsed_seconds=0.1,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"example_4"
            output.mkdir()
            stale_file = output/"six_stage_summary.csv"
            stale_file.write_text("obsolete", encoding="utf-8")
            stale_directory = output/"detailed_history_pages"
            stale_directory.mkdir()
            (stale_directory/"obsolete.txt").write_text("obsolete")
            save_optimization_results(
                artifacts,
                Path(directory),
                {
                    "directory_name": "example_4",
                    "detail_profile": "",
                    "write_stage_images": False,
                },
                quick=True,
            )
            self.assertFalse(stale_file.exists())
            self.assertFalse(stale_directory.exists())

    def test_output_manager_serializes_each_rationalization_pass(self):
        rib = Rib((0.0, 0.0), (1.0, 0.0), 1.0, "R1", 1)
        stages = [
            Stage("rationalized", [rib], np.array([0.2]), 1.0, 3),
            Stage("further_rationalized", [rib], np.array([0.2]), 1.02, 4),
        ]
        histories = {
            "rationalized": [{"event": "first"}],
            "further_rationalized": [{"event": "second"}],
        }
        flat_history = [
            {"pass_name": "rationalized", "event": "first"},
            {"pass_name": "further_rationalized", "event": "second"},
        ]
        artifacts = SimpleNamespace(
            config={
                "number": 3,
                "name": "test",
                "mesh": [1, 1],
                "linear_solver": "auto",
                "linear_solver_threads": 1,
                "algorithm": {"sensitivity_workers": 1},
            },
            optimizer=SimpleNamespace(
                analysis_count=7,
                rationalization_history=flat_history,
                volume=lambda ribs, values: 0.2,
            ),
            run=OptimizationRun(
                stages=stages,
                rationalization_histories=histories,
                log=[],
            ),
            elapsed_seconds=0.1,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = save_optimization_results(
                artifacts,
                Path(directory),
                {
                    "directory_name": "example_3",
                    "write_stage_images": False,
                },
                quick=True,
            )
            payload = json.loads(
                (output/"results.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["rationalization_histories"], histories)
            self.assertTrue(
                (output/"rationalization_history_rationalized.json").is_file()
            )
            self.assertTrue(
                (
                    output/
                    "rationalization_history_further_rationalized.json"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
