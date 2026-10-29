from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from example5 import CASE_CONFIG
import example5
from rib_layout_core import load_case
from tools import reproduce_paper


class PaperReproductionTests(unittest.TestCase):
    def test_standard_example5_cli_uses_same_case_and_output_contract(self):
        with mock.patch.object(example5, "run_example_cli", return_value=0) as entry:
            self.assertEqual(example5.main(["--output", "new-output"]), 0)
        entry.assert_called_once_with(example5.CASE_CONFIG, example5.OUTPUT_OPTIONS,
                                      ["--output", "new-output"])

    def test_paper_configuration_matches_canonical_case_defaults(self):
        spec = reproduce_paper.build_spec()
        cfg = spec["config"]
        self.assertEqual(cfg["mesh"], [80, 40])
        self.assertEqual(cfg["wall_thickness"], 1.0)
        self.assertEqual(cfg["domain"], [100.0, 50.0])
        self.assertEqual(cfg["supports"], {"type": "edge", "edge": "right"})
        self.assertEqual(cfg["load_cases"][0]["pressure"]["value"], -0.0001)
        self.assertEqual(cfg["algorithm"]["rationalization_max_iterations"], 100)
        self.assertEqual(cfg["algorithm"]["rationalization_min_iterations"], 11)
        self.assertIsNone(spec["iteration_cap"])
        self.assertEqual(CASE_CONFIG["mesh"], [80, 40])
        self.assertEqual(CASE_CONFIG["wall_thickness"], 1.0)
        self.assertEqual(load_case(5), cfg)

    def test_dry_plan_does_not_create_output_or_run_optimization(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "new-output"
            with mock.patch.object(reproduce_paper, "execute_trial") as execute:
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(reproduce_paper.main(["--output", str(out)]), 0)
            execute.assert_not_called()
            self.assertFalse(out.exists())

    def test_existing_and_in_source_outputs_are_rejected_before_execution(self):
        for out in (reproduce_paper.CODE_ROOT / "results/new-paper-run", reproduce_paper.CODE_ROOT):
            with self.subTest(output=out), self.assertRaises(ValueError):
                reproduce_paper.output_path(out)
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            marker = out / "keep.txt"
            marker.write_text("existing evidence", encoding="utf-8")
            with self.assertRaises(ValueError):
                reproduce_paper.output_path(out)
            self.assertEqual(marker.read_text(encoding="utf-8"), "existing evidence")

    def test_execution_records_failure_without_overwriting_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "failed-run"
            with mock.patch.object(reproduce_paper, "source_provenance", return_value={"source_aggregate_sha256": "snapshot"}):
                with mock.patch.object(reproduce_paper, "execute_trial", side_effect=RuntimeError("solver failure")):
                    with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "solver failure"):
                        reproduce_paper.main(["--output", str(out), "--execute"])
            record = json.loads((out / "reproduction.json").read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["error_type"], "RuntimeError")
            self.assertIn("config_sha256", record["plan"])
            with self.assertRaises(ValueError):
                reproduce_paper.output_path(out)

    def test_execution_detects_source_changes_and_retains_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "source-changed"
            with mock.patch.object(reproduce_paper, "source_provenance", side_effect=[
                {"source_aggregate_sha256": "before"}, {"source_aggregate_sha256": "after"}
            ]), mock.patch.object(reproduce_paper, "execute_trial", return_value={"result": "mock"}):
                with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "Source changed"):
                    reproduce_paper.main(["--output", str(out), "--execute"])
            self.assertEqual(json.loads((out / "reproduction.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
