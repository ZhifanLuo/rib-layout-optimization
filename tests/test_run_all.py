from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import run_all


class RunAllTests(unittest.TestCase):
    def test_runs_all_examples_sequentially_with_the_current_interpreter(self):
        calls = []

        def fake_runner(command, **kwargs):
            calls.append((command, kwargs))
            return subprocess.CompletedProcess(command, 0)

        status = run_all.run_examples(runner=fake_runner)

        self.assertEqual(status, 0)
        self.assertEqual(len(calls), 4)
        for number, (command, kwargs) in enumerate(calls, start=1):
            self.assertEqual(command[0], sys.executable)
            self.assertEqual(Path(command[1]).name, f"example{number}.py")
            self.assertEqual(kwargs["cwd"], run_all.CODE_ROOT)
            self.assertFalse(kwargs["check"])

    def test_stops_immediately_after_the_first_failed_example(self):
        calls = []

        def fake_runner(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(
                command, 17 if len(calls) == 2 else 0
            )

        status = run_all.run_examples(runner=fake_runner)

        self.assertEqual(status, 17)
        self.assertEqual(
            [Path(command[1]).name for command in calls],
            ["example1.py", "example2.py"],
        )

    def test_forwards_diagnostic_and_output_options_to_each_example(self):
        calls = []

        def fake_runner(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0)

        output = Path("custom-results")
        status = run_all.run_examples(
            quick=True,
            output=output,
            geometry_sweeps=0,
            runner=fake_runner,
        )

        self.assertEqual(status, 0)
        for command in calls:
            self.assertEqual(
                command[2:],
                [
                    "--quick",
                    "--output",
                    str(output),
                    "--geometry-sweeps",
                    "0",
                ],
            )


if __name__ == "__main__":
    unittest.main()
