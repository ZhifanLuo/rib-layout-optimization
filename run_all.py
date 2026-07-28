#!/usr/bin/env python
"""Run the four published examples sequentially with the current interpreter."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Callable, Sequence


CODE_ROOT = Path(__file__).resolve().parent
EXAMPLE_NUMBERS = (1, 2, 3, 4)


def run_examples(
    *,
    quick: bool = False,
    output: Path | None = None,
    geometry_sweeps: int | None = None,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> int:
    """Run every example in order and return the first nonzero exit code."""
    for number in EXAMPLE_NUMBERS:
        command = [sys.executable, str(CODE_ROOT / f"example{number}.py")]
        if quick:
            command.append("--quick")
        if output is not None:
            command.extend(("--output", str(output)))
        if geometry_sweeps is not None:
            command.extend(("--geometry-sweeps", str(geometry_sweeps)))

        print(f"Starting Example {number}...", flush=True)
        completed = runner(command, cwd=CODE_ROOT, check=False)
        if completed.returncode != 0:
            print(
                f"Example {number} failed with exit code "
                f"{completed.returncode}; remaining examples were not started.",
                file=sys.stderr,
                flush=True,
            )
            return int(completed.returncode)
        print(f"Example {number} completed successfully.", flush=True)

    print("All four examples completed successfully.", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Examples 1-4 sequentially and stop on the first failure"
        )
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use a reduced mesh for cases that define one (diagnostic only)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="override the shared output root passed to every example",
    )
    parser.add_argument(
        "--geometry-sweeps",
        type=int,
        default=None,
        help="override geometry iterations in every example (diagnostic only)",
    )
    args = parser.parse_args(argv)
    return run_examples(
        quick=args.quick,
        output=args.output,
        geometry_sweeps=args.geometry_sweeps,
    )


if __name__ == "__main__":
    raise SystemExit(main())
