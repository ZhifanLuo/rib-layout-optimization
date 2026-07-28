"""Shared runtime and path configuration for all four examples.

This module is imported before NumPy, SciPy, or Pardiso.  Keeping these
settings here makes every example use the same deterministic numerical
environment and the same project-relative output location.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results"
VIRTUAL_ENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

LINEAR_SOLVER = "auto"
LINEAR_SOLVER_THREADS = 1
SENSITIVITY_WORKERS = 1

# Internal compatibility names used by the FE solver.
DETERMINISTIC_LINEAR_SOLVER_THREADS = LINEAR_SOLVER_THREADS
DETERMINISTIC_SENSITIVITY_WORKERS = SENSITIVITY_WORKERS


def configure_runtime() -> None:
    """Pin numerical-library threads for reproducible formal runs."""
    os.environ["MKL_NUM_THREADS"] = str(LINEAR_SOLVER_THREADS)
    os.environ["OMP_NUM_THREADS"] = str(LINEAR_SOLVER_THREADS)
    os.environ["MKL_DYNAMIC"] = "FALSE"


configure_deterministic_runtime = configure_runtime


configure_runtime()
