"""Open-source reimplementation of the paper's rib-layout workflow."""

from rib_layout_env import configure_deterministic_runtime

# Apply deterministic thread settings before importing NumPy/SciPy-backed
# model modules through the public package interface.
configure_deterministic_runtime()

from rib_layout_core import load_case
from rib_layout_algorithms.model import Rib, StiffenedPlateModel
from rib_layout_algorithms.model_shell import ShellStiffenedPlateModel
from rib_layout_algorithms.optimization import RibLayoutOptimizer

__all__ = ["Rib", "StiffenedPlateModel", "ShellStiffenedPlateModel", "RibLayoutOptimizer", "load_case"]
__version__ = "0.1.0"
