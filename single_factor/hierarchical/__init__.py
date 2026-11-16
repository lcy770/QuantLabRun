"""Archive-scheduled PPO and reversible AST diffusion factor search."""
from pathlib import Path
import sys

# Both `python -m single_factor.hierarchical` and the backend worker resolve the
# adapters belonging to THIS checkout, including an uninstalled development tree.
_backend = str(Path(__file__).resolve().parents[2] / "backend")
if _backend not in sys.path:
    sys.path.insert(0, _backend)
