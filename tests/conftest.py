import os
import sys
import tempfile
from pathlib import Path


def pytest_configure() -> None:
    # Ensure `src/` is importable in test runs even if the project isn't installed.
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    sys.path.insert(0, str(src))
    # Keep subprocess workers aligned with the same import path.
    existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else str(src)
    # Tests should not report local launcher runs to a real Macrodata account just
    # because the developer shell has credentials configured.
    os.environ.pop("MACRODATA_API_KEY", None)
    os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="refiner-test-config-")
