import sys
from pathlib import Path


def pytest_configure() -> None:
    # Ensure `src/` is importable in test runs even if the project isn't installed.
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    sys.path.insert(0, str(src))
