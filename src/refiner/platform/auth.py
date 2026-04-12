from __future__ import annotations

import os
import tempfile
from pathlib import Path

API_KEY_ENV_VAR = "MACRODATA_API_KEY"


class MacrodataCredentialsError(RuntimeError):
    """Raised when Macrodata credentials are missing or rejected."""

    def __init__(self, message: str, *, missing: bool):
        super().__init__(message)
        self.missing = missing


def credentials_path() -> Path:
    """Return the local credential file path."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "macrodata" / "api_key"


def load_api_key() -> str:
    """Load the persisted API key from the local credential file."""
    path = credentials_path()
    if not path.exists():
        raise MacrodataCredentialsError(
            f"No credentials found at {path}",
            missing=True,
        )
    api_key = path.read_text(encoding="utf-8").strip()
    if not api_key:
        raise MacrodataCredentialsError(
            f"Credentials file at {path} is empty",
            missing=True,
        )
    return api_key


def current_api_key() -> str:
    """Return the current API key, preferring env var over local file."""
    env_value = os.environ.get(API_KEY_ENV_VAR)
    if env_value and env_value.strip():
        return env_value.strip()
    return load_api_key()


def save_api_key(api_key: str) -> Path:
    """Persist the API key with strict local file permissions."""
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(prefix=".api_key.", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(api_key)
            handle.write("\n")
        try:
            os.chmod(tmp_path, 0o600)
        except OSError:
            pass
        os.replace(tmp_path, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return path


def clear_api_key() -> bool:
    """Remove the persisted API key if it exists."""
    path = credentials_path()
    if not path.exists():
        return False
    path.unlink()
    return True
