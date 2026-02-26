"""Macrodata platform integration utilities."""

from .auth import (
    CredentialsError,
    clear_api_key,
    credentials_path,
    current_api_key,
    load_api_key,
    save_api_key,
)
from .config import (
    resolve_platform_base_url,
)
from .client import MacrodataClient
from .http import MacrodataApiError, verify_api_key

__all__ = [
    "CredentialsError",
    "credentials_path",
    "load_api_key",
    "current_api_key",
    "save_api_key",
    "clear_api_key",
    "resolve_platform_base_url",
    "MacrodataClient",
    "MacrodataApiError",
    "verify_api_key",
]
