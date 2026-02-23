"""Macrodata platform integration utilities."""

from .auth import (
    API_KEY_ENV_VAR,
    CredentialsError,
    clear_api_key,
    credentials_path,
    current_api_key,
    load_api_key,
    save_api_key,
)
from .config import (
    PLATFORM_BASE_URL_ENV_VAR,
    resolve_platform_base_url,
)
from .http import MacrodataApiError, verify_api_key

__all__ = [
    "API_KEY_ENV_VAR",
    "CredentialsError",
    "credentials_path",
    "load_api_key",
    "current_api_key",
    "save_api_key",
    "clear_api_key",
    "PLATFORM_BASE_URL_ENV_VAR",
    "resolve_platform_base_url",
    "MacrodataApiError",
    "verify_api_key",
]
