from __future__ import annotations

import os

PLATFORM_BASE_URL_ENV_VAR = "MACRODATA_BASE_URL"
_PLATFORM_BASE_URL = "https://macrodata.co"


def resolve_platform_base_url() -> str:
    """Resolve platform base URL from env override or product default."""
    env_value = os.environ.get(PLATFORM_BASE_URL_ENV_VAR)
    if env_value:
        return env_value.rstrip("/")
    return _PLATFORM_BASE_URL
