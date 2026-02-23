from __future__ import annotations

from refiner.platform.auth import (
    clear_api_key,
    credentials_path,
    current_api_key,
    load_api_key,
    save_api_key,
)


def test_save_and_load_api_key(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    path = save_api_key("ing_test123")
    assert path == credentials_path()
    assert load_api_key() == "ing_test123"
    assert path.read_text().strip() == "ing_test123"


def test_clear_api_key(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    save_api_key("ing_test123")
    assert clear_api_key() is True
    assert clear_api_key() is False


def test_current_api_key_prefers_env_over_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    save_api_key("ing_file")
    monkeypatch.setenv("MACRODATA_API_KEY", "ing_env")
    assert current_api_key() == "ing_env"


def test_current_api_key_falls_back_to_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    save_api_key("ing_file")
    monkeypatch.delenv("MACRODATA_API_KEY", raising=False)
    assert current_api_key() == "ing_file"
