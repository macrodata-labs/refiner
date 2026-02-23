from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from refiner.cli import auth


def _ok_payload() -> dict[str, object]:
    return {
        "object": "api_key",
        "id": "k1",
        "name": "Ingestion Backend Key",
        "created": 1,
        "enabled": True,
        "user": {"name": "Jane Doe", "username": "jane", "email": "jane@example.com"},
    }


def test_login_with_token_success(monkeypatch, capsys) -> None:
    monkeypatch.setattr(auth, "verify_api_key", lambda **_: _ok_payload())
    monkeypatch.setattr(auth, "save_api_key", lambda token: f"/tmp/{token}")
    monkeypatch.setattr(
        auth, "resolve_platform_base_url", lambda: "https://app.example.com"
    )

    rc = auth.cmd_login(Namespace(token="ing_abc", token_stdin=False, quiet=True))
    out = capsys.readouterr()

    assert rc == 0
    assert "Logged in as jane (jane@example.com)" in out.out
    assert "API key name: Ingestion Backend Key" in out.out


def test_login_invalid_token(monkeypatch, capsys) -> None:
    def _raise(**_: object) -> dict[str, object]:
        raise auth.MacrodataApiError(status=401, message="Invalid API key")

    monkeypatch.setattr(auth, "verify_api_key", _raise)
    monkeypatch.setattr(
        auth, "resolve_platform_base_url", lambda: "https://app.example.com"
    )

    rc = auth.cmd_login(Namespace(token="ing_bad", token_stdin=False, quiet=True))
    out = capsys.readouterr()

    assert rc == 1
    assert "Invalid API key." in out.err
    assert "/settings/tokens" in out.err


def test_login_warns_before_overwrite_before_prompt(
    monkeypatch, capsys, tmp_path
) -> None:
    existing = tmp_path / "api_key"
    existing.write_text("ing_old\n")

    monkeypatch.setattr(auth, "credentials_path", lambda: existing)
    monkeypatch.setattr(auth.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(auth.getpass, "getpass", lambda prompt: "replacement_key")
    monkeypatch.setattr(auth, "verify_api_key", lambda **_: _ok_payload())
    monkeypatch.setattr(auth, "save_api_key", lambda token: Path(f"/tmp/{token}"))
    monkeypatch.setattr(
        auth, "resolve_platform_base_url", lambda: "https://app.example.com"
    )

    rc = auth.cmd_login(Namespace(token=None, token_stdin=False, quiet=False))
    out = capsys.readouterr()

    assert rc == 0
    assert "already stored" in out.out
    assert "macrodata whoami" in out.out
    assert "overwrite" in out.out


def test_whoami_success(monkeypatch, capsys) -> None:
    monkeypatch.setattr(auth, "current_api_key", lambda: "ing_abc")
    monkeypatch.setattr(auth, "verify_api_key", lambda **_: _ok_payload())
    monkeypatch.setattr(
        auth, "resolve_platform_base_url", lambda: "https://app.example.com"
    )

    rc = auth.cmd_whoami(Namespace())
    out = capsys.readouterr()

    assert rc == 0
    assert "Logged in as jane (jane@example.com)" in out.out
    assert "API key name: Ingestion Backend Key" in out.out


def test_logout_no_credentials(monkeypatch, capsys) -> None:
    monkeypatch.setattr(auth, "clear_api_key", lambda: False)
    rc = auth.cmd_logout(Namespace())
    out = capsys.readouterr()

    assert rc == 0
    assert "No local credentials found." in out.out
