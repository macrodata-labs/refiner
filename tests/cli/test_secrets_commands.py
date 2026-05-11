from __future__ import annotations

from argparse import Namespace

from refiner.cli import secrets


class _FakeClient:
    def __init__(self) -> None:
        self.set_calls: list[dict[str, str]] = []
        self.delete_calls: list[dict[str, str]] = []

    def cli_list_secrets(self, *, env: str | None = None) -> dict[str, object]:
        return {
            "secrets": [
                {
                    "env": env or "default",
                    "name": "HF_TOKEN",
                    "updatedAt": "2026-05-11T10:00:00.000Z",
                }
            ]
        }

    def cli_set_secret(self, *, name: str, env: str, value: str) -> dict[str, object]:
        self.set_calls.append({"name": name, "env": env, "value": value})
        return {"secret": {"env": env, "name": name}}

    def cli_delete_secret(self, *, name: str, env: str) -> dict[str, object]:
        self.delete_calls.append({"name": name, "env": env})
        return {"success": True}


def test_secrets_list_prints_names(monkeypatch, capsys) -> None:
    monkeypatch.setattr(secrets, "create_client", lambda: _FakeClient())

    rc = secrets.cmd_secrets_list(Namespace(env="production", json=False))
    out = capsys.readouterr()

    assert rc == 0
    assert "production" in out.out
    assert "HF_TOKEN" in out.out


def test_secrets_set_reads_value_from_stdin(monkeypatch, capsys) -> None:
    client = _FakeClient()
    monkeypatch.setattr(secrets, "create_client", lambda: client)
    monkeypatch.setattr(secrets.sys.stdin, "read", lambda _size=-1: "hf_secret\r\n")

    rc = secrets.cmd_secrets_set(
        Namespace(name="HF_TOKEN", env="production", value_stdin=True, json=False)
    )
    out = capsys.readouterr()

    assert rc == 0
    assert client.set_calls == [
        {"name": "HF_TOKEN", "env": "production", "value": "hf_secret"}
    ]
    assert "hf_secret" not in out.out
    assert "production/HF_TOKEN" in out.out


def test_secrets_set_allows_max_size_value_with_trailing_newline(
    monkeypatch, capsys
) -> None:
    client = _FakeClient()
    monkeypatch.setattr(secrets, "create_client", lambda: client)
    monkeypatch.setattr(
        secrets.sys.stdin,
        "read",
        lambda _size=-1: ("x" * secrets.SECRET_VALUE_MAX_CHARS) + "\n",
    )

    rc = secrets.cmd_secrets_set(
        Namespace(name="HF_TOKEN", env="production", value_stdin=True, json=False)
    )
    out = capsys.readouterr()

    assert rc == 0
    assert client.set_calls == [
        {
            "name": "HF_TOKEN",
            "env": "production",
            "value": "x" * secrets.SECRET_VALUE_MAX_CHARS,
        }
    ]
    assert "too large" not in out.err


def test_secrets_set_rejects_large_stdin(monkeypatch, capsys) -> None:
    client = _FakeClient()
    monkeypatch.setattr(secrets, "create_client", lambda: client)
    monkeypatch.setattr(
        secrets.sys.stdin,
        "read",
        lambda _size=-1: "x" * (secrets.SECRET_VALUE_MAX_CHARS + 1),
    )

    rc = secrets.cmd_secrets_set(
        Namespace(name="HF_TOKEN", env="production", value_stdin=True, json=False)
    )
    out = capsys.readouterr()

    assert rc == 1
    assert client.set_calls == []
    assert "too large" in out.err


def test_secrets_remove_calls_delete(monkeypatch, capsys) -> None:
    client = _FakeClient()
    monkeypatch.setattr(secrets, "create_client", lambda: client)

    rc = secrets.cmd_secrets_remove(
        Namespace(name="HF_TOKEN", env="production", json=False)
    )
    out = capsys.readouterr()

    assert rc == 0
    assert client.delete_calls == [{"name": "HF_TOKEN", "env": "production"}]
    assert "production/HF_TOKEN" in out.out
