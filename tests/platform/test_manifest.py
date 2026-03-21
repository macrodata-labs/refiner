from __future__ import annotations

import sys
from contextlib import nullcontext
from email.message import Message
from pathlib import Path
from urllib import error as urllib_error

from refiner.platform.manifest import build_run_manifest, refiner_ref_exists_on_remote


def test_build_run_manifest_captures_script_from_argv(
    monkeypatch, tmp_path: Path
) -> None:
    script_path = tmp_path / "demo_job.py"
    script_path.write_text("print('hello')\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [str(script_path)])
    monkeypatch.setattr(
        "refiner.platform.manifest._resolve_installed_version",
        lambda: "0.2.0",
    )
    monkeypatch.setattr(
        "refiner.platform.manifest._resolve_direct_url_git_sha",
        lambda: "abc123def456",
    )
    monkeypatch.setattr(
        "refiner.platform.manifest._resolve_local_repo_git_sha",
        lambda: None,
    )

    manifest = build_run_manifest()

    assert manifest["version"] == 1
    assert manifest["script"]["path"] == str(script_path.resolve())
    assert manifest["script"]["text"] == "print('hello')\n"
    assert isinstance(manifest["script"]["sha256"], str)
    assert manifest["environment"]["python_version"]
    assert manifest["environment"]["refiner_version"] == "0.2.0"
    assert manifest["environment"]["refiner_ref"] == "abc123def456"
    assert isinstance(manifest["dependencies"], list)


def test_build_run_manifest_redacts_secret_values(monkeypatch, tmp_path: Path) -> None:
    script_path = tmp_path / "demo_job.py"
    script_path.write_text(
        "API_KEY = 'super-secret-value'\nprint(API_KEY)\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", [str(script_path)])
    monkeypatch.setattr(
        "refiner.platform.manifest._resolve_installed_version",
        lambda: "0.2.0",
    )
    monkeypatch.setattr(
        "refiner.platform.manifest._resolve_direct_url_git_sha",
        lambda: "abc123def456",
    )
    monkeypatch.setattr(
        "refiner.platform.manifest._resolve_local_repo_git_sha",
        lambda: None,
    )

    manifest = build_run_manifest()

    assert manifest["script"]["path"] == str(script_path.resolve())
    assert "super-secret-value" in manifest["script"]["text"]


def test_build_run_manifest_omits_stage_runtimes_by_default(
    monkeypatch, tmp_path: Path
) -> None:
    script_path = tmp_path / "demo_job.py"
    script_path.write_text("print('hello')\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [str(script_path)])
    monkeypatch.setattr(
        "refiner.platform.manifest._resolve_installed_version",
        lambda: "0.2.0",
    )
    monkeypatch.setattr(
        "refiner.platform.manifest._resolve_direct_url_git_sha",
        lambda: None,
    )
    monkeypatch.setattr(
        "refiner.platform.manifest._resolve_local_repo_git_sha",
        lambda: None,
    )

    manifest = build_run_manifest()

    assert "macrodata_cloud" not in manifest
    assert manifest["environment"]["refiner_version"] == "0.2.0"
    assert manifest["environment"]["refiner_ref"] is None


def test_refiner_ref_exists_on_remote_returns_true_on_success(monkeypatch) -> None:
    monkeypatch.setattr(
        "refiner.platform.manifest.urllib_request.urlopen",
        lambda request: nullcontext(object()),
    )

    assert refiner_ref_exists_on_remote("abc123") is True


def test_refiner_ref_exists_on_remote_returns_false_on_404(monkeypatch) -> None:
    def _raise_404(request):
        raise urllib_error.HTTPError(
            request.full_url,
            404,
            "Not Found",
            hdrs=Message(),
            fp=None,
        )

    monkeypatch.setattr(
        "refiner.platform.manifest.urllib_request.urlopen",
        _raise_404,
    )

    assert refiner_ref_exists_on_remote("abc123") is False
