from __future__ import annotations

import sys
from pathlib import Path

from refiner.platform.manifest import build_run_manifest


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
        "refiner.platform.manifest._resolve_refiner_ref",
        lambda: "abc123def456",
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
        "refiner.platform.manifest._resolve_refiner_ref",
        lambda: "abc123def456",
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
        "refiner.platform.manifest._resolve_refiner_ref",
        lambda: None,
    )

    manifest = build_run_manifest()

    assert "macrodata_cloud" not in manifest
    assert manifest["environment"]["refiner_version"] == "0.2.0"
    assert manifest["environment"]["refiner_ref"] is None
