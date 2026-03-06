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

    manifest = build_run_manifest()

    assert manifest["version"] == 1
    assert manifest["script"]["path"] == str(script_path.resolve())
    assert manifest["script"]["text"] == "print('hello')\n"
    assert isinstance(manifest["script"]["sha256"], str)
    assert manifest["environment"]["python_version"]
    assert isinstance(manifest["dependencies"], list)
