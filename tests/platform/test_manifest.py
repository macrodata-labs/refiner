from __future__ import annotations

import sys
from contextlib import nullcontext
from email.message import Message
from importlib import metadata as importlib_metadata
from pathlib import Path
from urllib import error as urllib_error

import pytest

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
    assert manifest["environment"]["refiner_extras"] == []
    assert isinstance(manifest["dependencies"], list)


def test_build_run_manifest_can_skip_dependency_capture(
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

    manifest = build_run_manifest(capture_dependencies=False)

    assert manifest["version"] == 1
    assert "dependencies" not in manifest


def test_build_run_manifest_merges_dependencies(monkeypatch, tmp_path: Path) -> None:
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
    monkeypatch.setattr(
        "refiner.platform.manifest._collect_dependencies",
        lambda: [
            {"name": "cloudpickle", "version": "3.1.2"},
            {"name": "ego-vision", "version": "0.1.1"},
            {"name": "Requests", "version": "2.32.5"},
            {"name": "torch", "version": "2.4.0"},
        ],
    )

    manifest = build_run_manifest(
        dependencies=[
            "torch==2.6.0",
            "ego-vision[models]==0.1.2",
            "new-package",
            "transformers>=4.55",
        ],
    )

    assert manifest["dependencies"] == [
        {"name": "cloudpickle", "version": "3.1.2"},
        {"name": "ego-vision[models]", "version": "0.1.2"},
        {"name": "Requests", "version": "2.32.5"},
        {"name": "torch", "version": "2.6.0"},
        {"name": "new-package"},
        {"name": "transformers>=4.55"},
    ]


def test_build_run_manifest_adds_dependencies_without_capture(
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

    manifest = build_run_manifest(
        capture_dependencies=False,
        dependencies=["torch==2.6.0"],
    )

    assert manifest["dependencies"] == [{"name": "torch", "version": "2.6.0"}]


def test_build_run_manifest_rejects_invalid_dependency(
    monkeypatch, tmp_path: Path
) -> None:
    script_path = tmp_path / "demo_job.py"
    script_path.write_text("print('hello')\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [str(script_path)])

    with pytest.raises(ValueError, match="dependencies contains invalid requirement"):
        build_run_manifest(dependencies=["not a requirement"])


def test_build_run_manifest_records_refiner_extras(monkeypatch, tmp_path: Path) -> None:
    script_path = tmp_path / "demo_job.py"
    script_path.write_text("print('hello')\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [str(script_path)])

    manifest = build_run_manifest(
        capture_dependencies=False,
        refiner_extras=["video", "hf", "hf"],
    )

    assert manifest["environment"]["refiner_extras"] == ["hf", "video"]
    assert "dependencies" not in manifest


def test_build_run_manifest_normalizes_refiner_extra_names(
    monkeypatch, tmp_path: Path
) -> None:
    script_path = tmp_path / "demo_job.py"
    script_path.write_text("print('hello')\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [str(script_path)])

    manifest = build_run_manifest(
        capture_dependencies=False,
        refiner_extras=["HAND_tracking", "hand-tracking", "hf"],
    )

    assert manifest["environment"]["refiner_extras"] == ["hand-tracking", "hf"]
    assert "dependencies" not in manifest


def test_build_run_manifest_validates_refiner_extras_from_package_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    script_path = tmp_path / "demo_job.py"
    script_path.write_text("print('hello')\n", encoding="utf-8")
    metadata = Message()
    metadata["Provides-Extra"] = "hf"
    metadata["Provides-Extra"] = "hand_tracking"

    monkeypatch.setattr(sys, "argv", [str(script_path)])
    monkeypatch.setattr(
        "refiner.platform.manifest.importlib_metadata.metadata",
        lambda name: metadata,
    )

    manifest = build_run_manifest(
        capture_dependencies=False,
        refiner_extras=["HAND-tracking"],
    )

    assert manifest["environment"]["refiner_extras"] == ["hand-tracking"]
    with pytest.raises(ValueError, match="unknown extra 'hand-trakcing'"):
        build_run_manifest(capture_dependencies=False, refiner_extras=["hand_trakcing"])


def test_build_run_manifest_rejects_string_refiner_extras(
    monkeypatch, tmp_path: Path
) -> None:
    script_path = tmp_path / "demo_job.py"
    script_path.write_text("print('hello')\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [str(script_path)])

    with pytest.raises(ValueError, match="refiner_extras must be a sequence"):
        build_run_manifest(capture_dependencies=False, refiner_extras="hf")


def test_build_run_manifest_skips_refiner_extra_validation_without_metadata_extras(
    monkeypatch, tmp_path: Path
) -> None:
    script_path = tmp_path / "demo_job.py"
    script_path.write_text("print('hello')\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [str(script_path)])
    monkeypatch.setattr(
        "refiner.platform.manifest.importlib_metadata.metadata",
        lambda _name: Message(),
    )

    manifest = build_run_manifest(capture_dependencies=False, refiner_extras=["bad"])

    assert manifest["environment"]["refiner_extras"] == ["bad"]


def test_build_run_manifest_rejects_unknown_refiner_extra(
    monkeypatch, tmp_path: Path
) -> None:
    script_path = tmp_path / "demo_job.py"
    script_path.write_text("print('hello')\n", encoding="utf-8")
    metadata = Message()
    metadata["Provides-Extra"] = "hf"
    metadata["Provides-Extra"] = "video"

    monkeypatch.setattr(sys, "argv", [str(script_path)])
    monkeypatch.setattr(
        "refiner.platform.manifest.importlib_metadata.metadata",
        lambda _name: metadata,
    )

    with pytest.raises(ValueError, match="unknown extra 'not-real'"):
        build_run_manifest(capture_dependencies=False, refiner_extras=["not-real"])


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


def test_build_run_manifest_environment_does_not_include_rundir_by_default(
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

    assert "rundir" not in manifest["environment"]


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


def test_manifest_prefers_macrodata_refiner_distribution(monkeypatch) -> None:
    def _version(name: str) -> str:
        if name == "macrodata-refiner":
            return "0.2.0"
        if name == "refiner":  # pragma: no cover - should not be queried
            raise AssertionError("legacy refiner distribution should not be used")
        raise importlib_metadata.PackageNotFoundError(name)

    class _Dist:
        version = "0.2.0"

        @staticmethod
        def read_text(name: str) -> str | None:
            if name == "direct_url.json":
                return '{"vcs_info":{"commit_id":"abc123def456"}}'
            return None

    def _distribution(name: str):
        if name == "macrodata-refiner":
            return _Dist()
        if name == "refiner":  # pragma: no cover - should not be queried
            raise AssertionError("legacy refiner distribution should not be used")
        raise importlib_metadata.PackageNotFoundError(name)

    monkeypatch.setattr(
        "refiner.platform.manifest.importlib_metadata.version", _version
    )
    monkeypatch.setattr(
        "refiner.platform.manifest.importlib_metadata.distribution",
        _distribution,
    )
    monkeypatch.setattr(
        "refiner.platform.manifest._resolve_local_repo_git_sha",
        lambda: None,
    )
    monkeypatch.setattr(sys, "argv", ["-c"])

    manifest = build_run_manifest()

    assert manifest["environment"]["refiner_version"] == "0.2.0"
    assert manifest["environment"]["refiner_ref"] == "abc123def456"
