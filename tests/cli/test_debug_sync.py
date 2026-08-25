from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from types import ModuleType

import cloudpickle
import pytest

from refiner.cli.debug_sync import (
    build_debug_sync_bundle,
    build_source_archive,
    find_project_root,
    pickle_project_modules_by_value,
)


def test_source_archive_excludes_secrets_and_build_artifacts(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "pipeline.py").write_text("PIPELINE = 1\n")
    (tmp_path / ".env").write_text("SECRET=nope\n")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "installed.py").write_text("NOPE = 1\n")

    archive = build_source_archive(tmp_path)

    with tarfile.open(fileobj=io.BytesIO(archive.payload), mode="r:gz") as tar:
        names = tar.getnames()
    assert names == ["src/pipeline.py"]
    assert archive.file_count == 1
    assert len(archive.sha256) == 64


def test_source_archive_rejects_projects_with_no_included_files(tmp_path) -> None:
    (tmp_path / ".env").write_text("SECRET=nope\n")

    with pytest.raises(ValueError, match="source archive is empty"):
        build_source_archive(tmp_path)


def test_source_archive_excludes_conventional_virtualenv(tmp_path) -> None:
    (tmp_path / "pipeline.py").write_text("PIPELINE = 1\n")
    (tmp_path / "venv" / "lib").mkdir(parents=True)
    (tmp_path / "venv" / "lib" / "dependency.py").write_text("SECRET = 1\n")

    archive = build_source_archive(tmp_path)

    with tarfile.open(fileobj=io.BytesIO(archive.payload), mode="r:gz") as tar:
        assert tar.getnames() == ["pipeline.py"]


def test_debug_sync_bundle_contains_one_complete_generation(tmp_path) -> None:
    (tmp_path / "pipeline.py").write_text("PIPELINE = 1\n")

    pipeline_payload = b"cloudpickle"
    pipeline_sha256 = hashlib.sha256(pipeline_payload).hexdigest()
    bundle = build_debug_sync_bundle(
        source_root=tmp_path,
        pipeline_payload=pipeline_payload,
        pipeline_sha256=pipeline_sha256,
        allocation_fingerprint="a" * 64,
    )

    with tarfile.open(fileobj=io.BytesIO(bundle.payload), mode="r:") as archive:
        assert archive.getnames() == [
            "sync.json",
            "source.tar.gz",
            "pipeline.cloudpickle",
        ]
        metadata_file = archive.extractfile("sync.json")
        assert metadata_file is not None
        metadata = json.loads(metadata_file.read())
    assert metadata["allocation_fingerprint"] == "a" * 64
    assert metadata["pipeline_sha256"] == pipeline_sha256
    assert metadata["source_sha256"] == bundle.source_sha256
    assert bundle.file_count == 1
    assert len(bundle.sha256) == 64


def test_project_root_prefers_nearest_project_marker(tmp_path) -> None:
    project = tmp_path / "project"
    nested = project / "pipelines"
    nested.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='example'\n")
    script = nested / "pipeline.py"
    script.write_text("pass\n")

    assert find_project_root(script) == project


def test_project_modules_are_pickled_without_remote_source_imports(tmp_path) -> None:
    module_path = tmp_path / "debug_helper.py"
    module_path.write_text("def transform(value):\n    return value + 1\n")
    spec = importlib.util.spec_from_file_location("debug_helper", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module.__name__] = module
    try:
        spec.loader.exec_module(module)
        with pickle_project_modules_by_value(tmp_path):
            payload = cloudpickle.dumps(module.transform)
    finally:
        sys.modules.pop(module.__name__, None)

    module_path.unlink()
    restored = cloudpickle.loads(payload)
    assert restored(2) == 3
    assert module.__name__ not in cloudpickle.list_registry_pickle_by_value()


def test_project_virtualenv_modules_are_not_registered_by_value(tmp_path) -> None:
    virtualenv_module = ModuleType("__main__")
    virtualenv_module.__file__ = str(tmp_path / ".venv" / "bin" / "macrodata")
    sys.modules["debug_virtualenv_main"] = virtualenv_module
    try:
        with pickle_project_modules_by_value(tmp_path):
            pass
    finally:
        sys.modules.pop("debug_virtualenv_main", None)
