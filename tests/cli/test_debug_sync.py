from __future__ import annotations

import io
import tarfile

from refiner.cli.debug_sync import build_source_archive


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
