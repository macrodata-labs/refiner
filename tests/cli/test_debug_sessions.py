from __future__ import annotations

from types import SimpleNamespace

from refiner.cli.debug_sessions import (
    find_session,
    new_session_record,
    remove_session,
    save_session,
)


class _Client:
    def __init__(self, workspace: str) -> None:
        self.base_url = "https://macrodata.test"
        self.workspace = workspace

    def verify_api_key(self):
        return SimpleNamespace(
            workspace=SimpleNamespace(slug=self.workspace),
            key_id="key-1",
            name="test",
        )


def test_session_registry_is_scoped_by_pipeline_endpoint_and_workspace(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    script = tmp_path / "project" / "pipeline.py"
    script.parent.mkdir()
    script.write_text("pass\n")
    client = _Client("workspace-a")
    record = new_session_record(
        script=script,
        project_root=script.parent,
        script_args=["--rows", "10"],
        job_id="job-1",
        client=client,  # type: ignore[arg-type]
    )

    save_session(record)

    assert find_session(script=script, client=client) == record  # type: ignore[arg-type]
    assert find_session(script=script, client=_Client("workspace-b")) is None  # type: ignore[arg-type]
    remove_session(script=script, client=client)  # type: ignore[arg-type]
    assert find_session(script=script, client=client) is None  # type: ignore[arg-type]
