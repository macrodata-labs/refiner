from __future__ import annotations

from typing import cast

from refiner.platform.client import MacrodataClient
from refiner.pipeline import RefinerPipeline
from refiner.runtime.launchers.base import BaseLauncher


class _DummyLauncher(BaseLauncher):
    def launch(self):  # pragma: no cover - not used by this unit test
        return None


def test_job_tracking_url_sanitizes_terminal_control_characters() -> None:
    launcher = _DummyLauncher(
        pipeline=cast(RefinerPipeline, object()), name="unit-test"
    )
    client = MacrodataClient(api_key="md_test", base_url="https://app.example.com")

    url = launcher._job_tracking_url(
        client=client,
        workspace_slug="macro\x07data",
        job_id="job-\x1b[31m",
    )

    assert url == "https://app.example.com/jobs/macrodata/job-[31m"
    assert "\x07" not in url
    assert "\x1b" not in url
