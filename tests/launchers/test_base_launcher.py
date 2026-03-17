from __future__ import annotations

from typing import cast

from refiner.platform.client import MacrodataClient
from refiner.pipeline import RefinerPipeline
from refiner.launchers.base import BaseLauncher
from refiner.pipeline.planning import PlannedStage, StageComputeRequirements


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


def test_run_manifest_includes_stage_worker_counts(monkeypatch) -> None:
    launcher = _DummyLauncher(
        pipeline=cast(RefinerPipeline, object()), name="unit-test", num_workers=2
    )
    monkeypatch.setattr(
        "refiner.launchers.base.build_run_manifest",
        lambda: {},
    )

    manifest = launcher._run_manifest(
        [
            PlannedStage(
                index=0,
                name="stage-0",
                pipeline=cast(RefinerPipeline, object()),
                compute=StageComputeRequirements(num_workers=2),
            ),
            PlannedStage(
                index=1,
                name="stage-1",
                pipeline=cast(RefinerPipeline, object()),
                compute=StageComputeRequirements(num_workers=1),
            ),
        ]
    )

    assert manifest == {
        "macrodata_cloud": {"stage_runtimes": [{"num_workers": 2}, {"num_workers": 1}]}
    }
