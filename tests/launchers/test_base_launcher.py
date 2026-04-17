from __future__ import annotations

from typing import cast

from refiner.platform.client import MacrodataClient
from refiner.pipeline import RefinerPipeline
from refiner.launchers.base import BaseLauncher
from refiner.pipeline.planning import (
    PlannedStage,
    StageComputeRequirements,
)


class _DummyLauncher(BaseLauncher):
    def launch(self):  # pragma: no cover - not used by this unit test
        return None


class _ResourceHintLauncher(_DummyLauncher):
    def _stage_compute_requirements(
        self, compute: StageComputeRequirements
    ) -> StageComputeRequirements:
        return StageComputeRequirements(
            num_workers=compute.num_workers,
            cpus_per_worker=1,
            gpus_per_worker=2,
        )


def test_job_tracking_url_sanitizes_terminal_control_characters() -> None:
    launcher = _DummyLauncher(
        pipeline=cast(RefinerPipeline, object()), name="unit-test"
    )
    client = MacrodataClient(api_key="md_test", base_url="https://app.\x9bexample.com")

    url = launcher._job_tracking_url(
        client=client,
        workspace_slug="macro\x07data",
        job_id="job-\x1b[31m",
    )

    assert url == "https://app.example.com/jobs/macrodata/job-[31m"
    assert "\x07" not in url
    assert "\x1b" not in url
    assert "\x9b" not in url


def test_compiled_plan_includes_stage_worker_counts(monkeypatch) -> None:
    launcher = _DummyLauncher(
        pipeline=cast(RefinerPipeline, object()), name="unit-test", num_workers=2
    )
    monkeypatch.setattr(
        "refiner.launchers.base.compile_planned_stages",
        lambda stages, **_: {
            "stages": [
                {
                    "name": stage.name,
                    "index": stage.index,
                    "requested_num_workers": stage.compute.num_workers,
                    "steps": [],
                }
                for stage in stages
            ]
        },
    )

    plan = launcher._compiled_plan(
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

    assert plan["stages"] == [
        {"name": "stage-0", "index": 0, "requested_num_workers": 2, "steps": []},
        {"name": "stage-1", "index": 1, "requested_num_workers": 1, "steps": []},
    ]


def test_compiled_plan_includes_stage_resource_hints(monkeypatch) -> None:
    launcher = _ResourceHintLauncher(
        pipeline=cast(RefinerPipeline, object()), name="unit-test", num_workers=2
    )
    monkeypatch.setattr(
        "refiner.launchers.base.compile_planned_stages",
        lambda stages, **_: {
            "stages": [
                {
                    "name": stage.name,
                    "index": stage.index,
                    **stage.compute.to_stage_plan_dict(),
                    "steps": [],
                }
                for stage in stages
            ]
        },
    )

    plan = launcher._compiled_plan(
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

    assert plan["stages"] == [
        {
            "name": "stage-0",
            "index": 0,
            "requested_num_workers": 2,
            "cpus_per_worker": 1,
            "gpus_per_worker": 2,
            "steps": [],
        },
        {
            "name": "stage-1",
            "index": 1,
            "requested_num_workers": 1,
            "cpus_per_worker": 1,
            "gpus_per_worker": 2,
            "steps": [],
        },
    ]
