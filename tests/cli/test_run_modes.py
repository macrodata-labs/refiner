from __future__ import annotations

import io

from refiner.cli.run.modes import CloudAttachContext, emit_cloud_followup_commands


def test_cloud_followup_commands_include_stage_metrics_when_stage_known() -> None:
    output = io.StringIO()

    emit_cloud_followup_commands(
        context=CloudAttachContext(
            job_id="job-1",
            job_name="cloud pipeline",
            tracking_url="https://example.test/jobs/job-1",
            stage_index=2,
        ),
        file=output,
    )

    assert "Metrics: macrodata jobs metrics job-1 2\n" in output.getvalue()


def test_cloud_followup_commands_skip_metrics_when_stage_unknown() -> None:
    output = io.StringIO()

    emit_cloud_followup_commands(
        context=CloudAttachContext(
            job_id="job-1",
            job_name="cloud pipeline",
            tracking_url="https://example.test/jobs/job-1",
            stage_index=None,
        ),
        file=output,
    )

    assert "Metrics:" not in output.getvalue()
