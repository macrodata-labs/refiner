from __future__ import annotations

from argparse import Namespace

from refiner.cli.jobs.follow import safe_text as _safe_text
from refiner.cli.jobs.common import _client, _run_job_command


def _render_cancel(payload: dict[str, object]) -> int:
    job_id = payload.get("jobId", payload.get("job_id"))
    requested = payload.get("requestedOperations", payload.get("requested_operations"))
    canceled = payload.get("canceledOperations", payload.get("canceled_operations"))
    failed = payload.get("failedOperations", payload.get("failed_operations"))
    print(
        "Canceled:"
        f" {_safe_text(job_id)}"
        f"  Requested: {_safe_text(requested)}"
        f"  Canceled: {_safe_text(canceled)}"
        f"  Failed: {_safe_text(failed)}"
    )
    return 0


def cmd_jobs_cancel(args: Namespace) -> int:
    return _run_job_command(
        as_json=args.json,
        fetch=lambda: _client().cli_cancel_job(job_id=args.job_id),
        renderer=_render_cancel,
    )
