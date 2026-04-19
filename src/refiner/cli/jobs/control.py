from __future__ import annotations

from argparse import Namespace

from refiner.cli.job_utils import safe_text as _safe_text
from refiner.cli.jobs.common import _client, _handle_error, _print_json
from refiner.platform.auth import MacrodataCredentialsError
from refiner.platform.client import MacrodataApiError


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
    try:
        payload = _client().cli_cancel_job(job_id=args.job_id)
    except (MacrodataApiError, MacrodataCredentialsError) as err:
        return _handle_error(err)
    return _print_json(payload) if args.json else _render_cancel(payload)
