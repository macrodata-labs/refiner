from __future__ import annotations

from argparse import Namespace
import sys

from refiner.cli.jobs.common import _client, _executor_text, _handle_error
from refiner.platform.auth import MacrodataCredentialsError
from refiner.platform.client import MacrodataApiError


def cmd_jobs_attach(args: Namespace) -> int:
    from refiner.cli import cloud_run

    try:
        client = _client()
        payload = client.cli_get_job(job_id=args.job_id)
    except (MacrodataApiError, MacrodataCredentialsError) as err:
        return _handle_error(err)

    job = payload.get("job")
    if not isinstance(job, dict):
        print("Job details unavailable.", file=sys.stderr)
        return 1
    if _executor_text(job.get("executorKind")) != "cloud":
        print(
            "`macrodata jobs attach` is only supported for cloud jobs.", file=sys.stderr
        )
        return 1

    try:
        return cloud_run.attach_to_cloud_job(
            client=client,
            job_id=args.job_id,
            initial_job_payload=payload,
            force_attach=False,
        )
    except cloud_run.CloudAttachDetached:
        return 130
    except SystemExit as err:
        code = err.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        print(str(code), file=sys.stderr)
        return 1
    except (MacrodataApiError, MacrodataCredentialsError) as err:
        return _handle_error(err)
