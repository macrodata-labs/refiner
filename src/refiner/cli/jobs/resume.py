from __future__ import annotations

import json
from argparse import Namespace
from typing import Any

import msgspec

from refiner.cli.jobs.common import _client, _handle_error
from refiner.cli.jobs.follow import safe_text as _safe_text
from refiner.platform.auth import MacrodataCredentialsError
from refiner.platform.client import (
    CloudResumeSelector,
    CloudRunResumeRequest,
    CloudRuntimeOverrides,
    MacrodataApiError,
)


def _print_resume_json(payload: Any) -> int:
    json_ready = payload
    if (
        not isinstance(payload, (dict, list, str, int, float, bool))
        and payload is not None
    ):
        json_ready = msgspec.to_builtins(payload)
    print(json.dumps(json_ready, indent=2, sort_keys=True))
    return 0


def _render_resume(payload: Any) -> int:
    print(
        "Resumed:"
        f" {_safe_text(getattr(payload, 'job_id', None))}"
        f"  Status: {_safe_text(getattr(payload, 'status', None))}"
        f"  Stage: {_safe_text(getattr(payload, 'stage_index', None))}"
    )
    return 0


def _resume_selector_from_args(args: Namespace) -> CloudResumeSelector:
    selector = CloudResumeSelector.from_mode(
        job_id=args.job_id,
        mode="latest-compatible" if args.latest_compatible else None,
        name=args.name,
        limit_to_me=args.limit_to_me,
    )
    if selector is None:
        raise ValueError(
            "resume selector must specify exactly one of job_id or latest_compatible"
        )
    return selector


def _resume_runtime_overrides_from_args(
    args: Namespace,
) -> CloudRuntimeOverrides | None:
    if (
        args.num_workers is None
        and args.cpus_per_worker is None
        and args.mem_mb_per_worker is None
        and args.gpus_per_worker is None
        and args.gpu_type is None
    ):
        return None
    return CloudRuntimeOverrides(
        num_workers=args.num_workers,
        cpus_per_worker=args.cpus_per_worker,
        mem_mb_per_worker=args.mem_mb_per_worker,
        gpus_per_worker=args.gpus_per_worker,
        gpu_type=args.gpu_type,
    )


def cmd_jobs_resume(args: Namespace) -> int:
    try:
        selector = _resume_selector_from_args(args)
        runtime_overrides = _resume_runtime_overrides_from_args(args)
        payload = _client().cloud_resume_job(
            request=CloudRunResumeRequest(
                selector=selector,
                runtime_overrides=runtime_overrides,
            )
        )
    except (MacrodataApiError, MacrodataCredentialsError, ValueError) as err:
        return _handle_error(err)
    return _print_resume_json(payload) if args.json else _render_resume(payload)
