from __future__ import annotations

import json
from argparse import Namespace
from typing import Any

import msgspec

from refiner.cli.jobs.common import _client, _handle_error
from refiner.cli.jobs.follow import safe_text as _safe_text
from refiner.platform.auth import MacrodataCredentialsError
from refiner.platform.client import (
    CloudRunResumeRequest,
    CloudRuntimeOverrides,
    MacrodataApiError,
)
from refiner.platform.client.models import CloudResumeSelector, build_resume_selector


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
    selector = build_resume_selector(
        job_id=args.job_id,
        latest_compatible=args.latest_compatible,
        name=args.name,
        limit_to_me=args.limit_to_me,
        require_selector=True,
    )
    if selector is None:  # pragma: no cover
        raise ValueError("resume selector parsing produced no selector")
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
        request = CloudRunResumeRequest(
            selector=selector,
            runtime_overrides=runtime_overrides,
        )
        client = _client()
        payload = (
            client.cloud_resume_job_raw(request=request)
            if args.json
            else client.cloud_resume_job(request=request)
        )
    except (MacrodataApiError, MacrodataCredentialsError, ValueError) as err:
        return _handle_error(err)
    return _print_resume_json(payload) if args.json else _render_resume(payload)
