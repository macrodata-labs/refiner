from __future__ import annotations

import argparse
import json
import os
import subprocess


DEFAULT_S3_CHECK = (
    "s3://macrodata-rerun-format-tests/dominique-sample/episode-5__base.rrd"
)
SECRET_NAMES = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_DEFAULT_REGION",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export short-lived AWS profile credentials into a Macrodata "
            "workspace secret environment for Rerun cloud benchmarks."
        )
    )
    parser.add_argument("--aws-profile", default="default")
    parser.add_argument("--secret-env", default="researcher")
    parser.add_argument(
        "--region",
        help=(
            "AWS region to store as AWS_DEFAULT_REGION. Defaults to the profile "
            "region, AWS_DEFAULT_REGION, AWS_REGION, or us-east-1."
        ),
    )
    parser.add_argument(
        "--s3-check",
        default=DEFAULT_S3_CHECK,
        help="S3 URI to verify with the selected AWS profile before updating secrets.",
    )
    parser.add_argument(
        "--skip-s3-check",
        action="store_true",
        help="Skip the local S3 access check before writing workspace secrets.",
    )
    return parser.parse_args()


def _run(
    args: list[str], *, input_text: str | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        input=input_text,
        text=True,
        capture_output=True,
        check=True,
    )


def _aws_profile_arg(profile: str) -> list[str]:
    return ["--profile", profile]


def _profile_region(profile: str) -> str | None:
    try:
        result = _run(["aws", "configure", "get", "region", *_aws_profile_arg(profile)])
    except subprocess.CalledProcessError:
        return None
    region = result.stdout.strip()
    return region or None


def _region(args: argparse.Namespace) -> str:
    return (
        args.region
        or _profile_region(args.aws_profile)
        or os.environ.get("AWS_DEFAULT_REGION")
        or os.environ.get("AWS_REGION")
        or "us-east-1"
    )


def _export_credentials(profile: str) -> dict[str, str]:
    result = _run(
        [
            "aws",
            "configure",
            "export-credentials",
            *_aws_profile_arg(profile),
            "--format",
            "json",
        ]
    )
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise ValueError("aws export-credentials did not return a JSON object")
    mapping = {
        "AWS_ACCESS_KEY_ID": payload.get("AccessKeyId"),
        "AWS_SECRET_ACCESS_KEY": payload.get("SecretAccessKey"),
        "AWS_SESSION_TOKEN": payload.get("SessionToken"),
    }
    missing = [key for key, value in mapping.items() if not isinstance(value, str)]
    if missing:
        raise ValueError(
            "aws export-credentials did not return required keys: " + ", ".join(missing)
        )
    return {key: str(value) for key, value in mapping.items()}


def _check_aws_access(args: argparse.Namespace) -> None:
    _run(["aws", "sts", "get-caller-identity", *_aws_profile_arg(args.aws_profile)])
    if not args.skip_s3_check:
        _run(["aws", "s3", "ls", args.s3_check, *_aws_profile_arg(args.aws_profile)])


def _set_secret(*, env: str, name: str, value: str) -> None:
    _run(
        ["macrodata", "secrets", "set", name, "--env", env, "--value-stdin"],
        input_text=value,
    )


def _secret_payload(args: argparse.Namespace) -> dict[str, str]:
    payload = _export_credentials(args.aws_profile)
    payload["AWS_DEFAULT_REGION"] = _region(args)
    return payload


def main() -> int:
    args = _parse_args()
    _check_aws_access(args)
    payload = _secret_payload(args)
    for name in SECRET_NAMES:
        _set_secret(env=args.secret_env, name=name, value=payload[name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
