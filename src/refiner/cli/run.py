from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

from refiner.cli.cloud_run import (
    CloudAttachDetached,
    _ATTACH_MODE_ENV_VAR,
    normalize_attach_mode,
)
from refiner.cli.local_run import LocalLaunchResumeError


def cmd_run(args: argparse.Namespace) -> int:
    script = Path(args.script).expanduser()
    if not script.exists():
        print(f"Script not found: {script}", file=sys.stderr)
        return 1
    if not script.is_file():
        print(f"Not a file: {script}", file=sys.stderr)
        return 1

    script_args = list(args.script_args)
    if script_args[:1] == ["--"]:
        script_args = script_args[1:]
    original_argv = sys.argv
    original_sys_path = list(sys.path)
    original_logs = os.environ.get("REFINER_LOCAL_LOGS")
    original_attach = os.environ.get(_ATTACH_MODE_ENV_VAR)
    cwd_entries = {"", str(Path.cwd())}
    try:
        if args.attach is not None:
            os.environ[_ATTACH_MODE_ENV_VAR] = normalize_attach_mode(args.attach)
        elif original_attach is None:
            os.environ.pop(_ATTACH_MODE_ENV_VAR, None)
        if args.logs is not None:
            os.environ["REFINER_LOCAL_LOGS"] = args.logs
        elif original_logs is None:
            os.environ.pop("REFINER_LOCAL_LOGS", None)
        sys.argv = [str(script), *script_args]
        sys.path = [
            str(script.parent.resolve()),
            *[entry for entry in original_sys_path if entry not in cwd_entries],
        ]
        try:
            runpy.run_path(str(script), run_name="__main__")
        except BrokenPipeError:
            return 141
        except CloudAttachDetached:
            return 130
        except KeyboardInterrupt as err:
            if err.args and not sys.stdout.isatty():
                print(str(err), file=sys.stderr)
            elif not err.args:
                print("Interrupted.", file=sys.stderr)
            return 130
        except LocalLaunchResumeError as err:
            if not sys.stdout.isatty():
                print(str(err), file=sys.stderr)
            return 1
        except SystemExit as err:
            code = err.code
            if code is None:
                return 0
            if isinstance(code, int):
                return code
            print(str(code), file=sys.stderr)
            return 1
    finally:
        sys.argv = original_argv
        sys.path = original_sys_path
        if original_logs is None:
            os.environ.pop("REFINER_LOCAL_LOGS", None)
        else:
            os.environ["REFINER_LOCAL_LOGS"] = original_logs
        if original_attach is None:
            os.environ.pop(_ATTACH_MODE_ENV_VAR, None)
        else:
            os.environ[_ATTACH_MODE_ENV_VAR] = original_attach
    return 0
