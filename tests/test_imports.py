from __future__ import annotations

import subprocess
import sys


def test_inference_import_does_not_cycle() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import refiner as mdr; mdr.inference"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr


def test_progress_is_available_from_top_level() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import refiner as mdr; assert callable(mdr.progress); assert mdr.Progress",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
