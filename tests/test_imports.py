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
