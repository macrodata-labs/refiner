from __future__ import annotations

import subprocess

import pytest

from refiner.worker.resources.gpu import available_gpu_ids


def test_available_gpu_ids_reports_nvidia_smi_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        "refiner.worker.resources.gpu.shutil.which", lambda _: "/bin/nvidia-smi"
    )

    def _raise_called_process_error(*args, **kwargs):  # noqa: ANN002, ANN003
        del args, kwargs
        raise subprocess.CalledProcessError(
            7,
            ["/bin/nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            stderr="driver not loaded\n",
        )

    monkeypatch.setattr(
        "refiner.worker.resources.gpu.subprocess.check_output",
        _raise_called_process_error,
    )

    with pytest.raises(
        RuntimeError, match="nvidia-smi failed with exit code 7: driver not loaded"
    ):
        available_gpu_ids()
