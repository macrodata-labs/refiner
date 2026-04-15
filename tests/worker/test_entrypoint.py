from __future__ import annotations

from refiner.worker import entrypoint


def test_entrypoint_sets_visible_gpus_before_loading_pipeline(
    tmp_path, monkeypatch
) -> None:
    payload_path = tmp_path / "pipeline.cloudpickle"
    payload_path.write_bytes(b"placeholder")
    assignments_dir = tmp_path / "stage-2" / "assignments"
    assignments_dir.mkdir(parents=True, exist_ok=True)
    (assignments_dir / "worker-worker-1.json").write_text("[]")

    events: list[str] = []

    monkeypatch.setattr(
        "sys.argv",
        [
            "entrypoint.py",
            "--pipeline-payload",
            str(payload_path),
            "--job-id",
            "job-1",
            "--stage-index",
            "2",
            "--worker-name",
            "worker-name",
            "--worker-id",
            "worker-1",
            "--rundir",
            str(tmp_path),
            "--gpu-ids",
            "0,1",
        ],
    )
    monkeypatch.setattr(
        entrypoint,
        "set_visible_gpu_ids",
        lambda gpu_ids: events.append(f"set:{','.join(gpu_ids)}"),
    )
    monkeypatch.setattr(
        entrypoint.cloudpickle,
        "load",
        lambda handle: events.append("load") or object(),
    )
    monkeypatch.setattr(
        entrypoint,
        "Worker",
        lambda **kwargs: type(
            "_FakeWorker",
            (),
            {
                "run": staticmethod(
                    lambda: type(
                        "_FakeStats",
                        (),
                        {
                            "claimed": 0,
                            "completed": 0,
                            "failed": 0,
                            "output_rows": 0,
                        },
                    )()
                )
            },
        )(),
    )

    assert entrypoint.main() == 0
    assert events[:2] == ["set:0,1", "load"]
