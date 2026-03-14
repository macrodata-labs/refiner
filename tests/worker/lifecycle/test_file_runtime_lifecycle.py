from __future__ import annotations

import os
import time
from pathlib import Path

from refiner.worker.lifecycle import FileRuntimeLifecycle
from refiner.pipeline.data.shard import Shard


def test_shard_id_is_stable() -> None:
    a1 = Shard(path="a", start=0, end=10).id
    a2 = Shard(path="a", start=0, end=10).id
    b = Shard(path="a", start=0, end=11).id
    assert a1 == a2
    assert a1 != b


def test_file_runtime_claim_complete(tmp_path: Path) -> None:
    lifecycle = FileRuntimeLifecycle(job_id="job1", worker_id=1, workdir=str(tmp_path))

    s1 = Shard(path="p1", start=0, end=1)
    s2 = Shard(path="p2", start=0, end=2)
    lifecycle.seed_shards([s1, s2])

    got1 = lifecycle.claim()
    assert got1 is not None
    lifecycle.heartbeat([got1])
    lifecycle.complete(got1)

    got2 = lifecycle.claim(previous=got1)
    assert got2 is not None
    assert got2.id != got1.id


def test_file_runtime_seed_without_worker_id(tmp_path: Path) -> None:
    lifecycle = FileRuntimeLifecycle(
        job_id="job3", worker_id=None, workdir=str(tmp_path)
    )
    lifecycle.seed_shards([Shard(path="p", start=0, end=1)])


def test_file_runtime_seed_overwrites_previous(tmp_path: Path) -> None:
    lifecycle = FileRuntimeLifecycle(job_id="job4", worker_id=1, workdir=str(tmp_path))
    s1 = Shard(path="p1", start=0, end=1)
    s2 = Shard(path="p2", start=0, end=2)
    lifecycle.seed_shards([s1])
    lifecycle.seed_shards([s2])
    got = lifecycle.claim()
    assert got is not None
    assert got.path == "p2"


def test_file_runtime_reclaims_stale_lease(tmp_path: Path) -> None:
    job_id = "job2"
    lifecycle1 = FileRuntimeLifecycle(
        job_id=job_id,
        worker_id=1,
        workdir=str(tmp_path),
        lease_seconds=10,
    )
    lifecycle2 = FileRuntimeLifecycle(
        job_id=job_id,
        worker_id=2,
        workdir=str(tmp_path),
        lease_seconds=10,
    )

    shard = Shard(path="p", start=0, end=1)
    lifecycle1.seed_shards([shard])

    got = lifecycle1.claim()
    assert got is not None

    leased_dir = tmp_path / "runs" / job_id / "lifecycle" / "stage-0" / "leased"
    leased_files = list(leased_dir.iterdir())
    assert len(leased_files) == 1
    lease_path = leased_files[0]
    old = int(time.time()) - 3600
    os.utime(lease_path, (old, old))

    got2 = lifecycle2.claim()
    assert got2 is not None
    assert got2.id == got.id


def test_file_runtime_stages_are_isolated(tmp_path: Path) -> None:
    stage_zero = FileRuntimeLifecycle(
        job_id="job5", stage_index=0, worker_id=1, workdir=str(tmp_path)
    )
    stage_one = FileRuntimeLifecycle(
        job_id="job5", stage_index=1, worker_id=1, workdir=str(tmp_path)
    )

    stage_zero.seed_shards([Shard(path="p0", start=0, end=1)])
    stage_one.seed_shards([Shard(path="p1", start=0, end=1)])

    claimed_zero = stage_zero.claim()
    claimed_one = stage_one.claim()

    assert claimed_zero is not None
    assert claimed_zero.path == "p0"
    assert claimed_one is not None
    assert claimed_one.path == "p1"


def test_file_runtime_prefers_same_locality_without_exact_boundary(
    tmp_path: Path,
) -> None:
    lifecycle = FileRuntimeLifecycle(job_id="job6", worker_id=1, workdir=str(tmp_path))
    shard0 = Shard(path="p0", start=0, end=1, global_ordinal=0)
    shard1 = Shard(path="p0", start=10, end=11, global_ordinal=1)
    shard2 = Shard(path="p1", start=0, end=1, global_ordinal=2)
    lifecycle.seed_shards([shard0, shard1, shard2])

    first = lifecycle.claim()
    assert first is not None
    lifecycle.complete(first)

    second = lifecycle.claim(previous=first)
    assert second is not None
    assert second.start_key == shard1.start_key
