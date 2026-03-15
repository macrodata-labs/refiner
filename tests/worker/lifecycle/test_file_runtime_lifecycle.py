from __future__ import annotations

import os
import time
from pathlib import Path

from refiner.pipeline.data.shard import FilePart, FilePartsDescriptor
from refiner.worker.lifecycle import LocalRuntimeLifecycle
from refiner.pipeline.data.shard import Shard
from refiner.platform.client.models import FinalizedShardWorker


def _shard(
    path: str, start: int, end: int, *, global_ordinal: int | None = None
) -> Shard:
    return Shard.from_file_parts(
        [FilePart(path=path, start=start, end=end)],
        global_ordinal=global_ordinal,
    )


def test_shard_id_is_stable() -> None:
    a1 = _shard("a", 0, 10).id
    a2 = _shard("a", 0, 10).id
    b = _shard("a", 0, 11).id
    assert a1 == a2
    assert a1 != b


def test_file_runtime_claim_complete(tmp_path: Path) -> None:
    lifecycle = LocalRuntimeLifecycle(
        job_id="job1", worker_id="1", workdir=str(tmp_path)
    )

    s1 = _shard("p1", 0, 1)
    s2 = _shard("p2", 0, 2)
    lifecycle.seed_shards([s1, s2])

    got1 = lifecycle.claim()
    assert got1 is not None
    lifecycle.heartbeat([got1])
    lifecycle.complete(got1)

    got2 = lifecycle.claim(previous=got1)
    assert got2 is not None
    assert got2.id != got1.id


def test_file_runtime_seed_without_worker_id(tmp_path: Path) -> None:
    lifecycle = LocalRuntimeLifecycle(
        job_id="job3", worker_id=None, workdir=str(tmp_path)
    )
    lifecycle.seed_shards([_shard("p", 0, 1)])


def test_file_runtime_seed_overwrites_previous(tmp_path: Path) -> None:
    lifecycle = LocalRuntimeLifecycle(
        job_id="job4", worker_id="1", workdir=str(tmp_path)
    )
    s1 = _shard("p1", 0, 1)
    s2 = _shard("p2", 0, 2)
    lifecycle.seed_shards([s1])
    lifecycle.seed_shards([s2])
    got = lifecycle.claim()
    assert got is not None
    assert isinstance(got.descriptor, FilePartsDescriptor)
    assert got.descriptor.parts[0].path == "p2"


def test_file_runtime_reclaims_stale_lease(tmp_path: Path) -> None:
    job_id = "job2"
    lifecycle1 = LocalRuntimeLifecycle(
        job_id=job_id,
        worker_id="1",
        workdir=str(tmp_path),
        lease_seconds=10,
    )
    lifecycle2 = LocalRuntimeLifecycle(
        job_id=job_id,
        worker_id="2",
        workdir=str(tmp_path),
        lease_seconds=10,
    )

    shard = _shard("p", 0, 1)
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
    stage_zero = LocalRuntimeLifecycle(
        job_id="job5", stage_index=0, worker_id="1", workdir=str(tmp_path)
    )
    stage_one = LocalRuntimeLifecycle(
        job_id="job5", stage_index=1, worker_id="1", workdir=str(tmp_path)
    )

    stage_zero.seed_shards([_shard("p0", 0, 1)])
    stage_one.seed_shards([_shard("p1", 0, 1)])

    claimed_zero = stage_zero.claim()
    claimed_one = stage_one.claim()

    assert claimed_zero is not None
    assert isinstance(claimed_zero.descriptor, FilePartsDescriptor)
    assert claimed_zero.descriptor.parts[0].path == "p0"
    assert claimed_one is not None
    assert isinstance(claimed_one.descriptor, FilePartsDescriptor)
    assert claimed_one.descriptor.parts[0].path == "p1"


def test_file_runtime_prefers_same_locality_without_exact_boundary(
    tmp_path: Path,
) -> None:
    lifecycle = LocalRuntimeLifecycle(
        job_id="job6", worker_id="1", workdir=str(tmp_path)
    )
    shard0 = _shard("p0", 0, 1, global_ordinal=0)
    shard1 = _shard("p0", 10, 11, global_ordinal=1)
    shard2 = _shard("p1", 0, 1, global_ordinal=2)
    lifecycle.seed_shards([shard0, shard1, shard2])

    first = lifecycle.claim()
    assert first is not None
    lifecycle.complete(first)

    second = lifecycle.claim(previous=first)
    assert second is not None
    assert second.start_key == shard1.start_key


def test_file_runtime_reports_finalized_workers(tmp_path: Path) -> None:
    lifecycle = LocalRuntimeLifecycle(
        job_id="job7", worker_id="2", workdir=str(tmp_path)
    )
    shard = _shard("p", 0, 1)
    lifecycle.seed_shards([shard])
    claimed = lifecycle.claim()
    assert claimed is not None
    lifecycle.complete(claimed)

    assert lifecycle.finalized_workers() == [
        FinalizedShardWorker(shard_id=shard.id, worker_id="2")
    ]
