from __future__ import annotations

import os
import time
from pathlib import Path

from refiner.ledger import FsLedger
from refiner.ledger.backend.base import LedgerConfig
from refiner.ledger.shard import Shard


def test_shard_id_is_stable() -> None:
    a1 = Shard(path="a", start=0, end=10).id
    a2 = Shard(path="a", start=0, end=10).id
    b = Shard(path="a", start=0, end=11).id
    assert a1 == a2
    assert a1 != b


def test_fs_ledger_claim_complete(tmp_path: Path) -> None:
    cfg = LedgerConfig(lease_seconds=600, heartbeat_seconds=120)
    ledger = FsLedger(run_id="run1", worker_id=1, workdir=str(tmp_path), config=cfg)

    s1 = Shard(path="p1", start=0, end=1)
    s2 = Shard(path="p2", start=0, end=2)
    ledger.seed_shards([s1, s2])

    got1 = ledger.claim()
    assert got1 is not None
    ledger.heartbeat(got1)
    ledger.complete(got1)

    got2 = ledger.claim()
    assert got2 is not None
    assert got2.id != got1.id


def test_fs_ledger_enqueue_without_worker_id(tmp_path: Path) -> None:
    cfg = LedgerConfig(lease_seconds=600, heartbeat_seconds=120)
    ledger = FsLedger(run_id="run3", worker_id=None, workdir=str(tmp_path), config=cfg)
    ledger.seed_shards([Shard(path="p", start=0, end=1)])


def test_fs_ledger_enqueue_overwrites_manifest(tmp_path: Path) -> None:
    cfg = LedgerConfig(lease_seconds=600, heartbeat_seconds=120)
    ledger = FsLedger(run_id="run4", worker_id=1, workdir=str(tmp_path), config=cfg)
    s1 = Shard(path="p1", start=0, end=1)
    s2 = Shard(path="p2", start=0, end=2)
    ledger.seed_shards([s1])
    ledger.seed_shards([s2])
    got = ledger.claim()
    assert got is not None
    assert got.path == "p2"


def test_fs_ledger_reclaims_stale_lease(tmp_path: Path) -> None:
    cfg = LedgerConfig(lease_seconds=10, heartbeat_seconds=120)
    run_id = "run2"
    ledger1 = FsLedger(run_id=run_id, worker_id=1, workdir=str(tmp_path), config=cfg)
    ledger2 = FsLedger(run_id=run_id, worker_id=2, workdir=str(tmp_path), config=cfg)

    shard = Shard(path="p", start=0, end=1)
    ledger1.seed_shards([shard])

    got = ledger1.claim()
    assert got is not None

    # Force the lease to be stale by setting an old mtime.
    leased_dir = tmp_path / "runs" / run_id / "ledger" / "leased"
    leased_files = list(leased_dir.iterdir())
    assert len(leased_files) == 1
    lease_path = leased_files[0]
    old = int(time.time()) - 3600
    os.utime(lease_path, (old, old))

    got2 = ledger2.claim()
    assert got2 is not None
    assert got2.id == got.id
