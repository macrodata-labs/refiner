from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from pathlib import Path

from refiner.readers.base import Shard

from .base import BaseLedger
from .config import load_ledger_config_from_env, resolve_workdir

# Import LedgerConfig for type annotations
from .backend.base import LedgerConfig


class FsLedger(BaseLedger):
    """Filesystem-backed shard ledger.

    Layout (under `<workdir>/runs/<run_id>/ledger/`):
      - manifest.json
      - leased/<shard_id>          (text: "<epoch_seconds>\\n<worker_id>\\n")
      - done/<shard_id>            (empty marker file)
      - failed/<shard_id>          (empty marker file)
      - failed/<shard_id>.error    (optional error text)
    """

    def __init__(
        self,
        *,
        worker_id: str,
        run_id: str,
        workdir: str | None = None,
        config: LedgerConfig | None = None,
    ):
        cfg = config or load_ledger_config_from_env()
        super().__init__(worker_id=worker_id, run_id=run_id, config=cfg)
        self.workdir = resolve_workdir(workdir)

        self._root = Path(self.workdir) / "runs" / self.run_id / "ledger"
        self._manifest_path = self._root / "manifest.json"
        self._leased_dir = self._root / "leased"
        self._done_dir = self._root / "done"
        self._failed_dir = self._root / "failed"

        self._root.mkdir(parents=True, exist_ok=True)
        self._leased_dir.mkdir(parents=True, exist_ok=True)
        self._done_dir.mkdir(parents=True, exist_ok=True)
        self._failed_dir.mkdir(parents=True, exist_ok=True)

    def enqueue(self, shards: Iterable[Shard]) -> None:
        # Merge shards into manifest (id -> {path,start,end}) deterministically.
        existing = self._read_manifest()
        by_id: dict[str, dict] = {s["id"]: s for s in existing}

        for shard in shards:
            by_id.setdefault(
                shard.id,
                {
                    "id": shard.id,
                    "path": shard.path,
                    "start": int(shard.start),
                    "end": int(shard.end),
                },
            )

        merged = [by_id[k] for k in sorted(by_id.keys())]
        payload = {"version": 1, "shards": merged}
        self._atomic_write_json(self._manifest_path, payload)

    def claim(self) -> Shard | None:
        now = int(time.time())
        lease_seconds = int(self.config.lease_seconds)

        manifest = self._read_manifest()
        for s in manifest:
            sid = s["id"]
            if (self._done_dir / sid).exists() or (self._failed_dir / sid).exists():
                continue

            lease_path = self._leased_dir / sid
            if self._try_create_lease(lease_path, now):
                return Shard(path=s["path"], start=int(s["start"]), end=int(s["end"]))

            # Someone else has it; if stale, attempt to reclaim.
            last = self._lease_last_heartbeat(lease_path)
            if last is not None and (now - last) > lease_seconds:
                try:
                    lease_path.unlink()
                except FileNotFoundError:
                    pass
                except Exception:
                    continue
                if self._try_create_lease(lease_path, now):
                    return Shard(
                        path=s["path"], start=int(s["start"]), end=int(s["end"])
                    )

        return None

    def heartbeat(self, shard: Shard) -> None:
        lease_path = self._leased_dir / shard.id
        now = int(time.time())
        # Best-effort: refresh the lease file even if it doesn't exist (creates it).
        self._write_lease(lease_path, now)

    def complete(self, shard: Shard) -> None:
        (self._done_dir / shard.id).touch(exist_ok=True)
        try:
            (self._leased_dir / shard.id).unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass

    def fail(self, shard: Shard, error: str | None = None) -> None:
        (self._failed_dir / shard.id).touch(exist_ok=True)
        if error:
            err_path = self._failed_dir / f"{shard.id}.error"
            try:
                err_path.write_text(error)
            except Exception:
                pass
        try:
            (self._leased_dir / shard.id).unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass

    def _read_manifest(self) -> list[dict]:
        if not self._manifest_path.exists():
            return []
        try:
            payload = json.loads(self._manifest_path.read_text())
        except Exception:
            return []
        shards = payload.get("shards")
        if not isinstance(shards, list):
            return []
        out: list[dict] = []
        for s in shards:
            if not isinstance(s, dict):
                continue
            if not isinstance(s.get("id"), str):
                continue
            if not isinstance(s.get("path"), str):
                continue
            out.append(
                {
                    "id": s["id"],
                    "path": s["path"],
                    "start": int(s.get("start", 0)),
                    "end": int(s.get("end", 0)),
                }
            )
        return out

    def _atomic_write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(payload, sort_keys=True))
        tmp.replace(path)

    def _try_create_lease(self, lease_path: Path, now: int) -> bool:
        try:
            fd = os.open(str(lease_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
        except Exception:
            return False
        try:
            os.write(fd, f"{now}\n{self.worker_id}\n".encode("utf-8"))
        finally:
            try:
                os.close(fd)
            except Exception:
                pass
        return True

    def _write_lease(self, lease_path: Path, now: int) -> None:
        try:
            lease_path.write_text(f"{now}\n{self.worker_id}\n")
        except Exception:
            # best-effort
            pass

    def _lease_last_heartbeat(self, lease_path: Path) -> int | None:
        try:
            txt = lease_path.read_text()
        except FileNotFoundError:
            return None
        except Exception:
            # fallback to mtime
            try:
                return int(lease_path.stat().st_mtime)
            except Exception:
                return None

        line = txt.splitlines()[0].strip() if txt else ""
        try:
            return int(line)
        except Exception:
            try:
                return int(lease_path.stat().st_mtime)
            except Exception:
                return None


__all__ = ["FsLedger"]
