from __future__ import annotations

import hashlib
import os
import statistics
import tempfile
import time

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as fs
import pyarrow.parquet as pq


NUM_ROWS = 200_000
ROW_GROUP_SIZE = 10_000
BATCH_SIZE = 4096
REPEATS = 8


class CountingFile:
    def __init__(self, path: str):
        self._handle = open(path, "rb")
        self.bytes_read = 0

    def read(self, n: int = -1) -> bytes:
        data = self._handle.read(n)
        self.bytes_read += len(data)
        return data

    def readinto(self, buffer) -> int | None:
        count = self._handle.readinto(buffer)
        if count is not None:
            self.bytes_read += count
        return count

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._handle.seek(offset, whence)

    def tell(self) -> int:
        return self._handle.tell()

    def close(self) -> None:
        self._handle.close()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    @property
    def closed(self) -> bool:
        return self._handle.closed


class CountingFileSystemHandler(fs.FileSystemHandler):
    def __init__(self, root: str):
        self.root = root
        self.opened_files: list[CountingFile] = []

    def get_type_name(self) -> str:
        return "counting"

    def normalize_path(self, path: str) -> str:
        return path

    def get_file_info(self, paths):
        if isinstance(paths, str):
            paths = [paths]
        return [
            fs.FileInfo(
                path,
                fs.FileType.File,
                size=os.path.getsize(os.path.join(self.root, path)),
            )
            for path in paths
        ]

    def get_file_info_selector(self, selector):
        raise NotImplementedError

    def open_input_file(self, path: str):
        opened = CountingFile(os.path.join(self.root, path))
        self.opened_files.append(opened)
        return pa.PythonFile(opened, mode="r")

    def open_input_stream(self, path: str):
        return self.open_input_file(path)

    def create_dir(self, path, recursive=True):
        raise NotImplementedError

    def delete_dir(self, path):
        raise NotImplementedError

    def delete_dir_contents(self, path, missing_dir_ok=False):
        raise NotImplementedError

    def delete_root_dir_contents(self):
        raise NotImplementedError

    def delete_file(self, path):
        raise NotImplementedError

    def move(self, src, dest):
        raise NotImplementedError

    def copy_file(self, src, dest):
        raise NotImplementedError

    def open_output_stream(self, path, metadata=None):
        raise NotImplementedError

    def open_append_stream(self, path, metadata=None):
        raise NotImplementedError

    def reset(self) -> None:
        self.opened_files.clear()

    def total_bytes(self) -> int:
        return sum(file.bytes_read for file in self.opened_files)


def _payload(row_index: int) -> bytes:
    digest = hashlib.sha256(str(row_index).encode("utf-8")).digest()
    return (digest * 32)[:1024]


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        parquet_path = os.path.join(temp_dir, "bench.parquet")
        pq.write_table(
            pa.table(
                {
                    "id": list(range(NUM_ROWS)),
                    "payload": [_payload(i) for i in range(NUM_ROWS)],
                }
            ),
            parquet_path,
            row_group_size=ROW_GROUP_SIZE,
            compression="snappy",
        )

        handler = CountingFileSystemHandler(temp_dir)
        pyfs = fs.PyFileSystem(handler)
        fragment = next(
            ds.dataset(
                "bench.parquet", filesystem=pyfs, format="parquet"
            ).get_fragments()
        )

        def run_iter_batches() -> tuple[int, int]:
            opened = CountingFile(parquet_path)
            parquet_file = pq.ParquetFile(pa.PythonFile(opened, mode="r"))
            rows = 0
            for batch in parquet_file.iter_batches(
                batch_size=BATCH_SIZE,
                columns=["id", "payload"],
            ):
                rows += batch.num_rows
            return rows, opened.bytes_read

        def run_scanner() -> tuple[int, int]:
            handler.reset()
            rows = 0
            for batch in fragment.scanner(
                columns=["id", "payload"],
                filter=None,
                batch_size=BATCH_SIZE,
            ).to_batches():
                rows += batch.num_rows
            return rows, handler.total_bytes()

        iter_times: list[float] = []
        iter_bytes: list[int] = []
        scanner_times: list[float] = []
        scanner_bytes: list[int] = []

        for _ in range(REPEATS):
            start = time.perf_counter()
            rows, nbytes = run_iter_batches()
            iter_times.append(time.perf_counter() - start)
            iter_bytes.append(nbytes)
            assert rows == NUM_ROWS

            start = time.perf_counter()
            rows, nbytes = run_scanner()
            scanner_times.append(time.perf_counter() - start)
            scanner_bytes.append(nbytes)
            assert rows == NUM_ROWS

        print(
            {
                "file_size": os.path.getsize(parquet_path),
                "iter_batches_mean_s": statistics.mean(iter_times),
                "iter_batches_min_s": min(iter_times),
                "iter_batches_bytes": iter_bytes[0],
                "scanner_mean_s": statistics.mean(scanner_times),
                "scanner_min_s": min(scanner_times),
                "scanner_bytes": scanner_bytes[0],
                "speedup_scanner_vs_iter_mean": statistics.mean(iter_times)
                / statistics.mean(scanner_times),
            }
        )


if __name__ == "__main__":
    main()
