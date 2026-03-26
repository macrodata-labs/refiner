from __future__ import annotations

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


def _build_table() -> pa.Table:
    ids = pa.array(list(range(NUM_ROWS)), type=pa.int64())
    keep = pa.array([(i // ROW_GROUP_SIZE) % 2 == 0 for i in range(NUM_ROWS)])
    payload = pa.array([f"{i:08d}-" + ("x" * 256) for i in range(NUM_ROWS)])
    return pa.table({"id": ids, "keep": keep, "payload": payload})


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        parquet_path = os.path.join(temp_dir, "bench.parquet")
        pq.write_table(
            _build_table(),
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
        filter_expr = ds.field("keep") == ds.scalar(True)

        def run_scanner_filter() -> tuple[int, int]:
            handler.reset()
            rows = 0
            for batch in fragment.scanner(
                columns=["id", "keep", "payload"],
                filter=filter_expr,
                batch_size=BATCH_SIZE,
            ).to_batches():
                rows += batch.num_rows
            return rows, handler.total_bytes()

        def run_precompute_row_groups() -> tuple[int, int]:
            handler.reset()
            row_groups = [
                row_group.id
                for row_group_fragment in fragment.split_by_row_group(
                    filter=filter_expr
                )
                for row_group in row_group_fragment.row_groups
            ]
            rows = 0
            for batch in (
                fragment.subset(row_group_ids=row_groups)
                .scanner(
                    columns=["id", "keep", "payload"],
                    filter=None,
                    batch_size=BATCH_SIZE,
                )
                .to_batches()
            ):
                rows += batch.num_rows
            return rows, handler.total_bytes()

        scanner_times: list[float] = []
        scanner_bytes: list[int] = []
        precompute_times: list[float] = []
        precompute_bytes: list[int] = []

        expected_rows = NUM_ROWS // 2
        for _ in range(REPEATS):
            start = time.perf_counter()
            rows, nbytes = run_scanner_filter()
            scanner_times.append(time.perf_counter() - start)
            scanner_bytes.append(nbytes)
            assert rows == expected_rows

            start = time.perf_counter()
            rows, nbytes = run_precompute_row_groups()
            precompute_times.append(time.perf_counter() - start)
            precompute_bytes.append(nbytes)
            assert rows == expected_rows

        print(
            {
                "file_size": os.path.getsize(parquet_path),
                "scanner_filter_mean_s": statistics.mean(scanner_times),
                "scanner_filter_min_s": min(scanner_times),
                "scanner_filter_bytes": scanner_bytes[0],
                "precompute_row_groups_mean_s": statistics.mean(precompute_times),
                "precompute_row_groups_min_s": min(precompute_times),
                "precompute_row_groups_bytes": precompute_bytes[0],
                "speedup_precompute_vs_scanner_mean": statistics.mean(scanner_times)
                / statistics.mean(precompute_times),
            }
        )


if __name__ == "__main__":
    main()
