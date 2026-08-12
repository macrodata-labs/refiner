from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import random
import time
from urllib.parse import urlsplit

import pyarrow as pa

import refiner as mdr
from refiner.io import DataFolder


BUCKET = "macrodata-hands-research"
ROOT = "data-lake/hand-detection"
OUTPUT_URI = f"s3://{BUCKET}/{ROOT}/ego-hos-read-benchmark-results-20260812-v2"
ROW_COUNT = 11_743
RANDOM_COUNT = 256
CONCURRENCY = 32
REPETITIONS = 3


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    repetition: int
    layout: str
    workload: str
    seconds: float
    images: int
    payload_bytes: int

    def row(self) -> dict[str, object]:
        row = asdict(self)
        row["images_per_second"] = self.images / self.seconds
        row["payload_mib_per_second"] = self.payload_bytes / (1 << 20) / self.seconds
        return row


def _consume_binary_array(array: pa.Array | pa.ChunkedArray) -> tuple[int, int]:
    images = 0
    payload_bytes = 0
    for value in array.to_pylist():
        if value[:3] != b"\xff\xd8\xff" or value[-2:] != b"\xff\xd9":
            raise ValueError("invalid JPEG")
        images += 1
        payload_bytes += len(value)
    return images, payload_bytes


def _read_blob(client: object, reference: dict[str, object]) -> tuple[int, int]:
    parsed = urlsplit(str(reference["path"]))
    offset = int(reference["offset"])
    size = int(reference["size"])
    response = client.get_object(
        Bucket=parsed.netloc,
        Key=parsed.path.lstrip("/"),
        Range=f"bytes={offset}-{offset + size - 1}",
    )
    data = response["Body"].read()
    if len(data) != size or data[:3] != b"\xff\xd8\xff" or data[-2:] != b"\xff\xd9":
        raise ValueError("invalid ranged JPEG")
    return 1, len(data)


def _read_file(client: object, path: str) -> tuple[int, int]:
    parsed = urlsplit(path)
    response = client.get_object(
        Bucket=parsed.netloc,
        Key=parsed.path.lstrip("/"),
    )
    data = response["Body"].read()
    if data[:3] != b"\xff\xd8\xff" or data[-2:] != b"\xff\xd9":
        raise ValueError("invalid JPEG object")
    return 1, len(data)


def _parallel_payloads(reader: object, values: list[object]) -> tuple[int, int]:
    images = 0
    payload_bytes = 0
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        for start in range(0, len(values), 512):
            for count, size in executor.map(reader, values[start : start + 512]):
                images += count
                payload_bytes += size
    return images, payload_bytes


def _inline_lance(
    dataset: object,
    indices: list[int] | None,
    repetition: int,
) -> BenchmarkResult:
    started = time.perf_counter()
    images = 0
    payload_bytes = 0
    if indices is None:
        for batch in dataset.scanner(columns=["image"], batch_size=256).to_batches():
            count, size = _consume_binary_array(batch.column(0))
            images += count
            payload_bytes += size
        workload = "full"
    else:
        table = dataset.take(indices, columns=["image"])
        images, payload_bytes = _consume_binary_array(table.column("image"))
        workload = "random256"
    return BenchmarkResult(
        repetition,
        "lance-inline",
        workload,
        time.perf_counter() - started,
        images,
        payload_bytes,
    )


def _external_lance(
    layout: str,
    dataset: object,
    client: object,
    indices: list[int] | None,
    repetition: int,
    *,
    packed: bool,
) -> BenchmarkResult:
    started = time.perf_counter()
    values: list[object] = []
    if indices is None:
        for batch in dataset.scanner(columns=["image"], batch_size=1024).to_batches():
            values.extend(batch.column(0).to_pylist())
        workload = "full"
    else:
        values = dataset.take(indices, columns=["image"]).column("image").to_pylist()
        workload = "random256"
    reader = (
        (lambda value: _read_blob(client, value))
        if packed
        else (lambda value: _read_file(client, value))
    )
    images, payload_bytes = _parallel_payloads(reader, values)
    return BenchmarkResult(
        repetition,
        layout,
        workload,
        time.perf_counter() - started,
        images,
        payload_bytes,
    )


def _parquet(
    parquet: object,
    indices: list[int] | None,
    repetition: int,
) -> BenchmarkResult:
    started = time.perf_counter()
    images = 0
    payload_bytes = 0
    if indices is None:
        for batch in parquet.iter_batches(batch_size=256, columns=["image"]):
            count, size = _consume_binary_array(batch.column(0))
            images += count
            payload_bytes += size
        workload = "full"
    else:
        starts = []
        current = 0
        for group in range(parquet.metadata.num_row_groups):
            starts.append(current)
            current += parquet.metadata.row_group(group).num_rows
        by_group: dict[int, list[int]] = {}
        for index in indices:
            group = max(i for i, start in enumerate(starts) if start <= index)
            by_group.setdefault(group, []).append(index - starts[group])
        for group, offsets in by_group.items():
            array = parquet.read_row_group(group, columns=["image"]).column("image")
            count, size = _consume_binary_array(array.take(pa.array(offsets)))
            images += count
            payload_bytes += size
        workload = "random256"
    return BenchmarkResult(
        repetition,
        "parquet-inline",
        workload,
        time.perf_counter() - started,
        images,
        payload_bytes,
    )


def _parquet_key(client: object) -> str:
    response = client.list_objects_v2(
        Bucket=BUCKET,
        Prefix=f"{ROOT}/ego-hos-images-inside-parquet/",
    )
    return next(
        item["Key"] for item in response["Contents"] if item["Key"].endswith(".parquet")
    )


def benchmark_reads(
    _task_rank: int,
    _num_tasks: int,
):
    import boto3
    from botocore.config import Config
    import lance
    import pyarrow.fs as pafs
    import pyarrow.parquet as pq

    client = boto3.client(
        "s3",
        config=Config(
            max_pool_connections=CONCURRENCY,
            retries={"max_attempts": 8, "mode": "adaptive"},
        ),
    )
    inline = lance.dataset(f"s3://{BUCKET}/{ROOT}/ego-hos-images-inside")
    packed = lance.dataset(f"s3://{BUCKET}/{ROOT}/ego-hos")
    files = lance.dataset(f"s3://{BUCKET}/{ROOT}/ego-hos-images-single-file")
    arrow_s3 = pafs.S3FileSystem(region="us-east-1")
    parquet_source = arrow_s3.open_input_file(f"{BUCKET}/{_parquet_key(client)}")
    parquet = pq.ParquetFile(parquet_source)
    indices = sorted(random.Random(20260812).sample(range(ROW_COUNT), RANDOM_COUNT))

    try:
        for repetition in range(1, REPETITIONS + 1):
            results = [
                _inline_lance(inline, indices, repetition),
                _external_lance(
                    "lance-packed",
                    packed,
                    client,
                    indices,
                    repetition,
                    packed=True,
                ),
                _external_lance(
                    "lance-files",
                    files,
                    client,
                    indices,
                    repetition,
                    packed=False,
                ),
                _parquet(parquet, indices, repetition),
                _inline_lance(inline, None, repetition),
                _external_lance(
                    "lance-packed",
                    packed,
                    client,
                    None,
                    repetition,
                    packed=True,
                ),
                _external_lance(
                    "lance-files",
                    files,
                    client,
                    None,
                    repetition,
                    packed=False,
                ),
                _parquet(parquet, None, repetition),
            ]
            for workload in ("random256", "full"):
                subset = [result for result in results if result.workload == workload]
                if len({result.images for result in subset}) != 1:
                    raise ValueError(f"image counts disagree for {workload}")
                if len({result.payload_bytes for result in subset}) != 1:
                    raise ValueError(f"payload sizes disagree for {workload}")
            for result in results:
                row = result.row()
                print(f"BENCHMARK {row}", flush=True)
                yield row
    finally:
        parquet_source.close()


if __name__ == "__main__":
    (
        mdr.task(benchmark_reads, num_tasks=1)
        .write_jsonl(DataFolder(OUTPUT_URI, auto_mkdir=False))
        .launch_cloud(
            name="egohos-read-benchmark",
            num_workers=1,
            cpus_per_worker=2,
            mem_mb_per_worker=8192,
            dependencies=["boto3>=1.34,<2", "pylance>=4.0.1,<11"],
            secrets=mdr.Secrets.env(
                name="default",
                keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
            ),
        )
    )
