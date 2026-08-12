from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import refiner as mdr


DATASET_URI = (
    "s3://macrodata-hands-research/data-lake/hand-detection/ego-hos-images-inside"
)
APPENDED_SAMPLE_SUFFIX = "__refiner_append_test_20260812"
ZERO_COLUMN = "refiner_test_zero"
FAILED_ATTEMPT_FRAGMENT = (
    "data-lake/hand-detection/ego-hos-images-inside/data/"
    "100101011101100011111100c8d96741539bdbbf949b0b7d20.lance"
)


def append_one_sample(_task_rank: int, _num_tasks: int):
    import lance
    from loguru import logger

    before = lance.dataset(DATASET_URI)
    before_version = int(before.version)
    before_count = int(before.count_rows())
    first_id = before.take([0], columns=["sample_id"])["sample_id"][0].as_py()
    appended_id = f"{first_id}{APPENDED_SAMPLE_SUFFIX}"
    if before.scanner(
        columns=["sample_id"],
        filter=f"sample_id = '{appended_id}'",
    ).count_rows():
        raise ValueError(f"sample already appended: {appended_id}")

    pipeline = (
        mdr.load_lance(
            DATASET_URI,
            version=before_version,
            batch_size=256,
            num_shards=1,
        )
        .filter(mdr.col("sample_id") == first_id)
        .with_column("sample_id", appended_id)
        .write_lance_dataset(DATASET_URI, mode="append")
    )
    with tempfile.TemporaryDirectory(prefix="egohos-append-") as rundir:
        try:
            pipeline.launch_local(
                name="egohos-append-one-sample",
                num_workers=1,
                rundir=rundir,
            )
        except Exception:
            for log_path in Path(rundir).glob("stage-*/logs/*.log"):
                logger.error("NESTED_REFINER_LOG\n{}", log_path.read_text())
            raise

    after = lance.dataset(DATASET_URI)
    after_count = int(after.count_rows())
    appended_count = int(
        after.scanner(
            columns=["sample_id"],
            filter=f"sample_id = '{appended_id}'",
        ).count_rows()
    )
    if after_count != before_count + 1 or appended_count != 1:
        raise ValueError(
            f"append verification failed: before={before_count}, "
            f"after={after_count}, matches={appended_count}"
        )
    result = {
        "before_version": before_version,
        "after_version": int(after.version),
        "before_rows": before_count,
        "after_rows": after_count,
        "appended_sample_id": appended_id,
    }
    logger.warning("EGOHOS_APPEND_RESULT {}", result)
    yield result


def add_zero_column(_task_rank: int, _num_tasks: int):
    import boto3  # ty: ignore[unresolved-import]
    from botocore.exceptions import ClientError
    import lance
    from loguru import logger

    before = lance.dataset(DATASET_URI)
    before_version = int(before.version)
    before_count = int(before.count_rows())
    if ZERO_COLUMN in before.schema.names:
        raise ValueError(f"column already exists: {ZERO_COLUMN}")

    referenced_files = {
        data_file.path
        for fragment in before.get_fragments()
        for data_file in fragment.data_files()
    }
    failed_relative_path = FAILED_ATTEMPT_FRAGMENT.rsplit("/data/", 1)[1]
    if failed_relative_path in referenced_files:
        raise ValueError("failed-attempt fragment unexpectedly became referenced")
    client = boto3.client("s3")
    orphan_removed = False
    try:
        client.head_object(
            Bucket="macrodata-hands-research",
            Key=FAILED_ATTEMPT_FRAGMENT,
        )
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") not in {"404", "NoSuchKey"}:
            raise
    else:
        client.delete_object(
            Bucket="macrodata-hands-research",
            Key=FAILED_ATTEMPT_FRAGMENT,
        )
        orphan_removed = True

    pipeline = (
        mdr.load_lance(
            DATASET_URI,
            version=before_version,
            columns=["sample_id"],
            batch_size=4096,
        )
        .with_column(ZERO_COLUMN, 0)
        .write_lance_dataset(
            DATASET_URI,
            mode="add_columns",
            columns=[ZERO_COLUMN],
        )
    )
    with tempfile.TemporaryDirectory(prefix="egohos-add-column-") as rundir:
        try:
            pipeline.launch_local(
                name="egohos-add-zero-column",
                num_workers=1,
                rundir=rundir,
            )
        except Exception:
            for log_path in Path(rundir).glob("stage-*/logs/*.log"):
                logger.error("NESTED_REFINER_LOG\n{}", log_path.read_text())
            raise

    after = lance.dataset(DATASET_URI)
    after_count = int(after.count_rows())
    values = after.to_table(columns=[ZERO_COLUMN])[ZERO_COLUMN]
    if (
        after_count != before_count
        or values.null_count
        or len(values) != before_count
        or values.to_pylist() != [0] * before_count
    ):
        raise ValueError("add_columns verification failed")
    result = {
        "before_version": before_version,
        "after_version": int(after.version),
        "rows": after_count,
        "column": ZERO_COLUMN,
        "type": str(after.schema.field(ZERO_COLUMN).type),
        "all_zero": True,
        "failed_attempt_orphan_removed": orphan_removed,
    }
    logger.warning("EGOHOS_ADD_COLUMN_RESULT {}", result)
    yield result


def inspect_fragments(_task_rank: int, _num_tasks: int):
    import json

    import boto3  # ty: ignore[unresolved-import]
    import lance
    from loguru import logger

    dataset = lance.dataset(DATASET_URI)
    fragments = [
        {
            "fragment_id": int(fragment.fragment_id),
            "rows": int(fragment.count_rows()),
            "files": [
                {"path": data_file.path, "fields": list(data_file.fields)}
                for data_file in fragment.data_files()
            ],
        }
        for fragment in dataset.get_fragments()
    ]
    client = boto3.client("s3")
    prefix = "data-lake/hand-detection/ego-hos-images-inside/_refiner_lance_fragments/"
    sidecars = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket="macrodata-hands-research", Prefix=prefix):
        for item in page.get("Contents", []):
            body = client.get_object(
                Bucket="macrodata-hands-research", Key=item["Key"]
            )["Body"].read()
            payload = json.loads(body)
            if payload.get("source_version") == int(dataset.version):
                sidecars.append(
                    {
                        "key": item["Key"],
                        "source_fragment_ids": payload.get("source_fragment_ids"),
                        "lance_schema": payload.get("lance_schema"),
                    }
                )
    logger.warning(
        "EGOHOS_FRAGMENT_INSPECTION {}",
        json.dumps(
            {
                "version": int(dataset.version),
                "fragments": fragments,
                "sidecars": sidecars,
            },
            sort_keys=True,
        ),
    )
    yield {"fragments": len(fragments), "sidecars": len(sidecars)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("append", "add-columns", "inspect"))
    args = parser.parse_args()
    fn = {
        "append": append_one_sample,
        "add-columns": add_zero_column,
        "inspect": inspect_fragments,
    }[args.mode]
    mdr.task(fn, num_tasks=1).launch_cloud(
        name=f"egohos-lance-{args.mode}",
        num_workers=1,
        cpus_per_worker=2,
        mem_mb_per_worker=8192,
        refiner_extras=["lance", "s3"],
        dependencies=["boto3>=1.34,<2"],
        secrets=mdr.Secrets.env(
            name="default",
            keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
        ),
    )


if __name__ == "__main__":
    main()
