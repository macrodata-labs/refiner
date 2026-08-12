from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
import tempfile
import zipfile

import pyarrow as pa

import refiner as mdr
from refiner.io import DataFolder


SOURCE_BUCKET = "macrodata-hands-research"
SOURCE_KEY = "data-lake/hand-detection/ego-hos-source/egohos_dataset.zip"
OUTPUT_ROOT = "s3://macrodata-hands-research/data-lake/hand-detection"
SPLITS = ("train", "val", "test_indomain", "test_outdomain")


def _safe_extract(archive_path: Path, output_dir: Path) -> None:
    root = output_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (output_dir / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe ZIP member path: {member.filename!r}")
        archive.extractall(output_dir)


def _dataset_root(extracted_dir: Path) -> Path:
    candidates = sorted(
        path.parent.parent
        for path in extracted_dir.rglob("train/image")
        if path.is_dir()
    )
    if len(candidates) != 1:
        raise ValueError(
            "Expected exactly one extracted EgoHOS dataset root, found "
            f"{len(candidates)}"
        )
    return candidates[0]


def _contact_png_bytes(semantic_path: Path) -> bytes:
    import cv2  # ty: ignore[unresolved-import]
    import numpy as np

    semantic = cv2.imread(str(semantic_path), cv2.IMREAD_UNCHANGED)
    if semantic is None or semantic.ndim != 2:
        raise ValueError(f"Invalid EgoHOS semantic mask: {semantic_path}")
    hand = np.isin(semantic, (1, 2)).astype(np.uint8)
    first_order_object = np.isin(semantic, (3, 4, 5)).astype(np.uint8)
    kernel = np.ones((5, 5), dtype=np.uint8)
    iterations = semantic.shape[1] // 456
    hand = cv2.dilate(hand, kernel, iterations=iterations)
    first_order_object = cv2.dilate(first_order_object, kernel, iterations=iterations)
    contact = ((hand + first_order_object) == 2).astype(np.uint8)
    encoded, png = cv2.imencode(".png", contact)
    if not encoded:
        raise ValueError(f"Could not encode EgoHOS contact mask: {semantic_path}")
    return png.tobytes()


def _iter_rows(
    dataset_root: Path,
    *,
    inline_images: bool,
) -> Iterator[dict[str, object]]:
    for split in SPLITS:
        split_dir = dataset_root / split
        images = sorted((split_dir / "image").glob("*.jpg"))
        if not images:
            raise FileNotFoundError(f"No JPEG images found for EgoHOS split {split!r}")
        for image_path in images:
            stem = image_path.stem
            semantic_path = split_dir / "label" / f"{stem}.png"
            if not semantic_path.is_file():
                raise FileNotFoundError(f"Missing semantic mask for {split}/{stem}")
            yield {
                "sample_id": f"{split}/{stem}",
                "split": split,
                "image": image_path.read_bytes() if inline_images else str(image_path),
                "annotations": {
                    "semantic_mask": semantic_path.read_bytes(),
                    "contact_mask": _contact_png_bytes(semantic_path),
                },
            }


def _download_egohos(*, inline_images: bool) -> Iterator[dict[str, object]]:
    import boto3  # ty: ignore[unresolved-import]

    # Path-backed assets are materialized after the generator is exhausted, so
    # extracted files must remain available for the worker process lifetime.
    temp_dir = Path(tempfile.mkdtemp(prefix="egohos-"))
    archive_path = temp_dir / "egohos_dataset.zip"
    boto3.client("s3").download_file(SOURCE_BUCKET, SOURCE_KEY, str(archive_path))
    extracted_dir = temp_dir / "dataset"
    extracted_dir.mkdir()
    _safe_extract(archive_path, extracted_dir)
    yield from _iter_rows(_dataset_root(extracted_dir), inline_images=inline_images)


def download_egohos_inline(
    _task_rank: int,
    _num_tasks: int,
) -> Iterator[dict[str, object]]:
    yield from _download_egohos(inline_images=True)


def download_egohos_paths(
    _task_rank: int,
    _num_tasks: int,
) -> Iterator[dict[str, object]]:
    yield from _download_egohos(inline_images=False)


def identity(row: mdr.Row) -> mdr.Row:
    return row


def _annotations_field() -> pa.Field:
    mask_metadata = {b"encoding": b"png", b"dtype": b"uint8"}
    return pa.field(
        "annotations",
        pa.struct(
            [
                pa.field(
                    "semantic_mask",
                    pa.large_binary(),
                    nullable=False,
                    metadata={
                        **mask_metadata,
                        b"logical_type": b"semantic_mask",
                        b"label_space": b"egohos.semantic.v1",
                    },
                ),
                pa.field(
                    "contact_mask",
                    pa.large_binary(),
                    nullable=False,
                    metadata={
                        **mask_metadata,
                        b"logical_type": b"binary_contact_mask",
                        b"foreground_value": b"1",
                        b"derived_from": b"semantic_mask",
                        b"derivation": b"egohos_generate_contact_boundary",
                    },
                ),
            ]
        ),
        nullable=False,
    )


def build_pipeline(variant: str) -> mdr.RefinerPipeline:
    inline_images = variant != "lance-files"
    source = download_egohos_inline if inline_images else download_egohos_paths
    image_dtype = (
        mdr.datatype.image_bytes() if inline_images else mdr.datatype.image_path()
    ).with_nullable(False)
    pipeline = (
        mdr.task(source, num_tasks=1)
        .map(
            identity,
            dtypes={
                "sample_id": pa.field("sample_id", pa.string(), nullable=False),
                "split": pa.field("split", pa.string(), nullable=False),
                "image": image_dtype,
                "annotations": _annotations_field(),
            },
        )
        .select("sample_id", "split", "image", "annotations")
    )
    if variant == "lance-inline":
        output = DataFolder(
            f"{OUTPUT_ROOT}/ego-hos-images-inside",
            auto_mkdir=False,
        )
        return pipeline.write_lance_dataset(output, mode="create")
    if variant == "lance-files":
        output = DataFolder(
            f"{OUTPUT_ROOT}/ego-hos-images-single-file",
            auto_mkdir=False,
        )
        return pipeline.write_lance_dataset(
            output,
            mode="create",
            assets=mdr.FileAssetConfig(
                subdir="_assets",
                max_in_flight=32,
                missing_policy="error",
            ),
        )
    if variant == "parquet-inline":
        output = DataFolder(
            f"{OUTPUT_ROOT}/ego-hos-images-inside-parquet",
            auto_mkdir=False,
        )
        return pipeline.write_parquet(output)
    raise ValueError(f"Unknown storage variant: {variant!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "variant",
        choices=("lance-inline", "lance-files", "parquet-inline"),
    )
    variant = parser.parse_args().variant
    build_pipeline(variant).launch_cloud(
        name=f"egohos-{variant}",
        num_workers=1,
        cpus_per_worker=2,
        mem_mb_per_worker=8192,
        dependencies=["boto3>=1.34,<2", "opencv-python-headless>=4.10,<5"],
        secrets=mdr.Secrets.env(
            name="default",
            keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
        ),
    )


if __name__ == "__main__":
    main()
