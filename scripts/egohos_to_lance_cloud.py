from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import tempfile
from urllib.request import urlretrieve
import zipfile

import pyarrow as pa

import refiner as mdr


DATASET_URL = (
    "https://www.modelscope.cn/datasets/OmniData/EgoHOS/resolve/"
    "master/raw/egohos_dataset.zip"
)
OUTPUT_URI = "s3://macrodata-hands-research/data-lake/hand-detection/ego-hos"
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


def _contact_path(split_dir: Path, stem: str) -> Path:
    candidates = (
        split_dir / "contact" / f"{stem}.png",
        split_dir / "label_contact" / f"{stem}.png",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing contact mask for {split_dir.name}/{stem}")


def iter_egohos_rows(dataset_root: Path) -> Iterator[dict[str, object]]:
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
            contact_path = _contact_path(split_dir, stem)
            yield {
                "sample_id": f"{split}/{stem}",
                "split": split,
                "image": str(image_path),
                "annotations": {
                    "semantic_mask": semantic_path.read_bytes(),
                    "contact_mask": contact_path.read_bytes(),
                },
            }


def download_egohos(_task_rank: int, _num_tasks: int) -> Iterator[dict[str, object]]:
    with tempfile.TemporaryDirectory(prefix="egohos-") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        archive_path = temp_dir / "egohos_dataset.zip"
        urlretrieve(DATASET_URL, archive_path)
        extracted_dir = temp_dir / "dataset"
        extracted_dir.mkdir()
        _safe_extract(archive_path, extracted_dir)
        yield from iter_egohos_rows(_dataset_root(extracted_dir))


def identity(row: mdr.Row) -> mdr.Row:
    return row


def build_pipeline() -> mdr.RefinerPipeline:
    mask_metadata = {
        b"encoding": b"png",
        b"dtype": b"uint8",
    }
    annotations = pa.field(
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
                    },
                ),
            ]
        ),
        nullable=False,
    )
    return (
        mdr.task(download_egohos, num_tasks=1)
        .map(
            identity,
            dtypes={
                "sample_id": pa.field("sample_id", pa.string(), nullable=False),
                "split": pa.field("split", pa.string(), nullable=False),
                "image": mdr.datatype.image_path().with_nullable(False),
                "annotations": annotations,
            },
        )
        .select("sample_id", "split", "image", "annotations")
        .write_lance_dataset(
            OUTPUT_URI,
            mode="create",
            assets=mdr.BlobAssetConfig(
                subdir="_assets",
                target_bytes=1 << 30,
                missing_policy="error",
            ),
        )
    )


if __name__ == "__main__":
    build_pipeline().launch_cloud(
        name="egohos-to-lance",
        num_workers=1,
        cpus_per_worker=2,
        mem_mb_per_worker=8192,
        secrets=mdr.Secrets.env(
            name="default",
            keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
        ),
    )
