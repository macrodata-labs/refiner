from __future__ import annotations

import asyncio
import importlib
import json
from typing import Any

import pytest

import refiner as mdr
from refiner.inference.internal.transport import APIResponse
from refiner.pipeline.data.row import DictRow

count_prior_module = importlib.import_module(
    "refiner.robotics.subtask_annotation.count_prior"
)


def test_domain_profiles_pin_policy_gold_and_model() -> None:
    assert mdr.robotics.WALDEN_V1.domain_id == "walden"
    assert mdr.robotics.WALDEN_V1.gold_set == "walden-97-operator-v1"
    assert mdr.robotics.WALDEN_V1.model_artifact == (
        "vjepa2-vit-l-tridet-walden-epoch-019"
    )
    assert len(mdr.robotics.WALDEN_V1.profile_hash) == 64
    assert mdr.robotics.ASSEMBLY_V1.domain_id == "assembly"
    assert mdr.robotics.ASSEMBLY_V1.gold_set == "assembly-40x5-consensus-v1"
    assert mdr.robotics.ASSEMBLY_V1.model_artifact is None


def test_count_prior_from_precomputed_segments() -> None:
    block = mdr.robotics.count_prior_from_segments(
        profile=mdr.robotics.WALDEN_V1,
        segments_column="partitioner",
    )

    row = block(
        DictRow(
            {
                "partitioner": [
                    {"start_s": 2, "end_s": 3, "score": 0.7},
                    {"start_s": 0, "end_s": 1, "score": 0.9},
                ]
            }
        )
    )

    assert row["partitioner_segment_count"] == 2
    assert row["partitioner_count_result"]["status"] == "ok"
    assert row["partitioner_count_result"]["segments"][0] == {
        "start_sec": 0.0,
        "end_sec": 1.0,
        "score": 0.9,
    }
    builtin = getattr(block, "__refiner_builtin_call__")
    assert builtin["args"]["profile_hash"] == mdr.robotics.WALDEN_V1.profile_hash


def test_count_prior_rejects_profile_without_domain_model() -> None:
    with pytest.raises(ValueError, match="model_artifact"):
        mdr.robotics.count_prior_from_segments(
            profile=mdr.robotics.ASSEMBLY_V1,
            segments_column="partitioner",
        )


def test_partitioner_count_prior_calls_learned_service(monkeypatch) -> None:
    monkeypatch.setenv("TEST_PARTITIONER_TOKEN", "test-token")
    seen: dict[str, Any] = {}

    async def _fake_post(client, path, payload, **kwargs):
        seen.update(
            {
                "base_url": client.base_url,
                "headers": dict(client.headers),
                "path": path,
                "payload": payload,
                **kwargs,
            }
        )
        return APIResponse(
            value={
                "status": "ok",
                "segments": [
                    {"start_s": 0.1, "end_s": 1.2, "score": 0.8},
                    {"start_s": 1.2, "end_s": 2.0, "score": 0.6},
                ],
            },
            response_headers={},
        )

    monkeypatch.setattr(count_prior_module, "post_json_to_api", _fake_post)
    block = mdr.robotics.partitioner_count_prior(
        profile=mdr.robotics.WALDEN_V1,
        endpoint="https://partitioner.example/segment",
        video_url_column="video_url",
        token_env="TEST_PARTITIONER_TOKEN",
    )

    async def _run():
        try:
            return await block(
                DictRow(
                    {
                        "video_url": "https://videos.example/episode.mp4?signature=x",
                        "data_hash": "episode-hash",
                    }
                )
            )
        finally:
            await block.aclose()

    row = asyncio.run(_run())

    assert row["partitioner_segment_count"] == 2
    assert row["partitioner_count_result"]["model_artifact"].endswith("epoch-019")
    assert seen["base_url"] == "https://partitioner.example"
    assert seen["path"] == "segment"
    assert seen["payload"]["data_hash"] == "episode-hash"
    assert seen["headers"] == {"Authorization": "Bearer test-token"}


def test_benchmark_runner_reports_edit_cost_and_paired_bootstrap(tmp_path) -> None:
    gold = [
        {
            "video_id": "a",
            "duration_s": 60,
            "segments": [
                {"start_sec": 0, "end_sec": 10},
                {"start_sec": 10, "end_sec": 20},
            ],
        },
        {
            "video_id": "b",
            "duration_s": 30,
            "segments": [{"start_sec": 0, "end_sec": 10}],
        },
    ]
    baseline = [
        {
            "video_id": "a",
            "segments": [{"start_sec": 0, "end_sec": 20}],
        },
        {"video_id": "b", "segments": []},
    ]
    candidate = [
        {"video_id": "a", "segments": gold[0]["segments"]},
        {"video_id": "b", "segments": gold[1]["segments"]},
    ]
    paths = {}
    for name, value in (
        ("gold", gold),
        ("current", baseline),
        ("candidate", candidate),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path

    report_path = tmp_path / "report.json"
    report = mdr.robotics.run_segmentation_benchmark(
        gold_path=paths["gold"],
        candidate_paths={
            "current": paths["current"],
            "candidate": paths["candidate"],
        },
        baseline="current",
        output_path=report_path,
        bootstrap_samples=100,
        seed=4,
    )

    assert report["candidates"]["candidate"]["aggregate"]["f1"] == 1.0
    comparison = report["paired_vs_baseline"]["candidate"]
    assert comparison["edit_cost_per_min"]["mean_improvement"] > 0
    assert comparison["f1"]["probability_improved"] == 1.0
    assert len(report["input_sha256"]["gold"]) == 64
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
