from __future__ import annotations

import asyncio
import importlib
import math
from typing import Any, cast

import pytest

import refiner as mdr
from refiner.inference import PoolingResponse
from refiner.inference import client as openai_module
from refiner.pipeline.data.row import DictRow
from refiner.robotics import reward as reward_module
from refiner.robotics.lerobot_format import LeRobotMetadata, LeRobotRow
from refiner.services import VLLMRuntimeServiceBinding
from refiner.services.discovery import collect_pipeline_services


def test_reward_score_declares_robometer_vllm_service() -> None:
    pipeline = mdr.from_items([{"text": "hello"}]).map_async(
        mdr.robotics.reward_score(model="aliangdw/Robometer-4B")
    )

    services = collect_pipeline_services(pipeline)

    assert len(services) == 1
    assert services[0].config == {
        "model_name_or_path": "aliangdw/Robometer-4B",
        "config": "correctness",
    }


def test_expected_progress_uses_softmax_over_progress_bin_centers() -> None:
    logits = [0.0] * 13
    logits[9] = 10.0

    assert reward_module.expected_progress(logits) == pytest.approx(
        8.997 / 9.0, abs=0.001
    )


def test_extract_token_logits_supports_openai_style_pooling_data() -> None:
    logits = [[float(index)] * 13 for index in range(2)]

    assert (
        reward_module._extract_token_logits(  # pyright: ignore[reportPrivateUsage]
            {"data": [{"data": logits}]}
        )
        == logits
    )


def test_reward_score_builds_robometer_pooling_request(monkeypatch) -> None:
    seen: dict[str, Any] = {}
    logits = [[0.0] * 13 for _ in range(2)]
    logits[0][3] = 10.0
    logits[0][10] = 0.0
    logits[1][9] = 10.0
    logits[1][10] = 10.0

    async def _fake_sample_video_frames(row, *, video_key, max_frames):
        seen["sample"] = {
            "episode_index": row.episode_index,
            "video_key": video_key,
            "max_frames": max_frames,
        }
        return [object(), object()]

    def _fake_frame_data_url(frame):
        del frame
        return "data:image/png;base64,frame"

    async def _fake_pooling(self, payload):
        seen["payload"] = dict(payload)
        return PoolingResponse(response={"data": [{"data": logits}]})

    class _FakeServiceManager:
        async def get(self, name: str):
            del name
            return VLLMRuntimeServiceBinding(
                name="vllm-test",
                kind="llm",
                endpoint="http://127.0.0.1:8000",
            )

    runtime_module = importlib.import_module("refiner.inference._runtime")

    monkeypatch.setattr(
        reward_module, "_sample_video_frames", _fake_sample_video_frames
    )
    monkeypatch.setattr(reward_module, "_frame_data_url", _fake_frame_data_url)
    monkeypatch.setattr(openai_module._OpenAIEndpointClient, "pooling", _fake_pooling)
    monkeypatch.setattr(
        runtime_module, "get_active_service_manager", lambda: _FakeServiceManager()
    )

    row = LeRobotRow(
        DictRow(
            {
                "episode_index": 7,
                "length": 100,
                "tasks": ["open the drawer"],
                "videos/observation.images.main/from_timestamp": 0.0,
                "videos/observation.images.main/to_timestamp": 1.0,
            }
        ),
        metadata=cast(LeRobotMetadata, None),
        frames=[],
    )
    score = mdr.robotics.reward_score(
        model="aliangdw/Robometer-4B",
        video_key="observation.images.main",
        max_frames=2,
    )

    result = asyncio.run(score(row))

    assert result["robometer_progress"] == pytest.approx(
        [3.0 / 9.0, 8.997 / 9.0], abs=0.001
    )
    assert result["robometer_success"] == pytest.approx(
        [0.5, 1.0 / (1.0 + math.exp(-10.0))]
    )
    assert result["reward_score"] == pytest.approx(result["robometer_progress"])
    assert seen["sample"] == {
        "episode_index": 7,
        "video_key": "observation.images.main",
        "max_frames": 2,
    }
    assert seen["payload"] == {
        "model": "aliangdw/Robometer-4B",
        "task": "token_classify",
        "use_activation": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "open the drawer"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,frame"},
                    },
                    {"type": "text", "text": "<|prog_token|>"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,frame"},
                    },
                    {"type": "text", "text": "<|prog_token|>"},
                ],
            }
        ],
    }
