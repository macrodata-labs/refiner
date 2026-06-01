from __future__ import annotations

import asyncio
import importlib
import math
from typing import Any, cast

import pytest

import refiner as mdr
from refiner.inference.providers import openai as openai_provider
from refiner.pipeline.data.row import DictRow
from refiner.robotics import reward as reward_module
from refiner.robotics.lerobot_format import LeRobotMetadata, LeRobotRow
from refiner.services import VLLMRuntimeServiceBinding
from refiner.services.discovery import collect_pipeline_services


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
        return {"data": [{"data": logits}]}

    class _FakeServiceManager:
        async def get(self, name: str):
            del name
            return VLLMRuntimeServiceBinding(
                name="vllm-test",
                kind="llm",
                endpoint="http://127.0.0.1:8000",
            )

    runtime_module = importlib.import_module("refiner.inference.internal.runtime")

    monkeypatch.setattr(
        reward_module, "_sample_video_frames", _fake_sample_video_frames
    )
    monkeypatch.setattr(reward_module, "_frame_data_url", _fake_frame_data_url)
    monkeypatch.setattr(openai_provider._OpenAIEndpointClient, "pooling", _fake_pooling)
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
    services = collect_pipeline_services(mdr.from_items([{}]).map_async(score))

    result = asyncio.run(score(row))

    assert services[0].config == {
        "model_name_or_path": "aliangdw/Robometer-4B",
        "config": "throughput",
    }
    assert "robometer_progress" not in result
    assert result["reward_score"] == pytest.approx([3.0 / 9.0, 8.997 / 9.0], abs=0.001)
    assert result["robometer_success"] == pytest.approx(
        [0.5, 1.0 / (1.0 + math.exp(-10.0))]
    )
    assert seen["sample"] == {
        "episode_index": 7,
        "video_key": "observation.images.main",
        "max_frames": 2,
    }
    assert seen["payload"] == {
        "model": "aliangdw/Robometer-4B",
        "task": "token_classify",
        "use_activation": False,
        "chat_template_kwargs": {
            "add_vision_id": True,
            "enable_thinking": False,
            "fps": 1,
        },
        "mm_processor_kwargs": {"do_resize": False},
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "The task for the robot is 'open the drawer'. Given "
                            "the trajectory video, predict the task progress at "
                            "each frame, how far along the robot is towards "
                            "completing the task, a float between 0 and 1, "
                            "where 0 is the starting state and 1 is when the "
                            "task is completed. If the robot is not performing "
                            "the same task, predict 0 progress."
                        ),
                    },
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
