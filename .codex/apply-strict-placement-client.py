from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file_path = Path(path)
    text = file_path.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement target, found {count}")
    file_path.write_text(text.replace(old, new, 1))


def replace_count(path: str, old: str, new: str, expected: int) -> None:
    file_path = Path(path)
    text = file_path.read_text()
    count = text.count(old)
    if count != expected:
        raise RuntimeError(f"{path}: expected {expected} replacement targets, found {count}")
    file_path.write_text(text.replace(old, new))


replace_once(
    "src/refiner/platform/client/models.py",
    'CloudProvider = Literal["aws", "oci", "gcp"]\n',
    'CloudPlacementMode = Literal["best_effort", "strict"]\nCloudProvider = Literal["aws", "oci", "gcp"]\n',
)
replace_once(
    "src/refiner/platform/client/models.py",
    '''    cloud: CloudProvider = "aws"\n    region: tuple[CloudRegion, ...] = ("us", "eu", "ca")\n''',
    '''    cloud: CloudProvider = "aws"\n    region: tuple[CloudRegion, ...] = ("us", "eu", "ca")\n    placement_mode: CloudPlacementMode = "best_effort"\n''',
)
replace_once(
    "src/refiner/platform/client/models.py",
    '''            "cloud": self.cloud,\n            "region": list(self.region),\n''',
    '''            "cloud": self.cloud,\n            "region": list(self.region),\n            "placement_mode": self.placement_mode,\n''',
)

replace_once(
    "src/refiner/platform/client/__init__.py",
    '''    CloudRuntimeConfig,\n    CloudProvider,\n''',
    '''    CloudRuntimeConfig,\n    CloudPlacementMode,\n    CloudProvider,\n''',
)
replace_once(
    "src/refiner/platform/client/__init__.py",
    '''    "CloudRuntimeConfig",\n    "CloudProvider",\n''',
    '''    "CloudRuntimeConfig",\n    "CloudPlacementMode",\n    "CloudProvider",\n''',
)

replace_once(
    "src/refiner/launchers/cloud.py",
    '''    CloudRuntimeConfig,\n    CloudProvider,\n''',
    '''    CloudRuntimeConfig,\n    CloudPlacementMode,\n    CloudProvider,\n''',
)
replace_once(
    "src/refiner/launchers/cloud.py",
    '''_SUPPORTED_CLOUDS = frozenset({"aws", "oci", "gcp"})\n_SUPPORTED_REGIONS = frozenset(\n''',
    '''_SUPPORTED_CLOUDS = frozenset({"aws", "oci", "gcp"})\n_SUPPORTED_PLACEMENT_MODES = frozenset({"best_effort", "strict"})\n_SUPPORTED_REGIONS = frozenset(\n''',
)
replace_once(
    "src/refiner/launchers/cloud.py",
    '''def _normalize_regions(value: str | Sequence[str]) -> tuple[CloudRegion, ...]:\n    values = (value,) if isinstance(value, str) else tuple(value)\n    if not values:\n        raise ValueError("region must contain at least one selector")\n    invalid = sorted({item for item in values if item not in _SUPPORTED_REGIONS})\n    if invalid:\n        supported = ", ".join(sorted(_SUPPORTED_REGIONS))\n        raise ValueError(\n            f"unsupported region selector(s): {', '.join(invalid)}; "\n            f"expected one or more of: {supported}"\n        )\n    return cast(tuple[CloudRegion, ...], tuple(dict.fromkeys(values)))\n\n\n''',
    '''def _normalize_regions(value: str | Sequence[str]) -> tuple[CloudRegion, ...]:\n    values = (value,) if isinstance(value, str) else tuple(value)\n    if not values:\n        raise ValueError("region must contain at least one selector")\n    invalid = sorted({item for item in values if item not in _SUPPORTED_REGIONS})\n    if invalid:\n        supported = ", ".join(sorted(_SUPPORTED_REGIONS))\n        raise ValueError(\n            f"unsupported region selector(s): {', '.join(invalid)}; "\n            f"expected one or more of: {supported}"\n        )\n    return cast(tuple[CloudRegion, ...], tuple(dict.fromkeys(values)))\n\n\ndef _normalize_placement_mode(value: str) -> CloudPlacementMode:\n    if value not in _SUPPORTED_PLACEMENT_MODES:\n        supported = ", ".join(sorted(_SUPPORTED_PLACEMENT_MODES))\n        raise ValueError(f"placement_mode must be one of: {supported}")\n    return cast(CloudPlacementMode, value)\n\n\n''',
)
replace_once(
    "src/refiner/launchers/cloud.py",
    '''        gpu: Optional GPU runtime request for cloud scheduling.\n        sync_local_dependencies: Whether to include packages detected from the\n''',
    '''        gpu: Optional GPU runtime request for cloud scheduling.\n        cloud: Public cloud provider used for worker placement.\n        region: Accepted region selectors for worker placement.\n        placement_mode: ``"best_effort"`` validates the actual worker placement;\n            ``"strict"`` also requests the native provider region constraint.\n        sync_local_dependencies: Whether to include packages detected from the\n''',
)
replace_once(
    "src/refiner/launchers/cloud.py",
    '''        cloud: CloudProvider = "aws",\n        region: CloudRegion | Sequence[CloudRegion] = ("us", "eu", "ca"),\n        sync_local_dependencies: bool = False,\n''',
    '''        cloud: CloudProvider = "aws",\n        region: CloudRegion | Sequence[CloudRegion] = ("us", "eu", "ca"),\n        placement_mode: CloudPlacementMode = "best_effort",\n        sync_local_dependencies: bool = False,\n''',
)
replace_once(
    "src/refiner/launchers/cloud.py",
    '''        self.cloud = _normalize_cloud(cloud)\n        self.region = _normalize_regions(region)\n        self.sync_local_dependencies = sync_local_dependencies\n''',
    '''        self.cloud = _normalize_cloud(cloud)\n        self.region = _normalize_regions(region)\n        self.placement_mode = _normalize_placement_mode(placement_mode)\n        self.sync_local_dependencies = sync_local_dependencies\n''',
)
replace_once(
    "src/refiner/launchers/cloud.py",
    '''                            cloud=self.cloud,\n                            region=self.region,\n                            cpus_per_worker=stage.compute.cpus_per_worker,\n''',
    '''                            cloud=self.cloud,\n                            region=self.region,\n                            placement_mode=self.placement_mode,\n                            cpus_per_worker=stage.compute.cpus_per_worker,\n''',
)

replace_once(
    "src/refiner/pipeline/pipeline.py",
    '''    from refiner.platform.client import CloudProvider, CloudRegion\n''',
    '''    from refiner.platform.client import (\n        CloudPlacementMode,\n        CloudProvider,\n        CloudRegion,\n    )\n''',
)
replace_once(
    "src/refiner/pipeline/pipeline.py",
    '''        cloud: CloudProvider = "aws",\n        region: CloudRegion | Sequence[CloudRegion] = ("us", "eu", "ca"),\n        sync_local_dependencies: bool = False,\n''',
    '''        cloud: CloudProvider = "aws",\n        region: CloudRegion | Sequence[CloudRegion] = ("us", "eu", "ca"),\n        placement_mode: CloudPlacementMode = "best_effort",\n        sync_local_dependencies: bool = False,\n''',
)
replace_once(
    "src/refiner/pipeline/pipeline.py",
    '''            region: One region selector or a sequence of selectors. Workers are\n                accepted when their placement matches any selector. This does\n                not request a priced Modal region constraint.\n            sync_local_dependencies: Include packages detected from the local\n''',
    '''            region: One region selector or a sequence of selectors. Workers are\n                accepted when their placement matches any selector.\n            placement_mode: ``"best_effort"`` lets the provider choose placement\n                and rejects workers that land outside the selectors. ``"strict"``\n                also requests the native provider region constraint and fails\n                instead of silently spilling to a different region.\n            sync_local_dependencies: Include packages detected from the local\n''',
)
replace_once(
    "src/refiner/pipeline/pipeline.py",
    '''            cloud=cloud,\n            region=region,\n            sync_local_dependencies=sync_local_dependencies,\n''',
    '''            cloud=cloud,\n            region=region,\n            placement_mode=placement_mode,\n            sync_local_dependencies=sync_local_dependencies,\n''',
)

replace_once(
    "tests/platform/test_cloud_client.py",
    '''    assert runtime.cloud == "aws"\n    assert runtime.region == ("us", "eu", "ca")\n''',
    '''    assert runtime.cloud == "aws"\n    assert runtime.region == ("us", "eu", "ca")\n    assert runtime.placement_mode == "best_effort"\n''',
)
replace_count(
    "tests/platform/test_cloud_client.py",
    '''                "cloud": "aws",\n                "region": ["us", "eu", "ca"],\n''',
    '''                "cloud": "aws",\n                "region": ["us", "eu", "ca"],\n                "placement_mode": "best_effort",\n''',
    2,
)

replace_once(
    "tests/launchers/test_cloud_launcher.py",
    '''    assert request.stage_payloads[0].runtime.cloud == "aws"\n    assert request.stage_payloads[0].runtime.region == ("us", "eu", "ca")\n''',
    '''    assert request.stage_payloads[0].runtime.cloud == "aws"\n    assert request.stage_payloads[0].runtime.region == ("us", "eu", "ca")\n    assert request.stage_payloads[0].runtime.placement_mode == "best_effort"\n''',
)
replace_once(
    "tests/launchers/test_cloud_launcher.py",
    '''    read_jsonl("input.jsonl").launch_cloud(\n        name="placed cloud", cloud="gcp", region=["uk", "us-west"]\n    )\n''',
    '''    read_jsonl("input.jsonl").launch_cloud(\n        name="placed cloud",\n        cloud="gcp",\n        region=["uk", "us-west"],\n        placement_mode="strict",\n    )\n''',
)
replace_once(
    "tests/launchers/test_cloud_launcher.py",
    '''    assert runtime.cloud == "gcp"\n    assert runtime.region == ("uk", "us-west")\n\n\n@pytest.mark.parametrize("cloud", ["auto", "azure", "AWS"])\n''',
    '''    assert runtime.cloud == "gcp"\n    assert runtime.region == ("uk", "us-west")\n    assert runtime.placement_mode == "strict"\n\n\n@pytest.mark.parametrize("placement_mode", ["native", "spill", "STRICT"])\ndef test_pipeline_launch_cloud_rejects_invalid_placement_mode(\n    monkeypatch, placement_mode\n) -> None:\n    _stub_cloud_submit(monkeypatch, fail_on_submit=True)\n\n    with pytest.raises(ValueError, match="placement_mode must be one of"):\n        read_jsonl("input.jsonl").launch_cloud(\n            name="demo cloud", placement_mode=placement_mode\n        )\n\n\n@pytest.mark.parametrize("cloud", ["auto", "azure", "AWS"])\n''',
)

replace_once(
    "docs/running-pipelines/cloud-launcher.md",
    '''Broad selectors are `us`, `eu`, `ca`, and `uk`. Narrow selectors are\n`us-east`, `us-central`, `us-south`, `us-west`, `eu-west`, `eu-north`, and\n`eu-south`. `eu` excludes the UK. Madrid is classified as `eu-south`; the\ndefensive `FRA*` and `AMS` aliases are classified as `eu-west`.\n\n## What gets submitted\n''',
    '''Broad selectors are `us`, `eu`, `ca`, and `uk`. Narrow selectors are\n`us-east`, `us-central`, `us-south`, `us-west`, `eu-west`, `eu-north`, and\n`eu-south`. `eu` excludes the UK. Madrid is classified as `eu-south`; the\ndefensive `FRA*` and `AMS` aliases are classified as `eu-west`.\n\nThe default `placement_mode="best_effort"` lets the provider choose a region,\nthen rejects and safely retries a worker that lands outside the requested\nselectors. Use strict placement when a job must not spill to another region:\n\n```python\npipeline.launch_cloud(\n    name="strict-eu-workers",\n    cloud="aws",\n    region=["eu-west", "uk"],\n    placement_mode="strict",\n)\n```\n\nStrict placement also sends the region list as a native provider placement\nconstraint. If the provider cannot satisfy it, the job remains fail-closed\ninstead of silently running elsewhere.\n\n## What gets submitted\n''',
)
