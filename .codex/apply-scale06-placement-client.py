from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement target, found {count}")
    target.write_text(text.replace(old, new, 1))


def append_once(path: str, marker: str, block: str) -> None:
    target = Path(path)
    text = target.read_text()
    if marker in text:
        raise RuntimeError(f"{path}: placement client block already present")
    if not text.endswith("\n"):
        text += "\n"
    target.write_text(text + block)


replace_once(
    "src/refiner/platform/client/models.py",
    "from typing import Any\n",
    "from typing import Any, Literal\n",
)
replace_once(
    "src/refiner/platform/client/models.py",
    """from refiner.worker.lifecycle import FinalizedShardWorker


class WorkspaceIdentity""",
    """from refiner.worker.lifecycle import FinalizedShardWorker

CloudPlacementMode = Literal["best_effort", "strict"]
CloudProvider = Literal["aws", "oci", "gcp"]
CloudRegion = Literal[
    "us",
    "eu",
    "ca",
    "uk",
    "us-east",
    "us-central",
    "us-south",
    "us-west",
    "eu-west",
    "eu-north",
    "eu-south",
]


class WorkspaceIdentity""",
)
replace_once(
    "src/refiner/platform/client/models.py",
    """@dataclass(frozen=True, slots=True)
class CloudRuntimeConfig:
    num_workers: int
    cpus_per_worker: int | None = None
    mem_mb_per_worker: int | None = None
    gpu: GPU | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "num_workers": self.num_workers,
        }
""",
    """@dataclass(frozen=True, slots=True)
class CloudRuntimeConfig:
    num_workers: int
    cpus_per_worker: int | None = None
    mem_mb_per_worker: int | None = None
    gpu: GPU | None = None
    cloud: CloudProvider = "aws"
    region: tuple[CloudRegion, ...] = ("us", "eu", "ca")
    placement_mode: CloudPlacementMode = "best_effort"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "num_workers": self.num_workers,
            "cloud": self.cloud,
            "region": list(self.region),
            "placement_mode": self.placement_mode,
        }
""",
)

replace_once(
    "src/refiner/platform/client/__init__.py",
    """    CloudRunCreateResponse,
    CloudRuntimeConfig,
    CreateJobResponse,
""",
    """    CloudRunCreateResponse,
    CloudRuntimeConfig,
    CloudPlacementMode,
    CloudProvider,
    CloudRegion,
    CreateJobResponse,
""",
)
replace_once(
    "src/refiner/platform/client/__init__.py",
    """    "CloudRunCreateResponse",
    "CloudRuntimeConfig",
    "CreateJobResponse",
""",
    """    "CloudRunCreateResponse",
    "CloudRuntimeConfig",
    "CloudPlacementMode",
    "CloudProvider",
    "CloudRegion",
    "CreateJobResponse",
""",
)

replace_once(
    "src/refiner/launchers/cloud.py",
    """    CloudRunCreateRequest,
    CloudRuntimeConfig,
    MacrodataApiError,
""",
    """    CloudRunCreateRequest,
    CloudRuntimeConfig,
    CloudPlacementMode,
    CloudProvider,
    CloudRegion,
    MacrodataApiError,
""",
)
replace_once(
    "src/refiner/launchers/cloud.py",
    """_FALLBACK_ENV_VAR = "MACRODATA_FALLBACK_TO_LATEST_PYPI"
_CLOUD_FILE_BATCH_SIZE = 100
_UUID_PATTERN""",
    """_FALLBACK_ENV_VAR = "MACRODATA_FALLBACK_TO_LATEST_PYPI"
_CLOUD_FILE_BATCH_SIZE = 100
_SUPPORTED_CLOUDS = frozenset({"aws", "oci", "gcp"})
_SUPPORTED_PLACEMENT_MODES = frozenset({"best_effort", "strict"})
_SUPPORTED_REGIONS = frozenset(
    {
        "us",
        "eu",
        "ca",
        "uk",
        "us-east",
        "us-central",
        "us-south",
        "us-west",
        "eu-west",
        "eu-north",
        "eu-south",
    }
)
_UUID_PATTERN""",
)
replace_once(
    "src/refiner/launchers/cloud.py",
    """    return f"{normalized_job_id}:{stage_index}"


@dataclass(frozen=True, slots=True)
""",
    """    return f"{normalized_job_id}:{stage_index}"


def _normalize_cloud(value: str) -> CloudProvider:
    if value not in _SUPPORTED_CLOUDS:
        supported = ", ".join(sorted(_SUPPORTED_CLOUDS))
        raise ValueError(f"cloud must be one of: {supported}")
    return cast(CloudProvider, value)


def _normalize_regions(value: str | Sequence[str]) -> tuple[CloudRegion, ...]:
    values = (value,) if isinstance(value, str) else tuple(value)
    if not values:
        raise ValueError("region must contain at least one selector")
    invalid = sorted({item for item in values if item not in _SUPPORTED_REGIONS})
    if invalid:
        supported = ", ".join(sorted(_SUPPORTED_REGIONS))
        raise ValueError(
            f"unsupported region selector(s): {', '.join(invalid)}; "
            f"expected one or more of: {supported}"
        )
    return cast(tuple[CloudRegion, ...], tuple(dict.fromkeys(values)))


def _normalize_placement_mode(value: str) -> CloudPlacementMode:
    if value not in _SUPPORTED_PLACEMENT_MODES:
        supported = ", ".join(sorted(_SUPPORTED_PLACEMENT_MODES))
        raise ValueError(f"placement_mode must be one of: {supported}")
    return cast(CloudPlacementMode, value)


@dataclass(frozen=True, slots=True)
""",
)
replace_once(
    "src/refiner/launchers/cloud.py",
    """        gpu: Optional GPU runtime request for cloud scheduling.
        sync_local_dependencies: Whether to include packages detected from the
""",
    """        gpu: Optional GPU runtime request for cloud scheduling.
        cloud: Public cloud provider used for worker placement.
        region: Accepted region selectors for worker placement.
        placement_mode: ``"best_effort"`` validates the actual worker placement;
            ``"strict"`` also requests the native provider region constraint.
        sync_local_dependencies: Whether to include packages detected from the
""",
)
replace_once(
    "src/refiner/launchers/cloud.py",
    """        gpu: GPU | None = None,
        sync_local_dependencies: bool = False,
""",
    """        gpu: GPU | None = None,
        cloud: CloudProvider = "aws",
        region: CloudRegion | Sequence[CloudRegion] = ("us", "eu", "ca"),
        placement_mode: CloudPlacementMode = "best_effort",
        sync_local_dependencies: bool = False,
""",
)
replace_once(
    "src/refiner/launchers/cloud.py",
    """        self.cpus_per_worker = cpus_per_worker
        self.mem_mb_per_worker = mem_mb_per_worker
        self.sync_local_dependencies = sync_local_dependencies
""",
    """        self.cpus_per_worker = cpus_per_worker
        self.mem_mb_per_worker = mem_mb_per_worker
        self.cloud = _normalize_cloud(cloud)
        self.region = _normalize_regions(region)
        self.placement_mode = _normalize_placement_mode(placement_mode)
        self.sync_local_dependencies = sync_local_dependencies
""",
)
replace_once(
    "src/refiner/launchers/cloud.py",
    """                            gpu=stage.compute.gpu,
                        ),
""",
    """                            gpu=stage.compute.gpu,
                            cloud=self.cloud,
                            region=self.region,
                            placement_mode=self.placement_mode,
                        ),
""",
)

replace_once(
    "src/refiner/pipeline/pipeline.py",
    """    from refiner.launchers.cloud import CloudLaunchResult
    from refiner.launchers.local import LaunchStats
    from refiner.launchers.secrets import SecretInput
""",
    """    from refiner.launchers.cloud import CloudLaunchResult
    from refiner.launchers.local import LaunchStats
    from refiner.launchers.secrets import SecretInput
    from refiner.platform.client import (
        CloudPlacementMode,
        CloudProvider,
        CloudRegion,
    )
""",
)
replace_once(
    "src/refiner/pipeline/pipeline.py",
    """        gpu: GPU | None = None,
        sync_local_dependencies: bool = False,
""",
    """        gpu: GPU | None = None,
        cloud: CloudProvider = "aws",
        region: CloudRegion | Sequence[CloudRegion] = ("us", "eu", "ca"),
        placement_mode: CloudPlacementMode = "best_effort",
        sync_local_dependencies: bool = False,
""",
)
replace_once(
    "src/refiner/pipeline/pipeline.py",
    """            gpu: Optional structured GPU request.
            sync_local_dependencies: Include packages detected from the local
""",
    """            gpu: Optional structured GPU request.
            cloud: Public cloud provider. Supported values are ``"aws"``,
                ``"oci"``, and ``"gcp"``.
            region: One region selector or a sequence of selectors. Workers are
                accepted when their placement matches any selector.
            placement_mode: ``"best_effort"`` lets the provider choose placement
                and rejects workers that land outside the selectors. ``"strict"``
                also requests the native provider region constraint and fails
                instead of silently spilling to a different region.
            sync_local_dependencies: Include packages detected from the local
""",
)
replace_once(
    "src/refiner/pipeline/pipeline.py",
    """            gpu=gpu,
            sync_local_dependencies=sync_local_dependencies,
""",
    """            gpu=gpu,
            cloud=cloud,
            region=region,
            placement_mode=placement_mode,
            sync_local_dependencies=sync_local_dependencies,
""",
)

replace_once(
    "docs/running-pipelines/cloud-launcher.md",
    """## What gets submitted
""",
    """## Cloud and region placement

Workers use AWS by default. Select one supported public cloud with `cloud`:

```python
pipeline.launch_cloud(
    name="gcp-workers",
    cloud="gcp",  # "aws", "oci", or "gcp"
)
```

By default, workers may run in the US, EEA, or Canada. Pass one selector or a
list; a worker is accepted when it matches any selector:

```python
pipeline.launch_cloud(
    name="north-america-workers",
    cloud="aws",
    region=["us-east", "ca"],
)
```

Broad selectors are `us`, `eu`, `ca`, and `uk`. Narrow selectors are
`us-east`, `us-central`, `us-south`, `us-west`, `eu-west`, `eu-north`, and
`eu-south`. `eu` excludes the UK. Madrid is classified as `eu-south`; the
defensive `FRA*` and `AMS` aliases are classified as `eu-west`.

The default `placement_mode="best_effort"` lets the provider choose a region,
then rejects and safely retries a worker that lands outside the requested
selectors. Use strict placement when a job must not spill to another region:

```python
pipeline.launch_cloud(
    name="strict-eu-workers",
    cloud="aws",
    region=["eu-west", "uk"],
    placement_mode="strict",
)
```

Strict placement also sends the region list as a native provider placement
constraint. If the provider cannot satisfy it, the job remains fail-closed
instead of silently running elsewhere.

## What gets submitted
""",
)

replace_once(
    "tests/launchers/test_cloud_launcher.py",
    """    assert request.stage_payloads[0].runtime.num_workers == 3
    assert request.stage_payloads[0].runtime.cpus_per_worker == 2
""",
    """    assert request.stage_payloads[0].runtime.num_workers == 3
    assert request.stage_payloads[0].runtime.cloud == "aws"
    assert request.stage_payloads[0].runtime.region == ("us", "eu", "ca")
    assert request.stage_payloads[0].runtime.placement_mode == "best_effort"
    assert request.stage_payloads[0].runtime.cpus_per_worker == 2
""",
)
append_once(
    "tests/launchers/test_cloud_launcher.py",
    "test_pipeline_launch_cloud_forwards_strict_placement",
    """


def test_pipeline_launch_cloud_forwards_strict_placement(monkeypatch) -> None:
    captured = _stub_cloud_submit(monkeypatch)

    read_jsonl("input.jsonl").launch_cloud(
        name="placed cloud",
        cloud="gcp",
        region=["uk", "us-west"],
        placement_mode="strict",
    )

    request = cast(CloudRunCreateRequest, captured["submit_request"])
    runtime = request.stage_payloads[0].runtime
    assert runtime is not None
    assert runtime.cloud == "gcp"
    assert runtime.region == ("uk", "us-west")
    assert runtime.placement_mode == "strict"


@pytest.mark.parametrize("placement_mode", ["native", "spill", "STRICT"])
def test_pipeline_launch_cloud_rejects_invalid_placement_mode(
    monkeypatch, placement_mode
) -> None:
    _stub_cloud_submit(monkeypatch, fail_on_submit=True)

    with pytest.raises(ValueError, match="placement_mode must be one of"):
        read_jsonl("input.jsonl").launch_cloud(
            name="demo cloud", placement_mode=placement_mode
        )


@pytest.mark.parametrize("cloud", ["auto", "azure", "AWS"])
def test_pipeline_launch_cloud_rejects_invalid_cloud(monkeypatch, cloud) -> None:
    _stub_cloud_submit(monkeypatch, fail_on_submit=True)

    with pytest.raises(ValueError, match="cloud must be one of"):
        read_jsonl("input.jsonl").launch_cloud(name="demo cloud", cloud=cloud)


@pytest.mark.parametrize("region", [[], ["ap-south"], ["us", "moon"]])
def test_pipeline_launch_cloud_rejects_invalid_regions(monkeypatch, region) -> None:
    _stub_cloud_submit(monkeypatch, fail_on_submit=True)

    with pytest.raises(ValueError, match="region"):
        read_jsonl("input.jsonl").launch_cloud(name="demo cloud", region=region)
""",
)

replace_once(
    "tests/platform/test_cloud_client.py",
    """

def test_cloud_client_cloud_submit_job_posts_to_cloud_runs(monkeypatch) -> None:
""",
    """

def test_cloud_runtime_config_preserves_positional_resource_order() -> None:
    gpu = GPU(count=2, type="h100", cuda_version="12.4")

    runtime = CloudRuntimeConfig(2, 4, 8192, gpu)

    assert runtime.cpus_per_worker == 4
    assert runtime.mem_mb_per_worker == 8192
    assert runtime.gpu is gpu
    assert runtime.cloud == "aws"
    assert runtime.region == ("us", "eu", "ca")
    assert runtime.placement_mode == "best_effort"


def test_cloud_client_cloud_submit_job_posts_to_cloud_runs(monkeypatch) -> None:
""",
)
replace_once(
    "tests/platform/test_cloud_client.py",
    """            "runtime": {
                "num_workers": 2,
                "cpus_per_worker": 4,
""",
    """            "runtime": {
                "num_workers": 2,
                "cloud": "aws",
                "region": ["us", "eu", "ca"],
                "placement_mode": "best_effort",
                "cpus_per_worker": 4,
""",
)
replace_once(
    "tests/platform/test_cloud_client.py",
    """            "runtime": {"num_workers": 1},
""",
    """            "runtime": {
                "num_workers": 1,
                "cloud": "aws",
                "region": ["us", "eu", "ca"],
                "placement_mode": "best_effort",
            },
""",
)
