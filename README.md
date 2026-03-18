# Refiner

<p align="left">
  <img src="https://macrodata.co/logo.svg" alt="Macrodata" width="180">
</p>

Refiner is an open-source engine for training data pipelines. It is built for turning raw, unstructured, and multimodal data into high-quality datasets for large model training, with shard-aware execution, built-in observability, and a seamless path from local runs to managed cloud execution.

Current focus:

- training data, not generic analytics ETL
- multimodal and model-centric processing
- robotics support today, more modalities coming
- integrated platform observability and cloud execution

## Quickstart

Install:

```bash
pip install macrodata-refiner
```

Create a Macrodata API key:

- https://macrodata.co/settings/api-keys

Log in:

```bash
macrodata login
macrodata whoami
```

Launch one pipeline locally:

```python
import refiner as mdr

pipeline = (
    mdr.read_jsonl("input/*.jsonl")
    .filter(lambda row: row["lang"] == "en")
    .map(lambda row: {"text": row["text"].strip()})
    .write_parquet("out/")
)

stats = pipeline.launch_local(
    name="english-cleanup",
    num_workers=2,
)
```

Launch the same pipeline on Macrodata Cloud:

```python
import refiner as mdr

pipeline = (
    mdr.read_jsonl("input/*.jsonl")
    .filter(lambda row: row["lang"] == "en")
    .map(lambda row: {"text": row["text"].strip()})
    .write_parquet("hf://datasets/macrodata/my-output")
)

result = pipeline.launch_cloud(
    name="english-cleanup",
    num_workers=4,
    cpus_per_worker=2,
    mem_mb_per_worker=4096,
)
```

`pip install` gives you:

- the Python package as `refiner`
- the CLI as `macrodata`

## Technical Highlights

- **Shard-aware runtime.**
  Readers plan shards explicitly, workers claim them through the runtime lifecycle, heartbeat them, and complete or fail them with durable state.
- **Fused execution.**
  Adjacent Python row steps and Arrow-backed vectorized segments are compiled into a tighter execution plan instead of forcing a materialization boundary at every transform.
- **Structured cloud submission.**
  Launchers compile a structured plan, serialize pipeline payloads, and attach a manifest with script text, dependency inventory, and ref/version metadata.
- **Secret-aware code capture.**
  Captured code and script text are redacted before submission so cloud introspection stays useful without leaking secret values.
- **Built-in observability.**
  Jobs, stages, workers, shards, logs, metrics, and manifests are all part of the platform path already.
- **Multistage training-data sinks.**
  Refiner already includes specialized paths like the LeRobot writer stages for robotics datasets.

## Docs

Start here:

- [Docs home](docs/index.md)
- [Pipeline basics](docs/pipeline-basics.md)
- [Launchers](docs/launchers.md)
- [CLI auth](docs/cli-auth.md)

Reference:

- [Local execution](docs/local-execution.md)
- [Readers and sharding](docs/readers-and-sharding.md)
- [Expression transforms](docs/expression-transforms.md)
- [Worker runtime](docs/worker-runtime.md)
- [Observability](docs/observability.md)
