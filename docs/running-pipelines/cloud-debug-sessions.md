---
title: "Cloud debug sessions"
description: "Iterate on a pipeline inside one retained cloud worker"
---

# Cloud debug sessions

Use a cloud debug session when a pipeline needs the cloud runtime, dependencies,
data access, CPU, or GPU, but you want to validate one worker before scaling out.

```python
import refiner as mdr

pipeline = mdr.read_jsonl("s3://datasets/input.jsonl")
pipeline.launch_cloud(name="debug-reader", debug=True)
```

The submission still creates a normal cloud job, registers its real shards, and
uses the normal worker entrypoint. It allocates one non-preemptible worker and
keeps that worker ready instead of immediately consuming shards.

The launch output prints the job ID and follow-up commands. Wait for the worker,
then run the pipeline against a private copy of its shard ledger:

```bash
macrodata debug status JOB_ID
macrodata debug run JOB_ID --max-shards 1
```

Every `debug run` rebuilds the private ledger from the job's registered shards.
Completing or failing a debug shard therefore does not mutate the job's real
shard ledger, and the same inputs are available on the next run. Omit
`--max-shards` to exercise all registered shards assigned to the retained
worker.

Synchronize the current project before another attempt:

```bash
macrodata debug sync JOB_ID .
macrodata debug run JOB_ID --max-shards 1
```

Sync excludes version-control data, virtual environments, caches,
`node_modules`, and dotenv files. The worker verifies the archive digest and
atomically replaces the prior source tree, so a failed upload cannot leave a
partial tree. Both the project root and its `src/` directory take precedence on
`PYTHONPATH` for subsequent attempts.

Source sync updates imported Python modules. When you change the pipeline
graph, reader, writer, or a callable captured into the serialized pipeline,
also replace the session's pipeline payload before running again:

```python
pipeline.sync_cloud_debug("JOB_ID")
```

This serializes the selected stage locally and atomically replaces the payload
used by the next attempt. Pass `stage_index=` when selecting a nonzero planned
stage.

Inspect the exact retained environment when imports or dependencies behave
differently from the submitting machine:

```bash
macrodata debug doctor JOB_ID
```

Doctor reports the worker Python executable/version, installed Refiner,
Cloudpickle, py-spy and Torch versions, GPU visibility, source digest, and the
pipeline payload path. It does not return mounted secret values.

Run an arbitrary non-interactive command with argv preserved exactly:

```bash
macrodata debug exec JOB_ID -- python -V
macrodata debug exec JOB_ID -- bash -lc 'nvidia-smi && pip freeze | head'
```

Close the session when you are done:

```bash
macrodata debug stop JOB_ID
```

Closing uses normal cloud job cancellation. The retained worker and its
ephemeral files are then removed.

## Current scope

Debug sessions are intentionally single-worker and start at the first pipeline
stage. They are for validating correctness, dependencies, resource behavior,
and performance before a separate normal launch scales out. Concurrent debug
attempts in one session are rejected.

## Internal Notes

The retained Modal function input is non-preemptible. A read-only snapshot of
registered shards seeds an atomic SQLite ledger in the container for each
attempt; the canonical cloud worker command receives that ledger explicitly.
