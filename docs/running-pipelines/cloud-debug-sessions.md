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

This serializes stage 0 locally and atomically replaces the payload used by the
next attempt. Retained debug sessions currently reject nonzero stages because
their runtime token, shard snapshot, and upstream boundary belong to stage 0.

Inspect the exact retained environment when imports or dependencies behave
differently from the submitting machine:

```bash
macrodata debug doctor JOB_ID
```

Doctor reports the worker Python executable/version, installed Refiner,
Cloudpickle, py-spy and Torch versions, GPU visibility, source digest, and the
pipeline payload path. It does not return mounted secret values.

Profile the exact next attempt and download a py-spy flamegraph:

```bash
refiner debug run JOB_ID --max-shards 1 --profile
```

The SVG is written to `refiner-debug-profile.svg` by default. Use
`--profile-output PATH` to choose another location, or retrieve the most recent
profile again with `refiner debug profile JOB_ID --output PATH`. (`macrodata`
remains an equivalent CLI alias.) Profiling
wraps the normal worker entrypoint; it does not use a reduced or synthetic
execution path.

Run an arbitrary non-interactive command with argv preserved exactly:

```bash
macrodata debug exec JOB_ID -- python -V
macrodata debug exec JOB_ID -- bash -lc 'nvidia-smi && pip freeze | head'
```

Exec commands time out after 20 minutes by default. Use `--timeout SECONDS` to
set a different limit for one command.

Close the session when you are done:

```bash
macrodata debug stop JOB_ID
```

Closing uses normal cloud job cancellation. The retained worker and its
ephemeral files are then removed.

If the retained worker exits or is replaced after an infrastructure failure,
the debug session fails instead of attaching to the replacement container.
Start a new debug session to continue. Source synced into the old container and
changes made with `debug exec` are ephemeral and are not recovered.
