---
title: "Cloud debug sessions"
description: "Iterate on a pipeline inside one retained cloud worker"
---

# Cloud debug sessions

Use a cloud debug session to validate a pipeline in its real cloud environment
before launching it at scale. Keep the pipeline script unchanged:

```python
import refiner as mdr

pipeline = mdr.read_jsonl("s3://datasets/input.jsonl")
pipeline.launch_cloud(name="debug-reader", num_workers=16)
```

Create a retained session by passing that script to the CLI:

```bash
macrodata debug pipeline.py
```

The command executes the script locally, requires exactly one
`launch_cloud(...)` call, and creates a normal cloud job from that launch. The
job registers its real stage-0 shards and allocates one non-preemptible worker.
The CLI remembers the session for the pipeline path, waits for the worker, and
synchronizes the current project before returning.

Run a small attempt:

```bash
macrodata debug run pipeline.py --max-shards 1
```

Attempts use the normal worker entrypoint, runtime token, environment,
resources, and telemetry. Every attempt starts with fresh shard state, so the
same inputs can be rerun without changing the job's normal execution records.

After editing source or the pipeline graph, perform a complete sync and rerun:

```bash
macrodata debug sync pipeline.py
macrodata debug run pipeline.py --max-shards 1
```

Sync executes the script again, captures its current cloud launch, uploads the
project source and serialized stage-0 pipeline together, derives fresh shards
inside the retained environment, and activates all three as one generation.
An interrupted or invalid sync leaves the previous generation runnable. Version
control data, virtual environments, caches, `node_modules`, and dotenv files are
excluded.

When local session state is available, sync reuses the script arguments
remembered when the session was created. Pass new arguments after `--` to
replace them, or end the command with `--` to clear them and run the script with
its default arguments:

```bash
macrodata debug sync pipeline.py -- --rows 20
macrodata debug sync pipeline.py --
```

If local session state is unavailable and you select the session with `--job`,
repeat any required script arguments after `--`. Without them, the script runs
with its default arguments:

```bash
macrodata debug sync pipeline.py --job JOB_ID -- --rows 20
```

Dependencies, Python and Refiner versions, CPU, memory, GPU, cloud placement,
runtime services, secrets, and plain environment variables are fixed when the
worker is allocated. If any of these settings change, sync asks you to stop and
create a new session:

```bash
macrodata debug stop pipeline.py
macrodata debug pipeline.py
```

Inspect the retained environment or execute a non-interactive command:

```bash
macrodata debug status pipeline.py
macrodata debug doctor pipeline.py
macrodata debug exec pipeline.py -- python -V
macrodata debug exec pipeline.py -- bash -lc 'nvidia-smi && pip freeze | head'
```

Exec commands time out after 20 minutes by default. Use `--timeout SECONDS` for
a different limit.

Profile an attempt with py-spy:

```bash
macrodata debug run pipeline.py --max-shards 1 --profile
```

The flamegraph is written to `refiner-debug-profile.svg`. Choose another path
with `--profile-output PATH`, or download the latest profile again:

```bash
macrodata debug profile pipeline.py --output profile.svg
```

The pipeline path is the normal session handle. When local session state is not
available, pass a job ID explicitly:

```bash
macrodata debug status --job JOB_ID
macrodata debug sync pipeline.py --job JOB_ID
```

Close the session when finished:

```bash
macrodata debug stop pipeline.py
```

If the retained worker crashes or Modal replaces it, the job and debug session
fail rather than silently continuing in a new container. Start a new session to
continue. Synchronized files, profiles, and changes made through exec are
ephemeral.

## Internal Notes

Shard claims and completions are written to an atomic, private SQLite ledger in
the retained worker rather than the job's canonical shard ledger. Each attempt
resets that private ledger and passes it explicitly to the normal cloud worker
entrypoint.
