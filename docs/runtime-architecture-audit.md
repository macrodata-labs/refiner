# Runtime Architecture Audit

This document tracks the current Refiner structure after the worker/platform cleanup.

## Current Structure

```text
src/refiner/
  cli/                    # user-facing CLI commands
  execution/              # execution engine and operator internals
  io/                     # generic file/file-set helpers
  launchers/              # job orchestration
    base.py
    cloud.py
    local.py
  pipeline/               # pipeline definition and source/data model
    data/
    expressions.py
    pipeline.py
    planning.py
    sources/
    steps.py
  platform/               # Macrodata backend integration
    auth.py
    client/
      api.py
      http.py
      models.py
      serialize.py
    manifest.py
  worker/                 # worker process runtime
    entrypoint.py
    lifecycle/
      base.py
      file.py
      platform.py
    metrics/
      api.py
      context.py
      otel.py
    resources/
      cpu.py
      memory.py
      network.py
    runner.py
    workdir.py
```

## Boundary Check

- `pipeline/`
  - owns pipeline composition, expressions, source abstractions, shard and row data
  - does not own worker lifecycle or launcher orchestration
- `execution/`
  - owns execution of compiled pipeline steps
  - does not own worker lifecycle, platform transport, or launch policy
- `worker/`
  - owns worker process boot, lifecycle integration, metrics emission, and resource helpers
  - `lifecycle/` is now the single home for runtime lifecycle contracts and implementations
  - `workdir.py` sits at the worker level because it is shared worker runtime state, not lifecycle logic
- `launchers/`
  - own job orchestration only
  - `LocalLauncher` always launches subprocess workers, even for one worker
  - `BaseLauncher` owns shared plan/manifest/platform job setup helpers
- `platform/`
  - owns API auth, transport, typed response decoding, cloud submission payloads, and manifest creation

## Remaining Known Leak

- `pipeline/sources/base.py` still emits worker metrics directly. That is the main remaining cross-boundary leak.
