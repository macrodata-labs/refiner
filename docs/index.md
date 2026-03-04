---
title: "Refiner Docs"
description: "User-facing guide and roadmap for building batch data pipelines with Refiner"
---

Refiner is a Python pipeline framework for row-oriented data processing across CSV, JSONL, and Parquet inputs.

## Long-Term Direction

Refiner is being built as a batch-first processing engine with:

- Implicit operator fusion inside stages (row and batch steps in one execution loop).
- Explicit materialization boundaries only when required (for example shuffle/sort/join).
- Long-lived worker execution with shard-based progress and retries.
- Async model-processing islands inside a stage without forcing stage handoff.
- Launcher portability across local, Slurm, and Ray runtimes.
- Observability-first lifecycle integration (job, stage, worker, shard, logs, and user OTEL metrics).

## What You Can Use Now

- Build pipelines with `read_csv(...)`, `read_jsonl(...)`, or `read_parquet(...)`.
- Add row-level transforms with `.map(...)`, batch transforms with `.batch_map(...)`, and expansion with `.flat_map(...)`.
- Run locally with lazy iteration (`for row in pipeline`) or eager collection (`pipeline.materialize()`).
- Run worker-driven execution with `Worker.run()` for shard-claiming and ledger updates.

## Start Here

1. `docs/pipeline-basics.md` for core API usage.
2. `docs/local-execution.md` for lazy local iteration and materialization.
3. `docs/readers-and-sharding.md` for input readers and shard behavior.
4. `docs/worker-runtime.md` for worker lifecycle and ledger interaction.
5. `docs/launchers.md` for local launcher usage.
6. `docs/cli-auth.md` for `macrodata login`, `whoami`, and `logout`.
7. `docs/observability.md` for Macrodata Observer lifecycle integration.

## Planned Additions

- `launch_local()` user API and launcher interfaces shared by future Slurm/Ray launchers.
- Stage/materialization boundary operators (shuffle, dedup, sort, join).
- Async-island execution model for model-based processing with pull-based completion.
- Expanded observability docs for backend metric querying and cardinality guardrails.

## Internal Notes

- These docs reflect current behavior in `src/refiner/pipeline.py`, `src/refiner/readers/`, and `src/refiner/runtime/worker.py`.
- Long-term goals are aligned with `OVERVIEW.md` and should remain consistent with architecture updates.
