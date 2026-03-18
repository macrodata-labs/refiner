---
title: "Refiner Docs"
description: "Documentation for building and running Refiner data pipelines"
---

Refiner is a Python batch pipeline framework with local and cloud launchers, built-in readers and sinks, and Macrodata platform integration.

## Start Here

If you are new to Refiner, read these first:

1. [Pipeline basics](pipeline-basics.md)
2. [Launchers](launchers.md)
3. [CLI auth](cli-auth.md)

## Top-Level Structure

### Getting Started

- [Pipeline basics](pipeline-basics.md): how to build a pipeline from a source, transforms, and a sink
- [Launchers](launchers.md): how to run pipelines locally or on Macrodata Cloud
- [CLI auth](cli-auth.md): how to create a key, log in, verify auth, and log out

### Execution Model

- [Local execution](local-execution.md): in-process iteration, `take()`, and `materialize()`
- [Readers and sharding](readers-and-sharding.md): reader behavior and shard planning
- [Expression transforms](expression-transforms.md): vectorized expression-backed operations
- [Worker runtime](worker-runtime.md): direct worker execution and runtime lifecycle integration

### Platform Integration

- [Observability](observability.md): jobs, stages, workers, shards, logs, and metrics in Macrodata

## Typical Flows

### Local development

Use:

- [Local execution](local-execution.md) for quick iteration in notebooks and scripts
- [Launchers](launchers.md) when you want multi-worker local jobs and runtime lifecycle tracking

### Cloud execution

Use:

- [CLI auth](cli-auth.md) to authenticate
- [Launchers](launchers.md) for `launch_cloud(...)`
- [Observability](observability.md) to understand what gets reported back to Macrodata

### Dataset-oriented pipelines

Use:

- [Readers and sharding](readers-and-sharding.md) for input planning
- [Pipeline basics](pipeline-basics.md) for sink attachment
- [Expression transforms](expression-transforms.md) when you want Arrow-backed transforms instead of Python row UDFs
