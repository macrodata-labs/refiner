# Successor to DataTrove — High-Level Architecture & Implementation

This document describes a **batch-only, ML-focused data processing system** inspired by Dataflow and MapReduce, adapted to **Slurm/HPC realities**, large-scale preprocessing, and **model-based inference with high startup cost**.

**Batch-only** here means:
- no real-time / online streaming
- no event-time, watermarks, triggers, timers
- all inputs are bounded

---

## 1. Design Principles

### What we optimize for
- Pure **batch preprocessing**
- **Shard-based progress**, not task-based
- **Long-lived workers** (avoid Slurm queue latency)
- **Implicit fusion by default**
- **Explicit materialization only at well-defined boundaries**
- **Bounded memory** with controlled disk spill
- **Efficient GPU inference** via async continuous batching

### What we explicitly do NOT promise
- Global ordering of records
- Exactly-once execution of user code
- Zero recomputation (retries are allowed)

---

## 2. Core Abstractions

### Shard
A **shard** is the smallest unit of scheduling and retry.

- Deterministically defined from input data
- Examples:
  - Parquet row-group ranges
  - JSONL byte ranges
  - `(file, offset_start, offset_end)`
- Many shards ≫ number of workers

---

### Stage
A **stage** is a sequence of operations.

- Inside a stage:
  - operations are **fully fused**
  - execution uses a **tight imperative loop** over shard rows
  - row/batch operators run in fused micro-batches (no per-step iterator chaining)
  - no disk writes unless memory limits force a spill
- Between stages:
  - data is **materialized**
  - progress is checkpointed
  - next stage consumes manifests

---

### Ledger
The **ledger** is the durable system of record.

- Not a process; a shared store
- Coordinates claims, retries, and commits

Minimal schema:
```

work_id
type: SHARD | MERGE | PARTITION
status: UNCLAIMED | CLAIMED | DONE | FAILED
lease_owner
lease_expiry
attempt
resource_class
temp_output_uri
error_code

```

---

## 3. Execution Model (Slurm-Friendly)

- User submits **N long-lived Slurm jobs**
- Each job runs a **worker loop**
- Workers pull work from the ledger

Worker loop:
1. Try to claim a READY **merge / partition job**
2. Else claim an UNCLAIMED **shard**
3. Execute work
4. Publish output
5. Update ledger
6. Repeat

No central scheduler process is required.

### Worker Execution (Fused)

Within a claimed shard, workers execute a fused plan with two logical step types:

- **Single-row step**: `row -> row | None`
- **Batchable step**: `list[row] -> list[row]`

Arbitrary mixes are supported by a per-step fused scheduler with step-local queues.
Batchable operators trigger only when their own queue reaches full batch size (tails flush at end-of-stream).

Workers batch across multiple claimed shards before completion:
- claim enough consecutive shards to satisfy current fused batch needs
- run fused operators on the concatenated row stream
- complete all claimed shards together on success (fail all on exception)
- this enables larger batchable windows without introducing a materialization boundary

Pseudo-flow:
1. Read rows from `reader.read_shard(shard)`
2. Apply fused row segments inline
3. Accumulate into bounded buffers for batchable segments
4. Emit outputs
5. Heartbeat periodically and complete/fail in ledger

### Cross-System Comparison (This Problem)

For fused in-worker execution with row/batch operators:

* **Spark**: fuses narrow operators inside a task (whole-stage style); breaks at shuffle/exchange.
* **Beam/Dataflow**: fuses adjacent element-wise transforms into runner stages; breaks at shuffle/stateful boundaries.
* **Daft**: optimizes lazy plans and executes fused partition-local operators; breaks at repartition/sort/join boundaries.

Refiner follows the same shape:
* fuse narrow local operators in one worker loop
* keep async inference as in-process fusion boundary
* materialize only at true stage boundaries (shuffle/sort/join/dedup)

---

## 4. Shard Lifecycle

### Claim
- Atomic transition `UNCLAIMED → CLAIMED`
- Lease with expiry
- Periodic heartbeats extend lease

### Execute
- Worker runs fused pipeline on shard
- Reads only the shard’s input range
- Produces **temporary output**

### Publish (Exactly-Once Output)
- Output written to unique temp path:
```

tmp/run_id/work_id/attempt_k/...

````
- Atomic publish:
- mark DONE
- record output URI
- Retries are safe (first publish wins)

---

## 5. Fusion vs Materialization

### Default: Implicit Fusion
Execution:

* Steps are inlined into one execution plan
* No intermediate files
* Bounded memory via micro-batch buffers
* Re-execution is cheap

---

### Fusion Boundaries (No Materialization)

A **fusion boundary** stops inlining but stays in the **same process**.

Use cases:
* Async inference islands
* Local execution-mode transitions (row-mode to async-mode and back)

Properties:
* No stage handoff
* No worker-count change
* No durable boundary output required

---

### Materialization Boundaries

Materialization happens only when required:

* Shuffle / GroupByKey
* Sort
* Dedup
* Join
* Explicit stage boundary
* Memory pressure spill

At a boundary:

* Data is written to disk/object store
* Next stage operates on these outputs
* Next stage may run with a different worker count/resource class

---

## 6. The Core Physical Primitive (Underlying All Boundaries)

Almost all “hard” batch operators are built from **one base pattern**:

### Partition → Spill → Reduce

#### 1. Partition (Fan-out)

While scanning input records:

* Compute a **bucket id**:

  * `hash(key) % P` for shuffle / dedup / join
  * range bucket for global sort
* Append record to an in-memory buffer for that bucket

#### 2. Spill (Controlled Disk Write)

* Each bucket buffer has a memory limit
* When exceeded:

  * buffer is flushed to disk as a **spill file (run)**
  * spill files are append-only chunks
* This guarantees bounded memory

> **Spill** = intentionally writing buffered data to disk to avoid OOM.

#### 3. Reduce (Fan-in)

* Each bucket becomes an independent work item
* A worker claims bucket `b`
* Reads all spill files for `b`
* Performs local heavy computation:

  * group by key
  * external sort
  * join
  * dedup
* Writes final output

---

## 7. Built-in Boundary Operators (Logical → Physical)

### Shuffle / GroupByKey

Logical:

```
GroupByKey(key_fn, combine_fn)
```

Physical:

* Hash-partition records
* Spill partitioned runs
* One reducer per partition
* Sort or hash-aggregate locally

---

### Sort

Logical:

```
Sort(key_fn, total_order=True/False)
```

Physical:

* Local in-memory sort + spill runs
* K-way merge
* Optional range partition for global order

---

### Dedup

Logical:

```
Dedup(key_fn, keep=first|min|max)
```

Physical:

* Shuffle by key
* Within partition, emit one record per key

---

### Join

Logical:

```
Join(left, right, key_fn, how=inner|left|...)
```

Physical:

* Shuffle both sides with same partitioner
* Local sort-merge or hash join per partition

---

## 8. Merges (Output Consolidation)

### Why merges are separate jobs

* Avoid race conditions
* Avoid “last shard does extra work”
* Make retries idempotent

### Merge Jobs

* Deterministic grouping (e.g. 5 shard outputs)
* When all shards in group DONE:

  * ledger creates a MERGE job
* Any worker can claim it

Merge job:

1. Read shard outputs
2. Write final merged file
3. Publish
4. Cleanup temp files

---

## 9. Ordering Semantics

* Outputs are defined as an **unordered set**
* Determinism = same set, not same order
* Ordering requires explicit `Sort` boundary

---

## 10. Model-Based Inference (High Startup Cost)

### Problem

* Model init expensive
* Want continuous batching
* Requires async runtime (e.g. vLLM)

### Async Inference Island

* Worker hosts:

  * one async event loop
  * long-lived inference server
* Sync pipeline feeds requests via queues
* Async loop submits requests continuously
* Results returned via queues

Only inference is async; rest remains synchronous.

Important:
* This is a **fusion boundary**, not a materialization boundary.
* Workers fuse all pre-inference operators, enter async island, then continue fused post-inference operators in the same process.
* Ordering can be relaxed for throughput unless explicit ordering is required.

### Local / Notebook Mode

`RefinerPipeline` exposes local execution helpers:
- `iter_rows()` for lazy fused execution over all shards (single-process, single-worker semantics)
- `materialize()` to eagerly collect all rows into memory
- `take(n)` for bounded sampling without full materialization

This mirrors the common pattern in Spark/Daft-style local usage:
- lazy transformation graph
- explicit terminal action for full materialization

---

## 11. Retry & Failure Model

* Shards are **at-least-once**
* Outputs are **exactly-once**
* Failures handled at shard or partition level
* OOM → escalate resource class
* Transient → retry
* Fatal → mark FAILED

---

## 12. Slurm Compatibility Strategy

* Submit workers once
* Over-shard inputs
* Dynamic behavior happens inside allocations
* No per-task Slurm submissions

---

## 13. Improvements vs DataTrove

| Aspect            | DataTrove       | New System          |
| ----------------- | --------------- | ------------------- |
| Progress tracking | task-based      | shard-based         |
| Disk usage        | explicit stages | implicit fusion     |
| Shuffle/dedup     | manual          | built-in operators  |
| Failure recovery  | coarse          | per-shard           |
| GPU inference     | per-task        | continuous batching |
| Slurm latency     | frequent        | amortized           |

---

## 14. Mental Model Summary

> *MapReduce extracted into reusable physical primitives.*

> *Shards are data, not tasks.*

> *Fusion by default, materialize only when forced.*

> *Correctness lives in the ledger; workers are replaceable.*
