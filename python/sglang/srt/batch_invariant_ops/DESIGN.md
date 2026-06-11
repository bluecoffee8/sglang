# batch_invariant_ops — Batch-Invariant Operations

## Purpose

Provides a mechanism to execute operations that produce the same result regardless of which requests are in the current batch (e.g., shared prefix computation, model-level initializations). The key insight is that these operations can be hoisted out of the per-request loop and executed once per forward pass.

## Key Files

| File | Role |
|---|---|
| `batch_invariant_ops.py` | Core `BatchInvariantOps` class — stores, deduplicates, and runs invariant ops |

## Design

```
Scheduler / ModelRunner
       │
       └─ BatchInvariantOps
              ├─ register(op_fn, ...)   ← called once at init by layers that need invariant ops
              └─ execute(forward_batch) ← called once per step before the model forward pass
                     │
                     └─ runs each registered op with the current batch context
```

A `BatchInvariantOps` instance is owned by `ModelRunner`. During model initialization, layers or modules can register operations that should fire once per forward step regardless of batch contents. This avoids redundant computation for shared-state side effects (e.g., updating rotary buffers, refreshing expert location mappings) that would otherwise be repeated per request.

## Invariants

- Each registered op is called exactly once per `execute()` call.
- Ops are called in registration order.
- Ops must not depend on per-request state; they receive only the `ForwardBatch` for shape/device metadata.
