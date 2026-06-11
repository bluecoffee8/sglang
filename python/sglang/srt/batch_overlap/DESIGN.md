# batch_overlap — Two-Batch Overlap (TBO)

## Purpose

Overlaps the attention computation of one micro-batch with the MoE all-to-all communication of another, hiding inter-GPU communication latency. This is the core throughput optimization for expert-parallel MoE models (e.g., DeepSeek-V3).

## Key Files

| File | Role |
|---|---|
| `operations.py` | Defines the atomic compute/communicate operation primitives |
| `operations_strategy.py` | Selects which strategy to use (overlap vs. serial) based on batch characteristics |
| `single_batch_overlap.py` | Single-batch execution path (no overlap, baseline) |
| `two_batch_overlap.py` | Two-batch overlap execution: interleaves compute A with comm B |

## Design

```
ModelRunner.forward()
       │
       └─ OperationsStrategy.select(batch)
              │
              ├─ SingleBatchOverlap   (baseline: serial attention → dispatch → FFN)
              │
              └─ TwoBatchOverlap      (overlap mode)
                     │
                     ├─ Chunk 0: attention(batch_A) || dispatch(batch_B)
                     ├─ Chunk 1: FFN(batch_A)       || attention(batch_B)
                     └─ Chunk 2: combine(batch_A)   || FFN(batch_B)
```

The `TwoBatchOverlap` class splits each decode step into two micro-batches. While one micro-batch runs compute (attention, FFN), the other runs communication (all-to-all dispatch/combine for MoE). CUDA streams are used to overlap the two. The `OperationsStrategy` decides whether to use overlap based on token count thresholds (`--tbo-token-distribution-threshold`).

## Dependencies

- `layers/communicator.py` — scatter/gather primitives
- `layers/moe/token_dispatcher` — DeepEP / Mooncake / Mori / Nixl dispatchers
- `model_executor/forward_batch_info.py` — `ForwardBatch`
- Requires EP (Expert Parallelism) and compatible attention backend
