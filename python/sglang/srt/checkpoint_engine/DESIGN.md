# checkpoint_engine — Checkpoint Saving & Loading

## Purpose

Manages incremental model checkpoint saves and weight updates during online training or fine-tuning co-located with inference. Allows the inference engine to receive updated weights from an external trainer without restarting.

## Key Files

| File | Role |
|---|---|
| `checkpoint_engine_worker.py` | Worker process that performs async checkpoint I/O |
| `update.py` | `UpdateWeightFromDiskReqInput` handler — applies a saved checkpoint to the live model |

## Design

```
External Trainer
       │  (writes checkpoint to disk)
       ▼
CheckpointEngineWorker (separate process / thread)
       ├─ Monitors checkpoint directory for new snapshots
       └─ Sends UpdateWeightReq → Scheduler → ModelRunner
                                                  │
                                                  └─ update.py: load_weights(path)
                                                         └─ hot-swaps model tensors in-place
```

The checkpoint engine runs alongside the inference server. When a new checkpoint appears, it triggers an `UpdateWeightFromDiskReqInput` message that flows through the standard IPC channel (ZeroMQ) to the `Scheduler`, which forwards it to `ModelRunner`. `ModelRunner` loads new weights in-place using the same `model_loader` infrastructure, minimizing disruption to ongoing inference.

## Relationship to weight_sync

`checkpoint_engine` handles file-based weight updates (trainer writes to disk first). `weight_sync/` handles direct tensor-level weight sync via IPC/NCCL without an intermediate file.
