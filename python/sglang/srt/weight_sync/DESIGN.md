# weight_sync — Live Weight Synchronization

## Purpose

Enables direct, low-latency weight updates from an external trainer to the running inference engine without disk I/O. Used in online learning / RLHF pipelines where the trainer and inference engine co-exist on the same node or cluster.

## Key Files

| File | Role |
|---|---|
| `tensor_bucket.py` | `TensorBucket` — accumulates weight tensors and triggers bulk sync |
| `utils.py` | IPC setup, NCCL group initialization for trainer↔inference sync |

## Design

```
Trainer (separate process)
    │
    └─ weight_sync.utils.create_sync_group()
            │   (NCCL process group spanning trainer + inference ranks)
            │
            └─ TensorBucket
                    ├─ add(param_name, tensor)   ← trainer calls after each weight update
                    ├─ flush()                   ← broadcast accumulated tensors to inference
                    └─ inference ModelRunner receives tensors
                            └─ hot-swap param.data in-place
```

### Bucketing

Sending individual tensors per parameter has high per-call NCCL overhead. `TensorBucket` accumulates tensors up to a threshold size (default 10 MB) and sends them in a single `broadcast` call, amortizing communication overhead.

### vs. checkpoint_engine

| | `weight_sync` | `checkpoint_engine` |
|---|---|---|
| Transfer path | NCCL direct GPU→GPU (or CUDA IPC) | disk file |
| Latency | Sub-millisecond for small updates | Seconds (I/O bound) |
| Use case | Online RL, rapid iteration | Periodic checkpoints |

## Integration

The `Engine.update_weights_from_tensor()` Python API calls into this subsystem. The trainer calls this API (or the HTTP `/update_weights` endpoint) to push new weights into the live inference server.
