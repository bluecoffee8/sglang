# distributed — Distributed Communication & Parallel State

## Purpose

Manages process group creation, distributed state (tensor/pipeline/data parallelism), and collective communication operations. Acts as the single source of truth for "which GPU am I on relative to other GPUs" and provides optimized all-reduce / broadcast primitives.

## Key Files

| File | Role |
|---|---|
| `parallel_state.py` | Process group creation and global getters (`get_tp_group()`, `get_pp_group()`, etc.) |
| `parallel_state_wrapper.py` | `ParallelState` dataclass aggregating all group handles |
| `communication_op.py` | High-level collective ops: `tensor_model_parallel_all_reduce`, `broadcast`, etc. |
| `naive_distributed.py` | Fallback all-reduce for non-NCCL backends |
| `utils.py` | TCP store setup, rank utilities |

## Design

```
init_distributed_environment()          ← wraps torch.distributed.init_process_group
        │
initialize_model_parallel(
    tp_size, pp_size, dp_size, ep_size, ...)
        │
        ├─ TP group  (tensor parallelism — split linear ops across GPUs)
        ├─ PP group  (pipeline parallelism — stage-by-stage execution)
        ├─ DP group  (data parallelism — replicate model, split requests)
        ├─ EP group  (expert parallelism — distribute MoE experts)
        └─ CP group  (context parallelism — split sequence dimension)
```

### Communication Backends

`communication_op.py` selects the best available backend per collective:
- **NCCL** — default on CUDA for all-reduce / all-gather
- **Custom all-reduce** (`--enable-custom-allreduce`) — hand-written kernel for small tensors
- **MSCCLPP** — MSCCL++ for specialized topologies
- **Symmetric memory** — NVLink symmetric memory all-reduce

Backend selection happens at `ModelRunner` init via `set_custom_all_reduce()` / `set_mscclpp_all_reduce()` / `set_torch_symm_mem_all_reduce()`.

### Process Group Getters

All code that needs a process group calls `get_tp_group()`, `get_pp_group()`, etc. These return `GroupCoordinator` objects (from vLLM lineage) that encapsulate both the `ProcessGroup` and rank metadata.

### Compilation Integration

`parallel_state.py` imports `register_split_op` from `compilation/compilation_config.py` and calls it at all-reduce sites so that `torch.compile` knows where to split the piecewise graph.
