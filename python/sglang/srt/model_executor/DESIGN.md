# model_executor — Model Execution & CUDA Graph Management

## Purpose

Owns the mechanics of running a model forward pass on GPU: constructing `ForwardBatch` tensors, managing CUDA graph capture and replay, and dispatching to the model's `forward()` method. This is the boundary between scheduling logic (which requests to run) and execution logic (how to run them).

## Key Files

| File | Role |
|---|---|
| `model_runner.py` | `ModelRunner` — top-level executor; loads model, manages state, drives forward passes |
| `forward_batch_info.py` | `ForwardBatch`, `ForwardMode` — per-step tensor inputs to the model |
| `forward_context.py` | `ForwardContext` — thread-local context holding current batch and backend |
| `cuda_graph_config.py` | `CudaGraphConfig` — which batch sizes/phases get graph capture |
| `cuda_graph_buffer_registry.py` | Registry of captured CUDA graph buffers per (phase, batch_size) |
| `input_buffers.py` | Pre-allocated input buffers reused across steps to avoid allocation overhead |
| `hook_manager.py` | `HookManager` — per-step lifecycle hooks (pre/post forward) |
| `model_runner_kv_cache_mixin.py` | KV cache initialization and management logic extracted from `ModelRunner` |
| `forward_batch_deepseek_mha_mixin.py` | DeepSeek MHA-specific `ForwardBatch` fields |
| `pool_configurator.py` | Configures memory pool sizing based on available GPU memory |
| `cpu_graph_runner.py` | CPU-mode CUDA graph equivalent (for CPU inference) |
| `mindspore_runner.py` | MindSpore backend runner |

## Design

```
Scheduler
    │
    └─ TpModelWorker.forward(schedule_batch)
            │
            └─ ModelRunner.forward(forward_batch)
                    │
                    ├─ [CUDA graph path]
                    │       CudaGraphBufferRegistry.lookup(phase, bs)
                    │               → replay captured graph
                    │
                    └─ [eager path]
                            ForwardContext.set(forward_batch, attn_backend)
                            model.forward(input_ids, positions, forward_batch)
                            → logits
```

### ForwardBatch

`ForwardBatch` is the central data object for one step:
- `input_ids` — token IDs for this step (all requests concatenated)
- `positions` — position indices
- `req_to_token` — mapping from request to its KV cache slot indices
- `forward_mode` — PREFILL / EXTEND / DECODE / DUMMY
- Per-request metadata (seq lens, lora IDs, sampling params, etc.)

`ForwardMode` determines which attention path to use:
- **PREFILL** — first pass, all tokens are new
- **EXTEND** — continuation with a prefix already in cache (chunked prefill)
- **DECODE** — single new token per request

### CUDA Graph Capture

At startup, `ModelRunner` runs warmup passes at discrete batch sizes defined by `CudaGraphConfig`. For each `(phase, batch_size)` pair:
1. Eager run with dummy data to trigger JIT compilation.
2. `torch.cuda.CUDAGraph.capture()` records the forward pass.
3. Replay uses pre-allocated `input_buffers` to avoid allocation inside the graph.

At runtime, if the current batch size matches a captured size, the graph is replayed; otherwise the eager path runs.

### Hook Manager

`HookManager` fires registered pre/post-forward hooks used by observability (profiling), canary checks, EPLB recording, and speculative decoding state updates.
