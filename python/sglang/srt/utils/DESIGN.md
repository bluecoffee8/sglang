# utils — Shared Utilities

## Purpose

General-purpose utilities used across the entire SGLang codebase. Covers async I/O helpers, device management, profiling, HTTP infrastructure, memory management, and miscellaneous helpers that don't belong to a specific component.

## Key Files

| File | Role |
|---|---|
| `common.py` | Core utilities: `get_bool_env_var`, `suppress_noisy_warnings`, `BumpAllocator`, etc. |
| `__init__.py` | Re-exports the most commonly used utilities |
| `hf_transformers_utils.py` | HuggingFace model/tokenizer loading helpers |
| `hf_transformers_patches.py` | Monkey-patches for HF bugs / version compat |
| `patch_tokenizer.py` | Tokenizer patching for SGLang-specific behavior |
| `patch_torch.py` | PyTorch patches and workarounds |
| `cuda_ipc_transport_utils.py` | CUDA IPC handle serialization for cross-process tensor sharing |
| `host_shared_memory.py` | POSIX shared memory utilities |
| `aio_rwlock.py` | Async read-write lock |
| `device_timer.py` | GPU event-based timing |
| `gauge_histogram.py` | Prometheus gauge-histogram hybrid |
| `watchdog.py` | Background thread watchdog to detect stuck processes |
| `slow_rank_detector.py` | Detects lagging TP ranks in distributed setups |
| `profile_utils.py`, `profile_merger.py` | PyTorch profiler wrapper and trace merging |
| `nvtx_pytorch_hooks.py` | NVTX range hooks for Nsight profiling |
| `rpd_utils.py` | ROCm Performance Data (RPD) profiler integration |
| `auth.py` | API key authentication middleware |
| `network.py` | `get_local_ip_auto()`, port utilities |
| `numa_utils.py` | NUMA-aware memory binding |
| `offloader.py` | CPU-offload utilities for tensors |
| `torch_memory_saver_adapter.py` | Adapter for memory-saving checkpointing |
| `request_logger.py` | Structured per-request logging |
| `scheduler_status_logger.py` | Periodic scheduler state logging |
| `multi_stream_utils.py` | CUDA multi-stream context management |
| `poll_based_barrier.py` | Polling barrier for distributed synchronization |
| `phase_checker.py` | Asserts code runs in the expected phase (prefill/decode) |
| `bench_utils.py` | Benchmarking helpers (latency measurement, throughput calculation) |
| `video_decoder.py` | Video frame decoding for multi-modal models |
| `weight_checker.py` | Validates model weight checksums |
| `json_response.py` | Fast JSON response helpers for FastAPI |
| `http_middleware_patch.py` | Custom FastAPI middleware |
| `log_utils.py` | Log formatting and filtering |
| `field_validators.py` | Pydantic field validators |
| `async_probe.py` | Async process health probing |
| `custom_op.py` | `register_custom_op` — wrapper for `torch.library.custom_op` |

## Key Abstractions

### BumpAllocator

A simple arena allocator over a pre-allocated tensor. Used in hot paths (e.g., `batch_overlap/`) to avoid Python `torch.zeros()` allocation overhead.

### WatchDog

`watchdog.py` runs a background thread that sends a heartbeat. If the main thread hangs for longer than a threshold, the watchdog logs a stack trace and optionally kills the process. Used to surface deadlocks in distributed scenarios.

### CUDA IPC Transport

`cuda_ipc_transport_utils.py` provides serializable handles for CUDA tensors that can be passed between processes (e.g., Scheduler → ModelRunner for KV cache sharing) without copying data through CPU.
