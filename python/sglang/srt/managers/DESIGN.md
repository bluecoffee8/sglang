# managers — Core Process Managers

## Purpose

Contains the three primary processes that form the SGLang serving pipeline: `TokenizerManager`, `Scheduler`, and `DetokenizerManager`. Also houses the data structures (`io_struct`, `schedule_batch`) passed between them, and supporting controllers for data parallelism, multi-modal processing, and disaggregation.

## Key Files

| File | Role |
|---|---|
| `tokenizer_manager.py` | `TokenizerManager` — tokenizes inputs, tracks active requests, streams results |
| `scheduler.py` | `Scheduler` — batch scheduling, KV cache management, drives forward pass |
| `detokenizer_manager.py` | `DetokenizerManager` — converts token IDs to text, manages streaming |
| `io_struct.py` | All IPC message types (requests, responses, control messages) |
| `schedule_batch.py` | `ScheduleBatch`, `Req` — the central batch and request state objects |
| `schedule_policy.py` | `SchedulePolicy` — algorithms for selecting which requests to run next |
| `tp_worker.py` | `TpModelWorker` — wraps `ModelRunner` for one TP worker |
| `data_parallel_controller.py` | Routes requests across DP replicas |
| `communicator.py` | ZMQ-based IPC between managers |
| `multimodal_processor.py` | Multi-modal input preprocessing (image, audio, video) |
| `disagg_service.py` | `DisaggService` — coordinates prefill↔decode server communication |
| `cache_controller.py` | Cache management commands (flush, resize, etc.) |
| `overlap_utils.py` | Utilities for prefill/decode computation overlap |
| `prefill_delayer.py` | Delays prefill batches to improve decode throughput |
| `hisparse_coordinator.py` | HiSparse attention coordinator for hybrid sparse models |

## Process Architecture

```
Client (HTTP/gRPC)
    │
    ▼
TokenizerManager (main process or subprocess)
    │  ZMQ push
    ▼
Scheduler (subprocess per TP group)
    │  ZMQ push
    ├─ TpModelWorker → ModelRunner → GPU forward pass
    │
    └─ ZMQ push
    ▼
DetokenizerManager (subprocess)
    │  ZMQ push back to TokenizerManager
    ▼
TokenizerManager → response to client
```

### TokenizerManager

Owns the tokenizer, handles:
- Tokenizing input text / encoding multi-modal inputs
- Distributing requests to Scheduler(s) (one per DP replica)
- Assembling streaming token chunks into final responses
- Session management (multi-turn context tracking)

### Scheduler

The heart of SGLang. Per step it:
1. Selects a batch of requests from the waiting queue (`SchedulePolicy`)
2. Allocates KV cache slots (`mem_cache/`)
3. Calls `TpModelWorker.forward()` → `ModelRunner`
4. Processes output tokens (finish detection, sampling)
5. Returns results to `DetokenizerManager`

### Schedule Batch (`schedule_batch.py`)

`ScheduleBatch` is the mutable object representing one forward pass:
- `Req` list — which requests are in this batch
- `ForwardBatch` — tensor-level view passed into `ModelRunner`
- prefix/extend/decode mode selection

### IO Struct (`io_struct.py`)

All ZMQ messages are typed dataclasses defined here. Key types:
- `TokenizedGenerateReqInput` — tokenized request sent Tokenizer→Scheduler
- `BatchTokenizedGenerateReqInput` — batch version
- `GenerationOutput` — per-token output Scheduler→Detokenizer
- `AbortReq` — cancellation

### Schedule Policy

`schedule_policy.py` implements multiple scheduling algorithms:
- **Continuous Batching** — greedily fill up to max batch size
- **Priority** — weighted priority queues
- **LPM** (Longest Prefix Match) — prioritizes requests with long cache hits

### Data Parallelism

`DataParallelController` routes incoming requests round-robin or by load to multiple Scheduler processes, each driving an independent TP group.
