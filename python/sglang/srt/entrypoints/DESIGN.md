# entrypoints — Server Entry Points & Engine

## Purpose

Hosts the externally-facing API servers (HTTP/gRPC) and the `Engine` class that orchestrates all internal processes. This is where client requests enter the system and where responses are assembled and returned.

## Key Files

| File | Role |
|---|---|
| `engine.py` | `Engine` — launches scheduler/tokenizer/detokenizer processes, top-level Python API |
| `EngineBase.py` | `EngineBase` ABC — shared interface for `Engine` and Ray engine variants |
| `http_server.py` | FastAPI application with OpenAI-compatible REST endpoints |
| `http_server_engine.py` | Adapter connecting `http_server` to `Engine` |
| `grpc_server.py` | gRPC server exposing generation endpoints |
| `engine_info_bootstrap_server.py` | Startup coordination server (for multi-node bootstrap) |
| `engine_score_mixin.py` | Score/reward endpoint mixin |
| `context.py` | Request context (correlation IDs, tracing) |
| `warmup.py` | Pre-serving warmup pass |
| `ssl_utils.py` | TLS certificate helpers |
| `v1_loads.py` | Legacy v1 API compatibility shims |
| `tool.py` | Tool/function-call response formatting |
| `harmony_utils.py` | Harmony protocol utilities |

## Design

```
HTTP Client
    │
    ▼
http_server.py  (FastAPI + uvloop)
    │
    └─ HttpServerEngine
            │
            └─ Engine
                    │
                    ├─ TokenizerManager   (process, ZMQ)
                    ├─ Scheduler          (process, ZMQ)
                    └─ DetokenizerManager (process, ZMQ)

gRPC Client
    │
    ▼
grpc_server.py
    └─ Engine (same instance)
```

### Engine Process Architecture

`Engine.__init__()` spawns three child processes connected via ZeroMQ IPC sockets:
- **TokenizerManager** — tokenizes input text, handles multi-modal encoding, routes to Scheduler
- **Scheduler** — batches requests, manages KV cache, drives the ModelRunner forward pass
- **DetokenizerManager** — converts output token IDs back to text, handles streaming

The `Engine` itself stays in the main process and forwards Python API calls over ZMQ.

### HTTP Endpoints

`http_server.py` implements OpenAI-compatible routes:
- `POST /v1/completions` — legacy text completion
- `POST /v1/chat/completions` — chat completion (streaming + non-streaming)
- `POST /v1/embeddings` — embedding generation
- `GET /v1/models` — list available models
- `GET /health` — health check
- Anthropic and Ollama compatibility endpoints

### Startup Sequence

1. `Engine.__init__()` — parse args, spawn processes.
2. `engine_info_bootstrap_server.py` — coordinates multi-node initialization.
3. `warmup.py` — runs a synthetic batch to warm up CUDA graphs.
4. HTTP/gRPC server starts accepting traffic.
