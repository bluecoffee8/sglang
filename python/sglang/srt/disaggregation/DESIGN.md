# disaggregation — Prefill-Decode Disaggregation

## Purpose

Separates the prefill (prompt encoding) and decode (autoregressive generation) phases onto different server instances. The prefill instance computes KV cache for the prompt and transfers it to the decode instance, which then continues generation. This enables independent scaling of prefill and decode capacity.

## Key Files

| File | Role |
|---|---|
| `prefill.py` | `SchedulerDisaggregationPrefillMixin` — prefill-side lifecycle |
| `decode.py` | `SchedulerDisaggregationDecodeMixin` — decode-side lifecycle |
| `decode_schedule_batch_mixin.py` | Decode batch scheduling hooks |
| `decode_hicache_mixin.py` | Decode-side HiCache integration |
| `decode_kvcache_offload_manager.py` | Manages KV cache offload during decode migration |
| `encode_server.py` | Prefill-side gRPC server for bootstrap/handshake |
| `encode_grpc_server.py` | gRPC service implementation for prefill server |
| `encode_receiver.py` | Decode-side receiver for KV transfer metadata |
| `kv_events.py` | KV transfer event types |
| `utils.py` | Shared types: `DisaggregationMode`, `TransferBackend`, `MetadataBuffers` |

## Design

```
Client
  │  (HTTP request)
  ▼
Prefill Server (encode)              Decode Server
  │                                        │
  ├─ Bootstrap Queue                       ├─ Receive bootstrap metadata
  │   └─ handshake with decode server      │   (host/port for KV transfer)
  │                                        │
  ├─ Waiting Queue                         ├─ PreallocQueue
  │   └─ run forward (prefill)             │   └─ allocate KV slots
  │                                        │
  ├─ KV Transfer (RDMA/NVLink/P2P)─────────→ KV arrives
  │                                        │
  └─ Inflight Queue                        └─ TransferQueue → run decode
      └─ poll until transfer done
```

### Transfer Backends

Controlled by `--disaggregation-transfer-backend`:
- **Mooncake** — RDMA-based high-throughput KV transfer
- **Nixl** — NVIDIA Inference Xfer Library
- **P2P** — direct GPU peer-to-peer copy (same-node)

### Three-Phase Prefill Lifecycle

1. **Bootstrap**: Prefill server contacts decode server to get the KV slot pre-allocation address.
2. **Forward**: Prefill runs the attention forward, filling KV cache into allocated slots.
3. **Transfer/Inflight**: KV data is sent to decode server; request stays in the inflight queue until transfer ACK.

### Optimistic Prefill Retry

If KV transfer fails (e.g., decode instance OOM), the prefill can be retried with a different decode target. The `SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB` env var simulates failures for testing.
