# multiplex — Prefill-Decode Multiplexing

## Purpose

Enables a single server instance to act as both a prefill server and a decode server simultaneously ("PD-mux"). Normally prefill-decode disaggregation requires separate deployments; this module allows co-location with logical separation, reducing the overhead of two-instance deployment for smaller workloads.

## Key Files

| File | Role |
|---|---|
| `pdmux_context.py` | `PDMuxContext` — per-request context tracking which role (P or D) handles it |
| `multiplexing_mixin.py` | `MultiplexingMixin` — mixin for Scheduler to support dual-role operation |

## Design

```
Single SGLang Instance
    │
    └─ Scheduler + MultiplexingMixin
            │
            ├─ PD role routing:
            │       request.disagg_type == PREFILL  → SchedulerDisaggregationPrefillMixin path
            │       request.disagg_type == DECODE   → SchedulerDisaggregationDecodeMixin path
            │       request.disagg_type == None     → normal unified path
            │
            └─ PDMuxContext tracks:
                    which requests are in prefill-role vs decode-role
                    KV transfer state for each
```

### When to Use

PD-mux is activated by `--disaggregation-mode=mux`. It is simpler to deploy than full disaggregation (no separate prefill/decode servers) but provides less independent scaling. Useful when P:D ratios are balanced or when operator overhead is a concern.
