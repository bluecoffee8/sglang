# observability — Metrics, Tracing & Monitoring

## Purpose

Exposes runtime behavior to external monitoring systems (Prometheus, OpenTelemetry, OpenResty/Mooncake trace). Tracks throughput, latency, queue depths, cache hit rates, and per-request timing across all pipeline stages.

## Key Files

| File | Role |
|---|---|
| `metrics_collector.py` | `SchedulerStats`, `ServerStats` — core metric snapshots collected each step |
| `request_metrics_exporter.py` | Exports per-request metrics (TTFT, TPOT, e2e latency) to Prometheus |
| `forward_pass_metrics.py` | Per-forward-pass counters (tokens, decode steps, etc.) |
| `req_time_stats.py` | `SchedulerReqTimeStats` — fine-grained per-request timing breakdown |
| `trace.py` | OpenTelemetry span creation and propagation |
| `mooncake_trace.py` | Mooncake-specific distributed trace hooks |
| `func_timer.py` | `FuncTimer` — lightweight function timing decorator |
| `startup_func_log_and_timer.py` | Logs timing of startup functions |
| `cpu_monitor.py` | Background CPU utilization monitoring |
| `label_transform.py` | Prometheus label normalization |
| `utils.py` | Bucket helpers for histogram metrics |

## Design

```
Scheduler (each step)
    │
    ├─ SchedulerStats.collect(batch)
    │       ├─ num_running_reqs, num_queue_reqs
    │       ├─ token_usage, cache_hit_rate
    │       └─ gen_throughput
    │
    └─ MetricsCollector.log_stats(stats)
            ├─ Prometheus gauges/counters  (/metrics endpoint)
            └─ stdout logging (if --log-requests)

Per-request lifecycle:
    TokenizerManager → SchedulerReqTimeStats.record_recv()
    Scheduler (schedule) → record_schedule()
    Scheduler (forward)  → record_forward()
    DetokenizerManager   → record_output()
    TokenizerManager     → RequestMetricsExporter.export(req)
            └─ Prometheus histograms: ttft_seconds, tpot_seconds, e2e_request_latency
```

### Prometheus Metrics

Key exported metrics:
- `sglang:num_running_reqs` — currently executing requests
- `sglang:num_queue_reqs` — waiting in scheduler queue
- `sglang:token_usage` — KV cache utilization (0–1)
- `sglang:cache_hit_rate` — prefix cache hit ratio
- `sglang:gen_throughput` — tokens generated per second
- `sglang:time_to_first_token_seconds` — TTFT histogram
- `sglang:time_per_output_token_seconds` — TPOT histogram

### OpenTelemetry Tracing

`trace.py` integrates with OTLP exporters when `--enable-otel-tracing` is set. Each request gets a root span; key stages (tokenize, schedule, forward, detokenize) are child spans.
