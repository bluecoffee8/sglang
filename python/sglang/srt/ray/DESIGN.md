# ray — Ray-Based Distributed Serving

## Purpose

Wraps SGLang's multi-process architecture in Ray actors to enable cluster-scale deployment, fault-tolerant serving, and integration with Ray Serve. Mirrors the structure of `entrypoints/` but uses Ray remote actors instead of `multiprocessing`.

## Key Files

| File | Role |
|---|---|
| `engine.py` | `RayEngine` — Ray-based equivalent of `Engine`; spawns actors |
| `http_server.py` | HTTP server that delegates to `RayEngine` |
| `scheduler_actor.py` | `SchedulerActor` — Ray actor wrapping `Scheduler` |
| `data_parallel_controller.py` | `RayDataParallelController` — routes requests across DP replicas using Ray |

## Design

```
Ray Cluster
    │
    └─ RayEngine (driver)
            │
            ├─ RayDataParallelController
            │       └─ routes requests to SchedulerActors
            │
            ├─ SchedulerActor  (Ray remote, GPU node 0)
            │       └─ Scheduler + ModelRunner (TP group 0)
            │
            └─ SchedulerActor  (Ray remote, GPU node 1)
                    └─ Scheduler + ModelRunner (TP group 1)
```

### vs. multiprocessing Engine

| Aspect | `entrypoints/Engine` | `ray/RayEngine` |
|---|---|---|
| Process model | Python multiprocessing | Ray remote actors |
| Communication | ZeroMQ IPC | Ray object store + gRPC |
| Fault tolerance | Manual restart | Ray actor resurrection |
| Cluster support | Single node (or manual multi-node) | Multi-node via Ray |

### Ray Serve Integration

`RayEngine` can be wrapped in a Ray Serve deployment for autoscaling and rolling updates. The HTTP server in `ray/http_server.py` can be replaced by Ray Serve's ingress.
