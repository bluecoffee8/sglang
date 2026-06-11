# kv_canary — KV Cache Correctness Canary

## Purpose

Instruments the KV cache to detect correctness regressions — specifically, cases where a cached KV entry is used for a position whose actual token ID does not match what was originally computed. Acts as a canary that fires when reuse produces incorrect outputs, without requiring end-to-end output comparison.

## Key Files

| File | Role |
|---|---|
| `api.py` | `install_canary()` — top-level setup; patches pool and returns `CanaryManager` |
| `config.py` | `CanaryConfig`, `CanaryMode` (NONE / CHECK / PERTURB) |
| `buffer_group.py` | `CanaryBufferGroup` — parallel token-ID buffers shadowing each KV pool slot |
| `capacities.py` | `CanaryLaunchCapacities` — sizing information for buffer allocation |
| `endpoint.py` | `CanaryEndpoint` — per-forward-pass check invocation |
| `expected_inputs.py` | Builds expected token IDs for each KV slot in a forward batch |
| `plan_input.py` | `CanaryPlanInput` — per-step plan for which slots to check |
| `radix_cache_walker.py` | Walks the radix tree to collect token IDs for cached prefixes |
| `req_to_expected_token_ids_manager.py` | Maps request → expected token ID sequence |
| `state.py` | Mutable canary state across steps |
| `sweep_plan_builder.py` | Builds a sweep plan to check all active slots over time |

## Design

```
ModelRunner.forward()
    │
    └─ CanaryManager.pre_forward(batch)
            │
            ├─ CanaryPlanInput: which (slot, layer) pairs to check this step
            ├─ expected_token_ids[slot] ← RadixCacheWalker + req token IDs
            └─ inject expected IDs into KV pool parallel buffer

    └─ (kernel runs, writes K/V to KV cache)

    └─ CanaryManager.post_forward(batch)
            └─ CanaryEndpoint.check()
                    └─ for each checked slot: stored_token_id == expected?
                            → mismatch → log error / raise
```

### Two Modes

- **CHECK** — reads back the token IDs stored in the canary buffer and asserts they match expected. Zero overhead on non-checked steps.
- **PERTURB** — deliberately writes wrong token IDs to some slots to test that the canary detects the fault (used to validate the canary itself).

### Why Shadow Buffers?

The KV cache stores floating-point activations, not token IDs. The canary maintains a separate integer buffer (one entry per KV slot) that records which token generated each slot. This allows ID-level correctness checks without decoding KV activations.
