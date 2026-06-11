# eplb — Expert Parallelism Load Balancing

## Purpose

Dynamically rebalances MoE expert assignments across GPU ranks to equalize compute load. Without rebalancing, popular experts receive more tokens than others, creating stragglers. EPLB periodically remaps logical experts to physical expert slots based on observed activation statistics.

## Key Files

| File | Role |
|---|---|
| `eplb_manager.py` | `EPLBManager` — orchestrates rebalance cadence and triggers updates |
| `expert_distribution.py` | `ExpertDistributionRecorder` — ring buffer of per-layer expert activation counts |
| `expert_location.py` | `ExpertLocationMetadata` — current logical→physical expert mapping |
| `expert_location_dispatch.py` | Dispatch-time lookup: maps token-selected expert IDs to physical slots |
| `expert_location_updater.py` | Applies a new `ExpertLocationMetadata` to the live model |

## Design

```
ModelRunner (each forward pass)
    │
    └─ ExpertDistributionRecorder.record(layer, expert_counts)
            │   (ring buffer, size = eplb_rebalance_num_iterations)
            │
EPLBManager.on_forward_pass_end()  (called every step)
    │
    └─ every N steps:
            │
            ├─ recorder.dump_record() → logical_count[layer][expert]
            │
            ├─ check if rebalance needed (utilization imbalance threshold)
            │
            ├─ ExpertLocationMetadata.init_by_eplb(counts) → new mapping
            │       └─ runs EPLB algorithm: pack experts onto ranks to minimize load imbalance
            │
            └─ ExpertLocationUpdater.update(new_metadata)
                    ├─ layer-by-layer weight migration (respects chunk size limit)
                    └─ broadcast new dispatch table to all ranks
```

### EPLB Algorithm

Given `logical_count[layer][expert]` (activation frequency), the algorithm assigns logical experts to physical slots minimizing the maximum slot load across ranks. The optimizer runs on CPU; the result is a permutation table applied via `ExpertLocationUpdater`.

### Chunked Updates

To avoid stalling inference during rebalancing, weight migration is chunked: `eplb_rebalance_layers_per_chunk` layers are updated per step. This spreads the migration cost across multiple inference steps.

## Configuration

Key server args: `--enable-eplb`, `--eplb-rebalance-num-iterations`, `--eplb-rebalance-layers-per-chunk`, `--expert-distribution-recorder-buffer-size`.
