# state_capturer — Model State Capture

## Purpose

Captures internal model states during inference for analysis, distillation, or online learning. Primary use case: capturing expert routing decisions in MoE models to feed EPLB or offline analysis tools.

## Key Files

| File | Role |
|---|---|
| `base.py` | `BaseStateCapturer` ABC — interface for all capturers |
| `routed_experts.py` | `RoutedExpertsCapturer` — records which experts were selected per token per layer |
| `indexer_topk.py` | `IndexerTopK` — efficient top-k index extraction kernel used by capturers |

## Design

```
ModelRunner forward pass
    │
    └─ FusedMoE.forward()
            │
            └─ RoutedExpertsCapturer.record(layer_id, expert_indices, token_count)
                    │
                    └─ appended to circular buffer

ExpertDistributionRecorder (eplb/)
    └─ reads from RoutedExpertsCapturer buffer
            └─ aggregates into per-layer counts → EPLB rebalancing
```

The state capturer is designed to add minimal overhead: captures use pre-allocated GPU buffers and only copy data to CPU on explicit `dump()` calls.

## Extension

To capture different states (e.g., attention weights, hidden norms), subclass `BaseStateCapturer` and register in `ModelRunner` via the hook system (`model_executor/hook_manager.py`).
