# lora — LoRA Adapter Management

## Purpose

Implements multi-adapter LoRA serving: multiple LoRA adapters can be loaded and hot-swapped at request granularity, following the S-LoRA / Punica design. Adapters are batched together so different requests in the same batch can use different adapters without separate forward passes.

## Key Files

| File | Role |
|---|---|
| `lora_manager.py` | `LoRAManager` — top-level orchestrator: adapter registry, activation, weight management |
| `lora.py` | `LoRAAdapter` — per-adapter weight container |
| `lora_config.py` | `LoRAConfig` — rank, alpha, target modules for one adapter |
| `layers.py` | `BaseLayerWithLoRA`, `LinearWithLoRA`, `FusedMoEWithLoRA` — LoRA-wrapped layer classes |
| `mem_pool.py` | `LoRAMemoryPool` — pre-allocated GPU buffer for adapter weights |
| `eviction_policy.py` | LRU/random eviction when adapter pool is full |
| `lora_registry.py` | `LoRARef` — adapter reference (path, ID); registry of loaded adapters |
| `lora_drainer.py` | `LoRADrainer` — waits for in-flight requests on an adapter before eviction |
| `lora_overlap_loader.py` | `LoRAOverlapLoader` — async adapter loading overlapped with inference |
| `lora_moe_runners.py` | MoE-specific LoRA computation runners |
| `lora_moe_runner_marlin.py` | Marlin-quantized MoE LoRA runner |
| `utils.py` | Target module name normalization, LoRA type detection |
| `deepseek_mla_correction.py` | Correction for DeepSeek MLA LoRA weight shapes |

## Design

```
Request(lora_id="my-adapter")
    │
    └─ LoRAManager
            │
            ├─ LoRARegistry: lora_id → LoRAAdapter
            │       (LRU eviction if > max_loras_per_batch)
            │
            ├─ LoRAMemoryPool: pre-allocated GPU buffers for lora_A, lora_B weights
            │       └─ evict_and_load(new_adapter) → copy weights into pool slot
            │
            └─ per forward batch:
                    ├─ activate_loras(batch.lora_ids)
                    │       └─ set weight pointers in each LinearWithLoRA layer
                    └─ LinearWithLoRA.forward(x)
                            ├─ base_weight(x)           (standard linear)
                            └─ + lora_B(lora_A(x)) * scaling  (batched per-request)
```

### Batched Multi-Adapter Computation

`LinearWithLoRA` uses the Punica/S-LoRA segmented GEMM kernel that handles a batch where each token's adapter index may differ. This avoids the naive approach of grouping tokens by adapter and running separate GEMMs.

### Async Loading

`LoRAOverlapLoader` prefetches adapter weights from disk into pinned host memory while inference runs, then streams them to GPU. Combined with `LoRADrainer` for safe eviction, this hides most adapter loading latency.

## Configuration

Key server args: `--enable-lora`, `--max-loras-per-batch`, `--max-lora-rank`, `--lora-paths`.
