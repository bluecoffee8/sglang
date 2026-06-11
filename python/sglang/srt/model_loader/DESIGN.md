# model_loader — Model Weight Loading

## Purpose

Handles loading pre-trained model weights from disk, remote storage, or Hugging Face Hub into the model's parameter tensors. Supports multiple formats (safetensors, pickle, sharded), quantization schemes, and remote-instance loading.

## Key Files

| File | Role |
|---|---|
| `loader.py` | `ModelLoader` — dispatches to the right loading strategy based on `LoadConfig` |
| `weight_utils.py` | Low-level utilities: file discovery, tensor shard iteration, dtype casting |
| `utils.py` | Higher-level helpers: model construction, parameter initialization |
| `remote_instance_weight_loader_utils.py` | Utilities for loading weights from another running SGLang instance |
| `ci_weight_validation.py` | Validates loaded weights match expected checksums in CI |

## Design

```
ModelRunner.__init__()
    │
    └─ ModelLoader(load_config)
            │
            ├─ DefaultModelLoader       (safetensors / pickle from local or HF Hub)
            ├─ ShardedModelLoader       (pre-sharded weights, one file per TP rank)
            ├─ QuantizedModelLoader     (GPTQ, AWQ, fp8 checkpoints)
            └─ RemoteInstanceLoader     (pull weights from another SGLang server)
```

### Loading Flow

1. Discover weight files in the model directory (`.safetensors`, `.bin`, `.pt`).
2. Iterate over shards; for each parameter tensor:
   a. Map the HF parameter name to the SGLang layer's parameter name.
   b. Apply TP sharding (slice along the right dimension for the current rank).
   c. Apply quantization (if needed: pack int4, compute scales, etc.).
   d. Copy into the model's pre-allocated parameter buffer.

### Name Mapping

Different model families use different parameter naming conventions. `weight_utils.py` contains per-architecture name remapping tables that translate HF checkpoint names to SGLang's internal naming.

### Remote Instance Loading

`remote_instance_weight_loader_utils.py` supports loading weights directly from another running SGLang instance via the `/get_weights_by_name` API. Used for weight synchronization in online training scenarios.
