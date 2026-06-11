# configs — Model & Server Configuration

## Purpose

Centralizes configuration dataclasses for model architectures, device settings, and loading parameters. Each file corresponds to either a model family's custom config (beyond what Hugging Face provides) or a cross-cutting concern like quantization or load format.

## Key Files

| File | Role |
|---|---|
| `model_config.py` | `ModelConfig` — primary config parsed from HF config + server args |
| `load_config.py` | `LoadConfig` — weight loading format, quantization, sharding |
| `device_config.py` | `DeviceConfig` — device type, CUDA/CPU/NPU selection |
| `update_config.py` | Patch `ModelConfig` post-init (e.g. unaligned CPU TP) |
| `utils.py` | Config utility helpers |
| `model_config_parser_registry.py` | Registry mapping HF `model_type` → custom config class |
| `linear_attn_model_registry.py` | Registry for linear-attention model types |
| `mamba_utils.py` | Mamba SSM configuration helpers |
| `modelopt_config.py` | NVIDIA ModelOpt quantization config |
| `*` (model-specific) | Architecture-specific config overrides for DeepSeek, Qwen3, Nemotron, etc. |

## Design

```
ServerArgs
    │
    └─ ModelConfig.from_pretrained(model_path, server_args)
            │
            ├─ AutoConfig (HF)
            │       └─ patched by model_config_parser_registry if model_type is known
            │
            ├─ DeepSeekV4Config, NemotronHConfig, ... (architecture extensions)
            │
            └─ ModelConfig fields:
                   hf_config, model_arch, dtype, tp_size, context_len, ...
```

### Model-Specific Configs

Models with non-standard HF config fields (e.g. DeepSeek MLA dimensions, Nemotron SSM layers) subclass `PretrainedConfig` and register themselves in `model_config_parser_registry.py`. `ModelConfig` uses the registry to instantiate the right class when loading.

### LoadConfig

Determines how weights are loaded: safetensors vs. pickle, remote vs. local, quantization format (fp8, int4, etc.), and pre-sharding strategy for large models.

## Key Types

- `ModelConfig.model_arch` — string architecture name used to look up model class in `models/registry.py`
- `ModelConfig.AttentionArch` — enum distinguishing MHA / MLA / MHA-variant backends
- `ModelImpl` — enum selecting which model implementation to use (native, vLLM-compat, etc.)
