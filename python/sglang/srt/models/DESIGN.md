# models — Model Architecture Implementations

## Purpose

Contains one Python file per supported model architecture (or model family). Each file implements the model's `forward()` method using SGLang's parallel layers, and registers the model with the global registry. This is the largest directory in `srt/` by file count.

## Key Files

| File | Role |
|---|---|
| `registry.py` | Maps `model_type` string → model class; `ModelRegistry.get_model_cls()` |
| `utils.py` | Shared helpers: `default_weight_loader`, `make_layers`, causality masks, etc. |
| `llama.py` | LLaMA / LLaMA-2 / LLaMA-3 decoder |
| `deepseek_v4.py` | DeepSeek-V4 MoE (MLA attention + MoE FFN) |
| `qwen3.py`, `qwen3_moe.py` | Qwen3 dense and MoE variants |
| `gemma4_unified.py` | Gemma4 (unified dense/MoE multi-modal) |
| `mistral.py`, `mixtral.py` | Mistral dense and Mixtral MoE |
| `transformers.py` | Generic HF Transformers compatibility shim |
| `*_eagle.py` | EAGLE draft models for speculative decoding |
| `*_nextn.py` | MTP (Multi-Token Prediction) next-token head models |
| `bert.py`, `roberta.py` | Encoder-only models |
| `llama_embedding.py` | LLaMA embedding (encoder) variant |
| `*_vl*.py`, `*_mm*.py` | Vision-language and multi-modal models |

## Model Registration Pattern

```python
# Every model file ends with:
EntryClass = LlamaForCausalLM  # or list for multiple arches

# registry.py discovers this and builds:
_MODEL_REGISTRY: dict[str, type] = {
    "LlamaForCausalLM": LlamaForCausalLM,
    "DeepseekV3ForCausalLM": DeepseekV3ForCausalLM,
    ...
}
```

`ModelConfig.model_arch` (parsed from HF config's `architectures` field) is the key used to look up the class.

## Standard Model Structure

```python
class LlamaForCausalLM(nn.Module):
    def __init__(self, config, quant_config, ...):
        self.model = LlamaModel(...)   # stacked transformer layers
        self.lm_head = ParallelLMHead(...)

    def forward(self, input_ids, positions, forward_batch):
        hidden = self.model(input_ids, positions, forward_batch)
        return self.lm_head(hidden)

    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]):
        # maps HF weight names → this model's parameters
        ...
```

## Speculative Draft Models

Files ending in `_eagle.py` or `_nextn.py` implement lightweight draft models used by speculative decoding workers. They share backbone weights with the target model but have smaller or absent LM heads, and are structured to support CUDA graph capture with `eagle_draft_cuda_graph_runner.py`.

## Multi-Modal Models

Vision-language models (e.g., `llava.py`, `internvl.py`, `qwen2_vl.py`) add:
- A visual encoder (`clip.py`, `siglip.py`, etc.)
- Image/video token projection layer
- Overridden `forward()` that interleaves visual and text tokens

Visual encoder CUDA graph capture is handled by `multimodal/vit_cuda_graph_runner.py`.
