# layers — Neural Network Layer Implementations

## Purpose

Provides SGLang-specific implementations of neural network layers that replace or extend standard PyTorch/HuggingFace modules. Key goals: tensor-parallel correctness, KV-cache-aware attention, quantization support, and CUDA graph compatibility.

## Key Files

| File | Role |
|---|---|
| `radix_attention.py` | `RadixAttention` — attention layer that reads/writes the paged KV cache |
| `linear.py` | `ColumnParallelLinear`, `RowParallelLinear`, `QKVParallelLinear` — TP-split linear layers |
| `vocab_parallel_embedding.py` | `VocabParallelEmbedding`, `ParallelLMHead` — TP embedding/logit layers |
| `layernorm.py` | Fused RMSNorm / LayerNorm kernels |
| `activation.py` | Fused activation functions (SiLU, GELU, etc.) |
| `logits_processor.py` | Logits post-processing (temperature, repetition penalty, grammar masks) |
| `sampler.py` | Token sampling (top-k, top-p, greedy) |
| `model_parallel.py` | Utility functions for model-parallel layer construction |
| `communicator.py` | `Communicator` — fused scatter/gather for MoE dispatch |
| `dp_attention.py` | Data-parallel attention and CP (context parallelism) group management |
| `moe/` | MoE layer implementations (`fused_moe_triton/`, `token_dispatcher/`, etc.) |
| `attention/` | Attention backend implementations (FlashInfer, FlashAttention, DSA, Mamba, etc.) |
| `quantization/` | Quantization configs and kernels (fp8, int4, AWQ, GPTQ, etc.) |
| `pooler.py` | Embedding pooling (mean, last-token, CLS) for encoder models |
| `parameter.py` | `BasevLLMParameter` — parameter class used for TP weight sharding |
| `elementwise.py` | Fused elementwise ops |
| `conv.py` | 1D/2D convolution layers used by audio models |

## Architecture

```
models/llama.py (or any model file)
    │
    ├─ QKVParallelLinear   (layers/linear.py)
    ├─ RadixAttention       (layers/radix_attention.py)
    │       └─ AttnBackend.forward(q, k, v, forward_batch)
    │               ├─ FlashInferAttnBackend
    │               ├─ FlashAttentionBackend
    │               └─ DSAAttnBackend (paged attention custom kernel)
    ├─ RowParallelLinear    (layers/linear.py)
    └─ RMSNorm              (layers/layernorm.py)
```

### RadixAttention

The central attention layer — it does not implement attention math directly but delegates to an `AttnBackend`. Its key responsibility is mapping request KV positions through `ForwardBatch.req_to_token` to physical KV cache slots, enabling KV reuse across requests.

### Tensor Parallelism

All parallel layers (Linear, Embedding, LM Head) accept a `tp_size` and `tp_rank` at construction and automatically split weight dimensions. During forward, they call `all_reduce` (via `communication_op.py`) where needed.

### Quantization

Quantized layers subclass the standard parallel layers and override `create_weights()` and `apply_weights()`. The quantization config (`layers/quantization/`) is injected by `ModelRunner` based on the loaded checkpoint's quant method.

### MoE Layers

`layers/moe/` contains:
- `FusedMoE` — top-level MoE layer combining gating + expert dispatch + FFN
- `fused_moe_triton/` — Triton kernel implementations for expert computation
- `token_dispatcher/` — all-to-all token routing backends (DeepEP, Mooncake, Mori, Nixl)
