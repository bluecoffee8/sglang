# sampling — Token Sampling

## Purpose

Defines sampling hyperparameters and the per-batch sampling state used during decoding. Implements temperature scaling, top-k/p/min-p filtering, repetition penalties, and custom logit processors.

## Key Files

| File | Role |
|---|---|
| `sampling_params.py` | `SamplingParams` — user-facing sampling configuration (temperature, top_p, stop, etc.) |
| `sampling_batch_info.py` | `SamplingBatchInfo` — GPU tensor representation of sampling params for a batch |
| `custom_logit_processor.py` | `CustomLogitProcessor` ABC — interface for user-supplied logit transforms |

## Design

```
Request(temperature=0.7, top_p=0.9, max_new_tokens=100)
    │
    └─ SamplingParams (validated, stored in Req)
            │
    Scheduler assembles batch
            │
            └─ SamplingBatchInfo.from_batch(batch)
                    ├─ temperatures tensor      [batch_size]
                    ├─ top_p tensor             [batch_size]
                    ├─ top_k tensor             [batch_size]
                    ├─ presence_penalties       [batch_size]
                    └─ per-request stop strings / stop token IDs

    ModelRunner forward pass → logits [batch_size, vocab_size]
            │
            └─ LogitsProcessor(logits, sampling_batch_info)
                    ├─ apply temperature
                    ├─ apply repetition/frequency/presence penalties
                    ├─ apply grammar masks (if constrained decoding)
                    ├─ apply custom logit processors
                    └─ sample(top_k, top_p, min_p) → next_token_ids
```

### SamplingParams

Key fields:
- `temperature` — controls randomness (0 = greedy, 1 = standard)
- `top_p`, `top_k`, `min_p` — nucleus / top-k / min-p filtering
- `frequency_penalty`, `presence_penalty`, `repetition_penalty` — diversity penalties
- `stop`, `stop_token_ids` — early stopping conditions
- `max_new_tokens`, `min_new_tokens` — length bounds
- `n` — number of completions to generate
- `json_schema`, `regex`, `ebnf` — constrained decoding specifications

### SamplingBatchInfo

Converts `SamplingParams` from Python objects to GPU tensors for efficient batch processing. The `LogitsProcessor` in `layers/logits_processor.py` reads from this to apply sampling transforms in a single vectorized kernel.

### Custom Logit Processors

Users can supply a `CustomLogitProcessor` subclass via `custom_params` to implement arbitrary logit transforms (e.g., classifier-free guidance, watermarking). The processor's `__call__(logits, batch_info)` is invoked before sampling.
