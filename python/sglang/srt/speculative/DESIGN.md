# speculative — Speculative Decoding

## Purpose

Accelerates autoregressive decoding by predicting multiple future tokens with a cheap draft model, then verifying them in a single pass with the large target model. Accepted tokens are produced at a fraction of the cost of individual target model steps.

## Key Files

| File | Role |
|---|---|
| `base_spec_worker.py` | `BaseSpecWorker`, `BaseDraftWorker` ABCs — interface all spec implementations share |
| `spec_info.py` | `SpecInput`, `SpecOutput` — per-step draft/verify data structures |
| `spec_registry.py` | Maps `--speculative-algo` name → worker class |
| `spec_utils.py` | Shared accept/reject logic, tree-based acceptance |
| `eagle_worker_v2.py` | `EagleWorkerV2` — EAGLE / EAGLE-2 speculative worker |
| `eagle_info.py`, `eagle_info_v2.py` | EAGLE-specific draft state structures |
| `eagle_utils.py` | EAGLE attention mask construction, draft tree building |
| `eagle_draft_cuda_graph_runner.py` | CUDA graph runner for EAGLE draft model |
| `eagle_draft_extend_cuda_graph_runner.py` | EAGLE draft extend (for long prefixes) |
| `multi_layer_eagle_worker_v2.py` | Multi-layer EAGLE (deeper draft trees) |
| `frozen_kv_mtp_worker_v2.py` | MTP (Multi-Token Prediction) worker with frozen KV |
| `frozen_kv_mtp_info.py` | MTP-specific draft info |
| `dflash_worker.py` | DFlash speculative worker |
| `ngram_worker.py` | N-gram draft worker (CPU-only, no draft model) |
| `external_corpus_manager.py` | External corpus for n-gram draft lookup |
| `adaptive_runtime_state.py` | `AdaptiveRuntimeState` — dynamically adjusts draft steps per batch |
| `adaptive_spec_params.py` | Adaptive speculative decoding parameters |
| `standalone_worker_v2.py` | Standalone (model-only) spec worker |
| `eagle_disaggregation.py` | EAGLE support in disaggregated prefill-decode mode |

## Design

```
Scheduler
    │
    └─ SpecWorker (e.g. EagleWorkerV2)
            │
            ├─ draft_worker (small/fast model)
            │       └─ draft_worker.draft(target_batch)
            │               └─ produces candidate_token_ids [batch, num_draft_tokens]
            │
            └─ target_worker (large/accurate model)
                    └─ target_worker.verify(target_batch, candidate_ids)
                            ├─ compute P(candidate | context) for each candidate
                            └─ accept/reject per token using rejection sampling
                                    → accepted_tokens (variable length)
```

### EAGLE Algorithm

EAGLE drafts tokens using a lightweight model that takes the target model's last hidden states (not just logits) as additional input. This gives the draft model much higher accuracy than a separate smaller model.

1. Target runs one step → hidden states + logits
2. EAGLE draft runs `num_draft_tokens` steps using target's hidden states
3. Target verifies all draft tokens in a single batched forward pass
4. Tokens accepted up to the first rejection are output

### MTP (Multi-Token Prediction)

`frozen_kv_mtp_worker_v2.py` implements speculative decoding for models with dedicated next-token head branches (e.g., DeepSeek-V3 MTP heads). The draft heads share the backbone KV cache and add minimal latency.

### Adaptive Step Control

`AdaptiveRuntimeState` monitors accept rates per batch size and adjusts `num_draft_tokens` up or down to maximize throughput. Small batches can support more draft steps; large batches benefit from fewer.

### N-gram Draft

`NgamWorker` needs no separate model — it predicts the next tokens by looking up the last N tokens in an n-gram table built from the prompt or a provided corpus. Zero draft model overhead.
