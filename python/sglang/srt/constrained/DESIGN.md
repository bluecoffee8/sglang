# constrained — Constrained Decoding

## Purpose

Enables structured output generation by applying grammar constraints to the token sampling process. Supports JSON schema, EBNF grammars, regex, and thinking-mode reasoning constraints. Multiple backend engines are supported and selected at runtime.

## Key Files

| File | Role |
|---|---|
| `base_grammar_backend.py` | `BaseGrammarBackend` ABC and `GrammarObject` base |
| `grammar_manager.py` | `GrammarManager` — per-scheduler orchestrator for grammar compilation and request lifecycle |
| `xgrammar_backend.py` | XGrammar backend (recommended, CUDA-accelerated) |
| `llguidance_backend.py` | LLGuidance backend |
| `outlines_backend.py` | Outlines backend |
| `outlines_jump_forward.py` | Jump-forward decoding optimization for Outlines |
| `reasoner_grammar_backend.py` | Thin backend for reasoning / thinking-budget mode |
| `utils.py` | Helpers (schema normalization, regex conversion) |

## Design

```
Scheduler
    │
    └─ GrammarManager
            │
            ├─ grammar_backend (XGrammar | LLGuidance | Outlines | Reasoner)
            │       └─ compile(json_schema | regex | ebnf) → GrammarObject
            │
            ├─ grammar_queue: [Req]   (async compilation in-flight)
            │
            └─ per step:
                    for req in batch:
                        if req.grammar:
                            logit_bias = req.grammar.get_next_token_bitmask()
                            apply to logits before sampling
```

### Grammar Lifecycle

1. Request arrives with `json_schema`, `regex`, or `ebnf` constraint.
2. `GrammarManager` submits async compilation to the backend threadpool.
3. Compiled `GrammarObject` is attached to the `Req`.
4. At each decode step, `GrammarObject.get_next_token_bitmask()` returns an allowed-token bitmask.
5. The bitmask is merged with logits in `logits_processor.py` before sampling.
6. On token acceptance, `GrammarObject.accept_token(token_id)` advances the automaton state.

### Backend Selection

Controlled by `--grammar-backend` server arg. XGrammar is the default because it runs mask computation on GPU, avoiding CPU→GPU transfers per step. All backends implement the same `BaseGrammarBackend` interface.

## Sync Coordination

For data-parallel deployments, grammar compilation results are broadcast from the entry rank to all DP replicas via `grammar_sync_group` to keep all ranks in sync.
