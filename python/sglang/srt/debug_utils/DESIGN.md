# debug_utils — Debugging & Introspection Utilities

## Purpose

Provides tools for diagnosing correctness issues, capturing model internals, and comparing outputs across runs or ranks. Intended for developer use during debugging, not production inference.

## Key Files

| File | Role |
|---|---|
| `dumper.py` | `Dumper` — selectively dumps intermediate tensors to disk during forward pass |
| `tensor_dump_forward_hook.py` | PyTorch forward hook that triggers `Dumper` at specified layers |
| `dump_loader.py` | Loads and deserializes previously dumped tensors |
| `dump_comparator.py` | Compares two dump sets (e.g. baseline vs. PR) tensor-by-tensor |
| `text_comparator.py` | Compares generated text outputs across runs |
| `log_parser.py` | Parses structured SGLang log lines for post-hoc analysis |
| `cuda_coredump.py` | Triggers CUDA core dumps for GPU crash analysis |
| `model_truncator.py` | Truncates a model to N layers for faster debug iteration |
| `pr_fix_toggle.py` | Env-flag-controlled toggle to revert specific PR fixes for A/B testing |

## Design

```
Forward pass
    │
    └─ tensor_dump_forward_hook (if SGLANG_DUMP_LAYERS set)
            │
            └─ Dumper.dump(tensor, layer_name, step)
                    │
                    └─ writes to /tmp/sglang_dump/<run_id>/<step>/<layer>.pt

Later:
    DumpComparator(run_a_dir, run_b_dir).compare()
        └─ per-layer abs/rel diff, flagging divergences
```

### Tensor Dumping

Set `SGLANG_DUMP_LAYERS=layer_name_pattern` and `SGLANG_DUMP_DIR=/path` to activate. The forward hook registers on matching layers and captures inputs/outputs. Dumps are keyed by `(step, layer_name)` for alignment across runs.

### Model Truncator

`ModelTruncator(n_layers=4)` is used in tests and debug sessions to slice the model to fewer transformer layers, reducing memory and iteration time while preserving the shape pipeline.
