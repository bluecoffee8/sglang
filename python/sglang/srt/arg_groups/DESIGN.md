# arg_groups — Argument Groups & Model-Specific CLI Hooks

## Purpose

Defines reusable argument groups for the server CLI and per-model argument injection hooks. Argument groups cluster related flags so they can be shared across entry points (HTTP server, engine, Ray actor) without duplicating `argparse` code. Model-specific hooks let individual architectures inject their own flags and post-parse validation logic.

## Key Files

| File | Role |
|---|---|
| `argparse_actions.py` | Custom `argparse.Action` subclasses for complex flag types |
| `deepseek_v4_hook.py` | DeepSeek-V4 specific arg defaults and validation |
| `hisparse_hook.py` | HiSparse sparse-attention arg injection |
| `nemotron_h_hook.py` | Nemotron-H hybrid SSM arg injection |
| `pd_disaggregation_hook.py` | Prefill-decode disaggregation arg injection |
| `speculative_hook.py` | Speculative decoding (EAGLE, MTP, N-gram) arg injection |

## Design

```
ServerArgs (server_args.py)
       │
       ├─ add_argument_group(...)   ← standard groups (model, memory, parallel, ...)
       │
       └─ model-specific hooks
              ├─ deepseek_v4_hook   (e.g. --mla-type, --dp-attention)
              ├─ hisparse_hook      (e.g. --hisparse-layers)
              ├─ nemotron_h_hook    (e.g. --nemotron-h-ssm-backend)
              ├─ pd_disaggregation_hook (e.g. --disaggregation-mode)
              └─ speculative_hook   (e.g. --speculative-algo, --num-draft-tokens)
```

Each hook is a module with a single entry point, typically `add_args(parser)` plus optional `post_parse(server_args)` for cross-flag validation. The `ServerArgs` class in `server_args.py` calls hooks based on detected model type or explicit flags, keeping the main arg parser file small.

## Extension Points

Add a new hook module, register it in the hook discovery list inside `server_args.py`, and implement `add_args` / `post_parse`. No changes to existing hooks are required.
