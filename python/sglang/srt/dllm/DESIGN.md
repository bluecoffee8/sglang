# dllm — Diffusion LLM Support

## Purpose

Provides configuration and scheduler mixin support for diffusion-based language models (dLLMs), which generate tokens through iterative denoising rather than autoregressive decoding. This is distinct from standard autoregressive generation and requires different scheduling logic.

## Key Files

| File | Role |
|---|---|
| `config.py` | `DllmConfig` — dLLM-specific hyperparameters (noise schedule, diffusion steps, etc.) |

## Design

dLLM support is structured as a mixin injected into `Scheduler`:

```
Scheduler
    ├─ SchedulerDllmMixin  (from dllm/mixin/scheduler.py)
    │       └─ overrides step() to run N denoising iterations instead of 1 AR step
    │
    └─ DllmConfig  (from dllm/config.py)
            └─ noise_schedule, num_diffusion_steps, mask_token_id, ...
```

The `SchedulerDllmMixin` replaces the single-token forward pass with a multi-step denoising loop. At each denoising step, a full batch forward is run with masked tokens; the model predicts un-masked tokens; tokens above a confidence threshold are unmasked and fixed. This repeats for `num_diffusion_steps`.

## Status

Early-stage / experimental. Most dLLM model implementations (e.g. `models/llada2.py`) are independently complete; this module holds the shared config and scheduler plumbing.
