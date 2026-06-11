# hardware_backend — Hardware Backend Abstraction

## Purpose

Placeholder / extension point for hardware-specific backend implementations beyond CUDA and ROCm. The primary per-device abstraction lives in `platforms/`; `hardware_backend/` is reserved for heavier backend code (e.g., custom kernels or runtime integrations) that is too large to include in `platforms/`.

## Design

Currently this directory contains no Python source files. Hardware-specific logic is split between:

- `platforms/` — lightweight platform detection and device configuration (`cuda.py`, `rocm.py`, `interface.py`)
- `layers/` — custom CUDA/Triton kernels for individual operators

As new hardware targets (e.g., TPU, XPU custom runtimes) require more substantial integration code, that code should be placed here under a subdirectory per target.
