# platforms — Hardware Platform Abstraction

## Purpose

Provides a thin abstraction layer over hardware platforms (CUDA, ROCm/HIP, CPU, NPU, XPU, MUSA). Each platform exposes a common interface so the rest of SGLang can write platform-agnostic code.

## Key Files

| File | Role |
|---|---|
| `interface.py` | `Platform` ABC — defines the platform contract |
| `device_mixin.py` | `DeviceMixin` — shared device-agnostic utilities |
| `cuda.py` | `CudaPlatform` — NVIDIA CUDA implementation |
| `rocm.py` | `RocmPlatform` — AMD ROCm / HIP implementation |
| `__init__.py` | `current_platform` singleton — auto-detected at import time |

## Design

```
from sglang.srt.platforms import current_platform

current_platform.is_cuda()          → bool
current_platform.get_device_name()  → str
current_platform.get_memory_info()  → (total, free)
current_platform.supports_fp8()     → bool
current_platform.device_type        → "cuda" | "rocm" | "cpu" | "npu" | ...
```

At import time, `__init__.py` detects the available hardware and sets `current_platform` to the appropriate subclass. Code throughout SGLang (`distributed/`, `layers/`, `model_executor/`) queries `current_platform` rather than calling `torch.cuda.is_available()` directly.

### Platform Detection Order

1. NPU (`torch_npu` available) → NpuPlatform
2. XPU (`intel_extension_for_pytorch` available) → XpuPlatform
3. ROCm (`torch.version.hip` is set) → RocmPlatform
4. MUSA → MusaPlatform
5. CPU fallback → CpuPlatform
6. Default → CudaPlatform
