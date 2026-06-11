# compilation — Torch Compile & JIT Backend

## Purpose

Wraps `torch.compile` to provide graph-level optimizations for model forward passes. Implements piecewise compilation (separate graphs per distinct operator group) to avoid recompilation when only part of the computation graph changes between prefill and decode phases.

## Key Files

| File | Role |
|---|---|
| `compile.py` | Top-level `compile_forward()` — entry point to trigger compilation |
| `backend.py` | Custom `torch.compile` backend dispatcher |
| `compilation_config.py` | `CompilationConfig` — flags, split points, captured shapes |
| `compilation_counter.py` | Tracks compilation invocations for debugging/metrics |
| `compile_phase.py` | Phase enum (WARMUP, CAPTURE, EXECUTE) |
| `compiler_interface.py` | Abstract interface between ModelRunner and backend |
| `cuda_piecewise_backend.py` | CUDA piecewise graph capture & replay |
| `npu_piecewise_backend.py` | NPU variant of piecewise backend |
| `pass_manager.py` | Ordered sequence of FX graph transformation passes |
| `inductor_pass.py` | Inductor-specific optimization passes |
| `fix_functionalization.py` | Fixes functionalization artifacts from `torch.compile` |
| `fx_utils.py` | FX graph utility helpers |
| `torch_compile_decoration.py` | `@sgl_torch_compile` decorator for model methods |
| `weak_ref_tensor.py` | Weak-reference wrapper to avoid tensor reference cycles |

## Design

```
@sgl_torch_compile
def model_forward(batch):
    ...

                ┌─ CompilationConfig
                │    (split ops, shapes, flags)
                │
compile_forward()
       │
       ├─ pass_manager.run(fx_graph)
       │       ├─ fix_functionalization
       │       └─ inductor_pass (optional)
       │
       └─ Backend.select()
               ├─ CUDAPiecewiseBackend   (piecewise CUDA graph capture)
               │       ├─ capture(warmup_batch)
               │       └─ replay(forward_batch)
               └─ NPUPiecewiseBackend
```

### Piecewise Compilation

The key innovation is splitting the FX graph at designated "split points" registered via `register_split_op()` (typically all-reduce boundaries). Each piece is compiled independently into a CUDA graph. At runtime the pieces are spliced back together, allowing shape-invariant graph regions to be reused across batch size changes.

## Configuration

Controlled by `--enable-torch-compile`, `--torch-compile-max-bs`, and `--cuda-graph-*` server args. The `CompilationConfig` is shared between the compilation infrastructure and `parallel_state.py` (which registers split ops at all-reduce sites).
