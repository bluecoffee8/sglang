# SGLang Kernel Contribution Quick Reference

## 🎯 Top 5 Recommended Starter Issues

### 1. NewGELU CUDA Kernel (Beginner ⭐)
```
Effort: ~1-2 days
Complexity: Low
Impact: High
Location: python/sglang/srt/layers/activation.py (TODO line)
Scope: Create sgl-kernel/csrc/elementwise/newgelu.cu

What to do:
- Implement unfused NewGELU: out = x * (1 + erf(x/sqrt(2)))/2
- Implement fused variant: newgelu(x) + residual
- Add Python wrapper
- Add benchmark in sgl-kernel/benchmark/
```

### 2. FP8 Support for Attention Merge (Intermediate ⭐⭐)
```
Effort: ~2-3 days
Complexity: Medium
Impact: High
Location: sgl-kernel/python/sgl_kernel/attention.py (lines 16-18)
File to modify: sgl-kernel/csrc/attention/merge_attn_states.cu

What to do:
- Add float8_e4m3fn dtype support to merge_state_v2
- Handle scale conversions for FP8
- Add test cases for FP8 inputs
- Benchmark against PyTorch fallback
```

### 3. Top-K Kernel for Large Vocab (Intermediate ⭐⭐)
```
Effort: ~2-3 days
Complexity: Medium
Impact: Medium
Location: sgl-kernel/python/sgl_kernel/top_k.py (TODO)
File to enhance: sgl-kernel/csrc/elementwise/topk.cu

What to do:
- Extend existing topk.cu to handle k > 512
- Implement merge-based approach for large k
- Add support for FP8/INT8 inputs
- Compare performance vs PyTorch topk
```

### 4. Concat + Projection Fusion (Intermediate ⭐⭐)
```
Effort: ~2-3 days
Complexity: Medium
Impact: Medium
Location: sgl-kernel/csrc/elementwise/concat_mla.cu (has TODOs)
File to complete: concat_mla.cu

What to do:
- Clean up concat_mla_absorb_q naming
- Implement full optimization for concat + linear projection
- Support variable tensor dimensions
- Add performance benchmarks
```

### 5. RoPE + Quantization Fusion (Advanced ⭐⭐⭐)
```
Effort: ~3-4 days
Complexity: High
Impact: High (20-30% speedup)
Files involved: 
- sgl-kernel/csrc/elementwise/pos_enc.cu
- sgl-kernel/csrc/gemm/per_token_quant_fp8.cu
- quantization directory

What to do:
- Create new kernel: apply_rope_and_quantize
- Support multiple precision combinations
- Fuse RoPE rotation + scale + quantization
- Benchmark against 2-kernel approach
```

---

## 📂 Key File Map for Different Contribution Types

### Adding a Simple Operation
```
Template workflow:
1. sgl-kernel/csrc/elementwise/my_op.cu          (CUDA kernel)
2. sgl-kernel/csrc/common_extension.cc           (register with torch)
3. sgl-kernel/include/sgl_kernel_ops.h           (declare)
4. sgl-kernel/python/sgl_kernel/elementwise.py   (Python wrapper)
5. sgl-kernel/tests/test_my_op.py                (tests)
6. sgl-kernel/benchmark/bench_my_op.py           (benchmark)

Use skill: /add-jit-kernel for simpler kernels
```

### Adding a MOE Kernel
```
Files to modify:
1. sgl-kernel/csrc/moe/my_moe_op.cu              (CUDA)
2. sgl-kernel/python/sgl_kernel/moe.py           (Python binding)
3. python/sglang/srt/layers/moe/                 (Integration)
4. sgl-kernel/benchmark/bench_moe_my_op.py       (Benchmark)

Reference implementations:
- moe_fused_gate.cu (20KB) - gate + expert selection fusion
- fused_qknorm_rope_kernel.cu (16KB) - query normalization + rope fusion
- moe_topk_softmax_kernels.cu (32KB) - top-k + softmax fusion
```

### Adding Quantization Kernel
```
Files involved:
1. sgl-kernel/csrc/quantization/my_quant.cu     (CUDA kernel)
2. sgl-kernel/csrc/gemm/per_token_quant_*.cu    (if GEMM-related)
3. sgl-kernel/python/sgl_kernel/gemm.py         (Python wrapper)
4. python/sglang/srt/layers/quantization/       (Integration)

Existing patterns:
- per_token_group_quant_8bit.cu (542 lines)
- fp8_gemm_kernel.cu (1532 lines)
- GPTQ kernel (1950 lines)

Note: Quantization kernels tend to be large due to template specialization
```

### Adding Communication Kernel (AllReduce, etc)
```
Files to modify:
1. sgl-kernel/csrc/allreduce/my_collective.cu   (CUDA)
2. sgl-kernel/python/sgl_kernel/allreduce.py    (Python wrapper)
3. python/sglang/srt/layers/communicator.py     (Integration point)

Existing patterns:
- custom_all_reduce.cuh (26KB) - main implementation
- quick_all_reduce.cuh (24KB) - performance-optimized variant
- mscclpp_allreduce.cuh (31KB) - MSCCL++ integration

Challenge: Requires testing on multi-GPU setup or use NCCL mocks
```

---

## 🔬 Code Patterns to Know

### CUDA Kernel Registration (common_extension.cc)
```cpp
m.def(
  "my_kernel(Tensor input, Tensor output, int param) -> ()");
m.impl("my_kernel", torch::kCUDA, &my_kernel_impl);
```

### Python Wrapper Pattern (sgl_kernel/*.py)
```python
def my_kernel(input: torch.Tensor, param: int) -> torch.Tensor:
    output = torch.empty_like(input)
    torch.ops.sgl_kernel.my_kernel.default(input, output, param)
    return output
```

### Test Pattern
```python
@pytest.mark.skipif(
    not is_sm90_supported(),
    reason="Requires compute capability 9.0+"
)
def test_my_kernel():
    # Test implementation
```

### Benchmark Pattern (triton.testing recommended)
```python
from triton.testing import do_bench_cudagraph

@triton.testing.perf_report([...])
def bench_my_kernel():
    torch.cuda.reset_peak_memory_stats()
    ms = do_bench_cudagraph(lambda: my_kernel(...))
    return ms
```

---

## 🏗️ Architecture Support Matrix

| Feature | CUDA | ROCM | MUSA | Metal | CPU |
|---------|------|------|------|-------|-----|
| FP8 GEMM | ✅ | ⚠️ | ⚠️ | ❌ | ❌ |
| INT8 | ✅ | ✅ | ⚠️ | ❌ | ✅ |
| AllReduce | ✅ | ✅ | ❌ | N/A | N/A |
| RoPE | ✅ | ✅ | ⚠️ | ❌ | ⚠️ |
| Sparse | ❌ | ❌ | ❌ | ❌ | ❌ |

Legend: ✅=Full support, ⚠️=Partial/Needs work, ❌=Missing

---

## 📊 Kernel Size Reference

| Kernel | Lines | Use Case | Notes |
|--------|-------|----------|-------|
| gptq_kernel.cu | 1950 | GPTQ quantization | Large due to many template instances |
| fp8_blockwise_moe.cu | 810 | MOE FP8 | Complex parameter combinations |
| moe_topk_softmax.cu | 822 | MOE routing | Well-optimized, reference quality |
| moe_fused_gate.cu | 523 | MOE gate + select | Good fusion example |
| per_token_group_quant_8bit_v2.cu | 542 | Per-token quantization | Good starting reference |
| topk.cu | 546 | Top-K selection | Can be extended for large K |
| causal_conv1d.cu | 669 | Mamba support | Partially complete |

*Tip: Kernels >1KB are often template-heavy. Look for specialization opportunities.*

---

## 🧪 Testing Infrastructure

### Run specific test
```bash
cd sgl-kernel
pytest tests/test_my_kernel.py -xvs
```

### Run all kernel tests
```bash
cd sgl-kernel
pytest tests/ -x
```

### Run benchmark
```bash
cd sgl-kernel
python benchmark/bench_my_kernel.py
```

### Build from source
```bash
cd sgl-kernel
make build  # Uses all CPUs
make build MAX_JOBS=2  # Limit parallelism
```

---

## 🎓 References & Learning

### Study These Before Contributing
1. **Attention kernels**: `attention/merge_attn_states.cu` & `cutlass_mla_kernel.cu`
2. **Fusion patterns**: `moe/moe_fused_gate.cu` & `moe/fused_qknorm_rope_kernel.cu`
3. **AllReduce**: `allreduce/custom_all_reduce.cuh` & `quick_all_reduce.cuh`
4. **Quantization**: `gemm/per_token_group_quant_8bit_v2.cu` & `fp8_gemm_kernel.cu`

### External Resources
- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass) - for GEMM kernels
- [Triton Documentation](https://triton-lang.org/) - Triton kernel examples in repo
- [NVIDIA Collective Communication Library](https://github.com/NVIDIA/nccl) - AllReduce patterns
- [ModernGPU](https://moderngpu.github.io/) - GPU algorithm patterns

---

## ⚠️ Common Pitfalls to Avoid

1. **Memory efficiency**: Test with actual batch sizes (not tiny ones)
2. **Precision handling**: FP8 scale management; don't forget conversions
3. **Thread block size**: Usually 128-256 threads; profile to find sweet spot
4. **Register pressure**: Complex fusions can hit register limits → spills
5. **Atomics**: Avoid on newer architectures; use reduce_max/shuffle instead
6. **Template bloat**: Each type combination = new kernel binary; use overloads

---

## 🚀 Submission Checklist

Before creating a PR:
- [ ] Code compiles without warnings
- [ ] Tests pass (`pytest tests/test_*.py -x`)
- [ ] Benchmarks included and show improvement
- [ ] Python wrapper complete
- [ ] Registered in `common_extension.cc` and header file
- [ ] Docstring explains the kernel purpose
- [ ] Handles edge cases (empty tensors, small sizes, etc.)
- [ ] Works on CUDA (minimum requirement)
- [ ] Memory usage analyzed (peak memory during benchmark)

---

## 📞 Questions & Support

- **SGLang Issues**: Check [GitHub issues](https://github.com/sgl-project/sglang)
- **Kernel-specific**: Look at related kernel implementations in csrc/
- **Architecture help**: Refer to CLAUDE.md in project root for project-specific guidance
- **Performance tips**: See benchmark examples for profiling best practices

