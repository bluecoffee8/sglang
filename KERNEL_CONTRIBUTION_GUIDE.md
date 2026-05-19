# SGLang Kernel Contribution Guide: Choose Your Path

This guide helps you find the kernel contribution opportunity that matches your skill level and interests.

---

## 🎯 Choose by Experience Level

### If You're New to Kernel Development

**Start with**: `NewGELU CUDA Kernel`
- **Why**: Single, self-contained operation; clear performance comparison
- **Time**: 1-2 days
- **Files**: Create `sgl-kernel/csrc/elementwise/newgelu.cu` (~150 lines)
- **Learning**: Thread patterns, block layout, PyTorch torch.ops integration
- **Difficulty**: ⭐ Beginner

**Next**: `Top-K Kernel Improvements`
- **Why**: Builds on existing kernel; extend rather than create new
- **Time**: 2-3 days
- **Files**: Enhance `sgl-kernel/csrc/elementwise/topk.cu`
- **Learning**: Merge-based algorithms, template specialization, performance tuning
- **Difficulty**: ⭐⭐ Intermediate

### If You Have GPU/CUDA Experience

**Start with**: `FP8 Support in Attention Kernels`
- **Why**: Extends well-tested code; immediate high-value impact
- **Time**: 2-3 days
- **Files**: Modify `sgl-kernel/csrc/attention/merge_attn_states.cu`
- **Learning**: Type handling, FP8 semantics, attention patterns
- **Difficulty**: ⭐⭐ Intermediate

**Next**: `RoPE + Quantization Fusion`
- **Why**: Combines two operations; practical impact on LLM inference
- **Time**: 3-4 days
- **Files**: New kernel fusing `pos_enc.cu` + quantization logic
- **Learning**: Kernel fusion, register pressure management, dual-operation coordination
- **Difficulty**: ⭐⭐ Intermediate

### If You Have Distributed Systems Experience

**Start with**: `Fused Reduce-Scatter for MOE`
- **Why**: Solves real distributed inference bottleneck
- **Time**: 3-4 days
- **Files**: New `allreduce/reduce_scatter_moe.cu` kernel
- **Learning**: Collective communication patterns, MOE token distribution, scheduling
- **Difficulty**: ⭐⭐⭐ Advanced

**Next**: `NVSHMEM AllReduce`
- **Why**: Cutting-edge communication pattern; enables next-gen multi-GPU setups
- **Time**: 1+ week
- **Files**: New `allreduce/nvshmem_allreduce.cu` + integration
- **Learning**: Symmetric memory, NVSHMEM API, GPU memory topology
- **Difficulty**: ⭐⭐⭐ Advanced

### If You're a Kernel Optimization Expert

**Consider**: `Selective Scan for Mamba`
- **Why**: Novel algorithm; almost no open-source GPU implementations
- **Time**: 2+ weeks (if you have reference papers/code)
- **Files**: New `sgl-kernel/csrc/mamba/selective_scan.cu`
- **Learning**: State space model algorithms, numerically stable reductions
- **Difficulty**: ⭐⭐⭐⭐ Expert

**Or**: `SM100 Specialized Optimizations`
- **Why**: New GPU architecture; minimal existing code
- **Time**: 1-2 weeks
- **Files**: Enhance existing kernels with SM100-specific features
- **Learning**: Hopper+ architecture features, atomic-free collectives, warp scheduling
- **Difficulty**: ⭐⭐⭐ Advanced

---

## 🔍 Choose by Interest Area

### 🧮 I want to work on QUANTIZATION
```
Interest: Quantization
        ↓
    Experience with FP8?
    ├─ Yes → FP8 support in attention [2-3 days, ⭐⭐]
    └─ No  → NewGELU first [1-2 days, ⭐]
             then FP8 attention [2-3 days, ⭐⭐]
        ↓
    Ready for fusion? → RoPE + Quant fusion [3-4 days, ⭐⭐]
        ↓
    Advanced? → Group quant optimizations [varies, ⭐⭐⭐]
```

### 🔗 I want to work on DISTRIBUTED/COMMUNICATION
```
Interest: Communication
        ↓
    Know NCCL/MSCCL?
    ├─ Yes → Fused Reduce-Scatter [3-4 days, ⭐⭐⭐]
    └─ No  → Study existing allreduce code first
        ↓
    Ready for latest tech? → NVSHMEM AllReduce [1+ week, ⭐⭐⭐]
        ↓
    Time investment? → Allreduce + Compute Fusion [1-2 weeks, ⭐⭐⭐]
```

### 🚀 I want high PERFORMANCE impact
```
Interest: Performance
        ↓
    Type of improvement?
    ├─ Fused ops       → RoPE + Quant [3-4 days, ⭐⭐] or
    │                    AllReduce + RMSNorm [2-3 weeks, ⭐⭐⭐]
    ├─ New operation   → NewGELU [1-2 days, ⭐] or
    │                    Top-K improvements [2-3 days, ⭐⭐]
    └─ Communication   → Fused Reduce-Scatter [3-4 days, ⭐⭐⭐]
```

### 🏗️ I want to BUILD something NEW
```
Interest: Building new capability
        ↓
    What architecture?
    ├─ Mamba/SSM        → Selective Scan kernel [2+ weeks, ⭐⭐⭐⭐]
    ├─ Sparse models    → Sparse GEMM operations [2-3 weeks, ⭐⭐⭐]
    ├─ Emerging GPU     → SM100 optimizations [1-2 weeks, ⭐⭐⭐]
    └─ Multi-backend    → ROCm/MUSA quantization [1-2 weeks, ⭐⭐⭐]
```

### 🔧 I want QUICK wins (< 1 week)
```
Quick wins available:
1. NewGELU CUDA kernel           [1-2 days, ⭐, Medium impact]
2. Top-K improvements            [2-3 days, ⭐⭐, Medium-High impact]
3. Concat-MLA optimization       [2-3 days, ⭐⭐, Medium impact]
4. Small FP8 additions           [2-3 days, ⭐⭐, Medium impact]
```

---

## 📋 Detailed Opportunity Profiles

### Profile 1: NewGELU CUDA Kernel ✨ BEST FOR BEGINNERS

**Fit If You**:
- Want to learn kernel development
- Have CUDA basics down
- Want quick feedback loop
- Don't need multi-GPU complexity

**What You'll Build**:
```
Before:
  hidden = torch.nn.functional.gelu(x, approximate='tanh')  # Python slow

After:
  hidden = torch.ops.sgl_kernel.newgelu(x)  # CUDA fast
  # ~20% speedup for activation-heavy models
```

**Skill Development**:
- ✅ CUDA kernel basics
- ✅ PyTorch torch.ops integration
- ✅ Benchmarking methodology
- ✅ Fused operation concepts
- ⚠️ Limited on: multi-GPU, complex memory patterns

**Estimated Impact**: Medium (activations are ~5-10% of inference time)

**Starting Checklist**:
- [ ] Read `sgl-kernel/csrc/elementwise/activation.cu` (existing activations)
- [ ] Understand: GELU formula, thread block sizing, warp reduction
- [ ] Create: `newgelu.cu` implementation
- [ ] Test: Numerical correctness vs. PyTorch
- [ ] Benchmark: vs. `F.gelu(x, approximate='tanh')`
- [ ] Integrate: Python wrapper, torch registration, tests

**Success Looks Like**: 15-20% faster NewGELU with identical outputs

---

### Profile 2: FP8 in Attention Merge 💾 INTERMEDIATE, HIGH IMPACT

**Fit If You**:
- Understand attention mechanisms
- Have FP8 training/inference experience
- Want to enable 8-bit inference
- Don't mind complex type handling

**What You'll Build**:
```
Before:
  merge_state_v2(v_fp32, s_fp32, ...)  # Fallback for FP8

After:
  merge_state_v2(v_fp8, s_fp8, ...)    # Native FP8 support
  # 30-40% memory savings for large batch inference
```

**Skill Development**:
- ✅ FP8 semantics and scale management
- ✅ Type-safe kernel template specialization
- ✅ Attention state handling
- ✅ Numerical stability with low precision
- ⚠️ Limited on: torch.compile, dynamic shapes

**Estimated Impact**: High (enables low-precision attention inference)

**Starting Checklist**:
- [ ] Study: `sgl-kernel/csrc/attention/merge_attn_states.cu` (current code)
- [ ] Understand: attention state merging algorithm
- [ ] Review: FP8 patterns in `sgl-kernel/csrc/gemm/fp8_*` kernels
- [ ] Implement: FP8 variant of merge_state_v2
- [ ] Test: Against PyTorch with FP8 inputs
- [ ] Benchmark: Latency and memory usage

**Success Looks Like**: FP8 attention merge works end-to-end, <5% error vs. FP32

---

### Profile 3: RoPE + Quantization Fusion 🔥 INTERMEDIATE-ADVANCED, HIGH IMPACT

**Fit If You**:
- Understand both RoPE and quantization
- Comfortable with complex register usage
- Want 20-30% speedup on real models
- Ready for 3-4 day effort

**What You'll Build**:
```
Before:
  rope_out = apply_rope(x)                    # Kernel 1
  quant_out = quantize(rope_out, scales)     # Kernel 2
  # 2x kernel launch overhead

After:
  quant_out = fused_rope_and_quantize(x, scales)  # Single kernel
  # 20-30% speedup via fusion
```

**Skill Development**:
- ✅ Kernel fusion strategies
- ✅ Register pressure management
- ✅ Complex parameter passing
- ✅ Performance tuning/profiling
- ✅ Supporting multiple precision types
- ⚠️ Limited on: advanced synchronization patterns

**Estimated Impact**: Very High (10-20% end-to-end speedup on generation)

**Starting Checklist**:
- [ ] Profile: Real model to confirm bottleneck
- [ ] Study: `pos_enc.cu` (RoPE) and `per_token_quant_fp8.cu` (quantization)
- [ ] Design: Fused kernel memory layout
- [ ] Implement: Single pass rope + quant
- [ ] Test: Outputs match separate kernels
- [ ] Benchmark: Memory bandwidth, latency, power

**Success Looks Like**: Fused kernel faster than two separate kernels; results numerically correct

---

### Profile 4: Fused Reduce-Scatter for MOE 🌐 ADVANCED, VERY HIGH IMPACT

**Fit If You**:
- Deep distributed systems knowledge
- Understand MOE token routing
- Comfortable with collective communication
- Have multi-GPU testing capability

**What You'll Build**:
```
Before:
  expert_out = experts(local_tokens)     # GEMM
  global_out = reduce_scatter(expert_out) # Separate collective

After:
  global_out = fused_reduce_scatter_gemm(tokens)  # One pass
  # 30-50% communication overhead reduction
```

**Skill Development**:
- ✅ Collective communication algorithms
- ✅ Expert-parallel distributed scheduling
- ✅ Multi-GPU synchronization
- ✅ Advanced memory pooling
- ✅ Performance debugging on multi-GPU

**Estimated Impact**: Very High (can reduce MOE bottleneck by 30-50%)

**Starting Checklist**:
- [ ] Study: `quick_all_reduce.cuh` and MOE routing patterns
- [ ] Profile: Identify reduce-scatter as actual bottleneck
- [ ] Understand: MOE token distribution and expert assignment
- [ ] Design: Fused reduce-scatter with parallel GEMM chunks
- [ ] Implement: New kernel for fused operation
- [ ] Test: Multi-GPU correctness and scaling

**Success Looks Like**: Distributed MOE inference 30% faster; scales across 8+ GPUs

---

### Profile 5: NVSHMEM AllReduce 🚀 ADVANCED, NEXT-GEN

**Fit If You**:
- Expert in CUDA and GPU architecture
- Understand symmetric memory concepts
- Have access to modern GPU hardware (H100+, GB200)
- Can invest 1-2 weeks

**What You'll Build**:
```
Before:
  torch.distributed.all_reduce(x)  # Uses NCCL (PCIe, NVLink-based)

After:
  sgl_kernel.nvshmem_allreduce(x)  # Uses symmetric memory
  # 3-5x faster on GB200; minimal latency variance
```

**Skill Development**:
- ✅ NVSHMEM API and synchronization
- ✅ GPU memory topology
- ✅ Fabric/interconnect patterns
- ✅ Real-time system programming
- ✅ Advanced GPU architecture knowledge

**Estimated Impact**: Extreme (enables new scale of distributed inference)

**Starting Checklist**:
- [ ] Review: FlashInfer's allreduce fusion approach (already uses symmetric memory)
- [ ] Learn: NVSHMEM API and persistent kernels
- [ ] Design: AllReduce algorithm compatible with NVSHMEM
- [ ] Implement: Kernel + workspace management
- [ ] Test: Correctness on multi-GPU setup
- [ ] Benchmark: Compare NCCL vs. NVSHMEM variants

**Success Looks Like**: NVSHMEM allreduce 3x faster than NCCL on compatible hardware

---

## ✅ Decision Matrix

| Opportunity | Time | Difficulty | Impact | Best For |
|---|---|---|---|---|
| NewGELU | 1-2d | ⭐ | Med | Beginners |
| Top-K Improvements | 2-3d | ⭐⭐ | Med-High | Intermediate learners |
| FP8 Attention | 2-3d | ⭐⭐ | High | FP8 enthusiasts |
| RoPE + Quant Fusion | 3-4d | ⭐⭐ | Very High | Fusion experts |
| Reduce-Scatter MOE | 3-4d | ⭐⭐⭐ | Very High | Distributed systems |
| NVSHMEM AllReduce | 1-2w | ⭐⭐⭐ | Extreme | GPU arch experts |
| Selective Scan | 2-3w | ⭐⭐⭐⭐ | Very High | Research experts |

---

## 🚦 Go/No-Go Checklist Before Starting

Before diving in, verify:

- [ ] **Performance validated**: Does the kernel actually solve a bottleneck?
  - Profile a real model using `/generate-profile` or torch.profiler
  - Verify current operation is in top-5 hotspots

- [ ] **Reference implementation exists**: Can you learn from similar code?
  - For quantization: Study `per_token_group_quant_8bit_v2.cu`
  - For fusion: Study `moe_fused_gate.cu`
  - For communication: Study `quick_all_reduce.cuh`

- [ ] **Benchmark framework ready**: Can you measure improvement?
  - Understand `triton.testing.do_bench_cudagraph` usage
  - Have simple PyTorch baseline for comparison

- [ ] **Testing infrastructure ready**: Can you validate correctness?
  - Know how to run `pytest sgl-kernel/tests/`
  - Can compare outputs against PyTorch reference

- [ ] **Hardware available**: Do you have access to test hardware?
  - At minimum: Single H100 or similar
  - For distributed: 2+ A100/H100s strongly recommended

---

## 🎓 Learning Path Recommendation

**If you're brand new to kernel development**:
1. NewGELU (learn the basics)
2. Top-K improvements (build confidence)
3. FP8 Attention (tackle type complexity)
4. RoPE + Quant fusion (master fusion patterns)

**If you have CUDA experience**:
1. FP8 Attention (directly applicable knowledge)
2. RoPE + Quant fusion (extend to multi-operation)
3. Fused Reduce-Scatter (tackle distributed)

**If you're a distributed systems expert**:
1. Fused Reduce-Scatter (apply your expertise)
2. NVSHMEM AllReduce (work with latest tech)
3. AllReduce + Computation Fusion (complex patterns)

---

## 📞 Next Steps

1. **Pick your opportunity** from decision matrix above
2. **Run the Go/No-Go checklist** to verify readiness
3. **Study reference code** listed in the profile
4. **Start with a benchmark** of current performance
5. **Implement incrementally** with frequent testing
6. **Use `/add-jit-kernel` or `/add-sgl-kernel` skill** when ready

Good luck! 🚀
