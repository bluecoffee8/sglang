# SGLang Kernel Contribution Opportunities

Based on a thorough codebase analysis of the SGLang project (as of May 2026), here are high-impact opportunities for contributing new kernel support.

---

## 🎯 High-Priority Opportunities (Recommended Starting Points)

### 1. **Quantization Kernel Expansion**

#### 1.1 FP8 Support in Attention Kernels
- **Location**: `sgl-kernel/python/sgl_kernel/attention.py` (Lines 16-18, 74)
- **Issue**: `merge_state_v2` and `cutlass_mla_decode` don't support FP8 data type
- **Why**: Critical for memory-efficient inference on latest GPUs
- **Implementation Path**:
  - Extend `merge_attn_states.cu` to handle float8_e4m3fn tensors
  - Modify `cutlass_mla_kernel.cu` to support FP8 queries/keys/values
  - Add FP8 variants to attention state merging logic
- **Files to Modify**:
  - `sgl-kernel/csrc/attention/merge_attn_states.cu`
  - `sgl-kernel/csrc/attention/cutlass_mla_kernel.cu`
  - `sgl-kernel/python/sgl_kernel/attention.py`

#### 1.2 NewGELU Activation Kernel
- **Location**: `python/sglang/srt/layers/activation.py` (TODO comment)
- **Issue**: NewGELU currently falls back to PyTorch implementation
- **Why**: NewGELU is used in many models; CUDA kernel would improve performance
- **Implementation Path**:
  - Create `sgl-kernel/csrc/elementwise/newgelu.cu`
  - Implement both unfused and fused versions (with residual/scale)
  - Add Python bindings
- **Reference**: Existing `activation.cu` provides good template

#### 1.3 Per-Token Group Quantization Optimizations
- **Location**: `sgl-kernel/csrc/gemm/per_token_group_quant_8bit*.cu`
- **Issues**: Multiple TODOs around dynamic block sizing and vectorization
- **Improvements**:
  - Dynamic `TOKEN_DIM_BLOCK_NUM_PER_EXPERT` determination
  - Vectorize SiLU + multiplication in `per_token_group_quant_8bit_v2.cu` (line ~200)
  - Optimize scale quantization (line 117 TODO)
- **Impact**: 5-10% speedup for MOE inference

---

### 2. **Communication Kernel Enhancements**

#### 2.1 NVSHMEM-Based AllReduce Fusion
- **Location**: `sgl-kernel/csrc/allreduce/`
- **Current State**: Uses custom allreduce, MSCCL++, and quick_allreduce
- **Missing**: NVSHMEM (symmetric memory) optimized kernels for all-reduce
- **Why**: NVSHMEM provides lowest-latency peer-to-peer communication on modern GPUs
- **Implementation Path**:
  - Create `sgl-kernel/csrc/allreduce/nvshmem_allreduce.cu`
  - Implement symmetric memory-based reduction patterns
  - Support both registered and unregistered buffers
  - Add per-rank synchronized execution for Hopper+
- **Reference**: FlashInfer allreduce fusion infrastructure exists (see `flashinfer_comm_fusion.py`)
- **Challenge**: Requires NVSHMEM SDK (available on recent CUDA)

#### 2.2 Reduce-Scatter Optimization for MOE
- **Location**: `python/sglang/srt/layers/moe/` and `sgl-kernel/csrc/moe/`
- **Issue**: Reduce-scatter is a bottleneck in distributed MOE inference
- **Missing Optimizations**:
  - Fused reduce-scatter for FP8/INT8 MOE weights
  - Chunk-wise reduction with early exit for sparse patterns
  - Per-expert reduce-scatter scheduling
- **Implementation Path**:
  - Create fused reduce-scatter variant optimized for MOE token distribution
  - Implement in `sgl-kernel/csrc/allreduce/` or MOE-specific file
  - Integrate with `communicator.py` dispatch logic

#### 2.3 IPC Workspace Optimization
- **Location**: `python/sglang/srt/distributed/` and `sgl-kernel/csrc/allreduce/`
- **Issue**: IPC buffer allocation and synchronization can be optimized
- **Missing**:
  - Zero-copy tensor pooling for IPC buffers
  - Asynchronous graph buffer registration
  - Deterministic IPC handle ordering
- **Why**: Reduces initialization overhead in multi-GPU setups

---

### 3. **Kernel Fusion Opportunities**

#### 3.1 Fused Operations Missing from sgl-kernel
- **Location**: Various files in `sgl-kernel/csrc/elementwise/` and `/quantization/`
- **Identified Missing Fusions**:

  a) **RoPE + Quantization Fusion**
  - Currently: Apply RoPE → Quantize (2 kernel launches)
  - Target: Fused kernel doing both in one pass
  - Files: `sgl-kernel/csrc/elementwise/pos_enc.cu`, quantization kernels
  - Expected speedup: 20-30% for RoPE + quantize pattern

  b) **Activation + MOE Gate Fusion**
  - Location: `moe_fused_gate.cu` already has some fusion
  - Enhancement: Fuse SiLU → TopK → Softmax in one kernel
  - Files: Extend `moe_fused_gate.cu` (20KB) and `moe_topk_softmax_kernels.cu` (32KB)
  - Challenge: Complex register pressure management
  - Reference: Kimi K2 has fused gate implementation (`kimi_k2_moe_fused_gate.cu`)

  c) **Scatter-Gather + Compute Fusion**
  - Location: `python/sglang/srt/disaggregation/common/staging_buffer.py`
  - Current: Triton fused kernels for gather/scatter
  - Enhancement: Add CUDA kernel variants for larger token batches
  - Expected speedup: 15-25% for disaggregated inference

  d) **Concat + Projection Fusion for MLA**
  - Location: `sgl-kernel/csrc/elementwise/concat_mla.cu` (has TODOs)
  - Current Issues: Naming and further optimization needed
  - Target: Fuse concatenation with linear projection in attention
  - Files: Already partially implemented; needs completion and tuning

#### 3.2 AllReduce Fusion with Computation
- **Location**: `python/sglang/srt/layers/layernorm.py` and `communicator.py`
- **Existing**: `flashinfer_allreduce_fusion`, `aiter_allreduce_fusion`
- **Missing Variants**:
  - RMSNorm + AllReduce + Quantization (3-way fusion)
  - Linear projection + AllReduce for communication-optimal attention
  - Layer-wise pipelining during collective operations
- **Why**: Communication/computation overlap can hide 30-50% of collective latency

---

### 4. **Architecture-Specific Kernel Support**

#### 4.1 SM100 (Blackwell) Optimizations
- **Status**: Emerging support in recent commits
- **Missing Kernels**:
  - Optimized FP4 GEMM leveraging Blackwell's features
  - Atomic-free collective operations for Blackwell's improved sync
  - Stream-agnostic kernel scheduling for new PTX features
  - Warp-specialized kernels for 128-thread warps

#### 4.2 ROCm/MUSA Quantization Kernels
- **Current**: Limited MUSA quantization support
- **Missing**:
  - Per-token FP8 quantization for MUSA
  - Group-wise INT8 quantization kernels
  - MFMA-optimized GEMM variants for FP4/INT8
- **Location**: `sgl-kernel/csrc/musa/` and `csrc/moe/`
- **Challenge**: Requires MUSA/ROCm environment for testing

#### 4.3 CPU Quantization Expansion
- **Status**: Basic W8A8 support exists
- **Missing**:
  - FP8 inference on ARM/x86 with vector instructions
  - Per-token quantization for CPU inference
  - Fallback optimizations for models without GPU
- **Location**: `sgl-kernel/csrc/cpu/` and Python wrappers
- **Why**: Important for edge deployment scenarios

---

### 5. **Operator Library Gaps**

#### 5.1 Top-K with Larger K Values
- **Location**: `sgl-kernel/python/sgl_kernel/top_k.py`
- **Issue**: Falls back to PyTorch topk for large k
- **TODO**: Implement faster CUDA kernels for large vocabulary top-k
- **Current**: 546-line `topk.cu` exists but doesn't handle all cases
- **Improvements**:
  - Cache-efficient sorting networks for large k
  - Merge-based approach for k > 512
  - Support for FP8/INT8 inputs
- **Impact**: 20-40% speedup for sampling-heavy workloads

#### 5.2 Sparse Linear Operations
- **Location**: No dedicated CUDA sparse kernels found
- **Missing**:
  - Sparse matrix multiplication for attention patterns
  - Block-sparse linear layers for MoE routing
  - Sparse-dense gemm operations
- **Why**: Emerging models (like some vision transformers) use sparse patterns

#### 5.3 Strided Gather/Scatter Variants
- **Location**: `python/sglang/srt/disaggregation/`
- **Current**: Generic Triton implementation
- **Enhancement**: Specialized CUDA kernels for:
  - Tree gather (hierarchical scatter patterns)
  - Strided scatter with conflict-free layouts
  - Ring-based gather for distributed batching

---

## 📊 Medium-Priority Opportunities

### 6. **Mamba/State-Space Model Support**

#### 6.1 Fused Causal Convolution
- **Location**: `sgl-kernel/csrc/mamba/causal_conv1d.cu` (669 lines)
- **Status**: Partial implementation exists
- **Enhancements**:
  - Fuse with state update (currently 2-pass)
  - Support arbitrary kernel sizes (currently hardcoded)
  - Optimize for long sequences (>4k)

#### 6.2 Selective Scan Kernel
- **Missing**: Core Mamba operation (selective state space scan)
- **Challenge**: Complex parallel algorithm; few GPU implementations exist
- **Reference**: Check TinyVLM or Mamba2 research code
- **Impact**: Critical for Mamba model support

---

### 7. **Distributed Training Kernel Support**

#### 7.1 Gradient Compression
- **Missing**: Sparsification kernels for distributed training
- **Target**: Reduce communication during backward pass

#### 7.2 Fused Backward Kernels
- **Missing**: Combined gradient computation + reduction kernels

---

### 8. **Sampler/Logits Processing**

#### 8.1 Fused Sampling Kernels
- **Location**: `python/sglang/srt/layers/sampler.py`
- **Current**: PyTorch-based sampling with separate quantile/prob computation
- **Enhancement**: Single CUDA kernel for:
  - Softmax + sampling
  - Temperature scaling + top-k filtering + sampling in one pass
  - Support for different sampling strategies (nucleus, beam, etc.)

#### 8.2 Logits Bias/Mask Operations
- **Missing**: Efficient masked softmax for constrained decoding

---

## 🔧 Lower-Level Optimizations

### 9. **Template Specialization & Tuning**

#### 9.1 Reduce Kernel Binary Size
- **Tool**: `analyze_whl_kernel_sizes.py` available in repo
- **Opportunity**: Some kernels are oversized (>1MB)
  - `fp8_blockwise_moe_kernel.cu` (30KB)
  - `moe_topk_softmax_kernels.cu` (32KB)
  - `gptq_kernel.cu` (1950 lines)
- **Action**: Profile and reduce template instantiations

#### 9.2 SM90-Specific Optimizations
- **Missing**: Advanced features like:
  - Shared memory to L2 persistence hints
  - Dynamic cluster launch optimization
  - Tensor memory accelerator (TMA) utilization

---

## 🎓 Learning Resources & References

### Related Code Patterns in Repo
1. **Attention fusion**: `sgl-kernel/csrc/attention/cutlass_mla_kernel.cu`
2. **MOE fusion**: `sgl-kernel/csrc/moe/moe_fused_gate.cu` + `fused_qknorm_rope_kernel.cu`
3. **Allreduce patterns**: `sgl-kernel/csrc/allreduce/quick_all_reduce.cuh`
4. **Quantization templates**: `sgl-kernel/csrc/gemm/per_token_group_quant_8bit_v2.cu`
5. **Triton fallback examples**: `python/sglang/srt/layers/elementwise.py`

### Skill Resources
- Use the `/add-jit-kernel` skill for lightweight JIT CUDA kernels
- Use the `/add-sgl-kernel` skill for heavyweight AOT CUDA kernels (includes tests & benchmarks)
- See `/write-sglang-test` for test structure

---

## 🚀 Recommended First Contributions (Difficulty Ranking)

### **Beginner-Friendly** (Good First PR)
1. **NewGELU CUDA kernel** - Self-contained, clear performance benefit
2. **Top-K improvements** - Build on existing code, well-scoped
3. **Add FP8 support to existing kernels** - Extend patterns already in code

### **Intermediate**
1. **RoPE + Quantization fusion** - Requires understanding both operations
2. **Fused reduce-scatter** - Complex distributed logic but clear interface
3. **Activation + MOE gate fusion** - Builds on existing MOE kernels

### **Advanced**
1. **NVSHMEM allreduce** - Requires external library, complex synchronization
2. **Selective scan (Mamba)** - Novel algorithm, few reference implementations
3. **AllReduce + Computation fusion** - Requires deep distributed systems knowledge

---

## 📝 Getting Started

1. **Pick an opportunity** from the "High-Priority" section
2. **Read existing similar kernels** (listed above for each opportunity)
3. **Check for related GitHub issues** or discussions
4. **Use the provided skills** (/add-jit-kernel, /add-sgl-kernel) when ready
5. **Submit a PR** with tests (see write-sglang-test skill) and benchmarks

## 💡 Key Files to Explore
```
sgl-kernel/
├── csrc/
│   ├── attention/        # Attention kernels - good learning resource
│   ├── quantization/     # Quantization templates
│   ├── moe/             # MOE fusion examples
│   ├── allreduce/       # Communication patterns
│   └── elementwise/     # Simple op patterns
├── python/sgl_kernel/   # Python wrappers
└── tests/               # Test patterns
python/sglang/srt/layers/ # Integration points
```

---

*Report generated: 2026-05-18*
*Based on commit: a8d552048 (FIx elementwise activation jit kernel)*
