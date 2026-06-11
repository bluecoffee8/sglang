# mem_cache — KV Cache & Memory Management

## Purpose

Manages GPU memory for KV caches across all active requests. Provides two levels:
1. **Prefix cache** (logical) — tracks which token sequences have KV computed and manages reuse via a radix tree.
2. **Memory pool** (physical) — allocates and frees raw GPU tensor pages backing the KV cache.

## Key Files

| File | Role |
|---|---|
| `memory_pool.py` | `ReqToTokenPool`, `TokenToKVPoolAllocator`, `KVCache` — physical KV memory |
| `radix_cache.py` | `RadixCache` — LRU radix tree for prefix reuse |
| `base_prefix_cache.py` | `BasePrefixCache` ABC — interface all cache implementations share |
| `unified_radix_cache.py` | `UnifiedRadixCache` — wraps multiple caches for multi-pool setups |
| `hiradix_cache.py` | `HiRadixCache` — two-level (GPU + HiCache storage) radix cache |
| `hicache_storage.py` | `HiCacheStorage` — external storage backend for overflowed KV |
| `mamba_radix_cache.py` | Mamba SSM state cache (different from KV) |
| `swa_memory_pool.py` | `SWATokenToKVPoolAllocator` — sliding-window attention pool |
| `deepseek_v4_memory_pool.py` | DeepSeek-V4 MLA-specific KV pool |
| `multimodal_cache.py` | Cache for encoded multi-modal tokens |
| `chunk_cache.py` | `ChunkCache` — non-prefix-reuse chunked allocation |
| `evict_policy.py` | Eviction policy implementations (LRU, FIFO) |
| `kv_cache_builder.py` | Constructs the right cache/pool combination from `ServerArgs` |
| `cache_init_params.py` | `CacheInitParams` — sizing parameters |
| `events.py` | `KVCacheEventMixin` — event hooks for HiCache sync |
| `common.py` | Shared helpers (`kv_to_page_indices`, `release_kv_cache`) |
| `registry.py` | Cache type registry |

## Two-Level Memory Design

```
Physical layer (memory_pool.py)
    ├─ ReqToTokenPool
    │       map: req_id → [token_position → kv_slot_index]
    ├─ TokenToKVPoolAllocator
    │       free list of available slot indices
    └─ KVCache
            GPU tensor of shape [num_slots, num_layers, 2, num_heads, head_dim]
            (2 = K and V)

Logical layer (radix_cache.py)
    └─ RadixTree
            root → child nodes keyed by token sequence segments
            each leaf → set of KV slot indices (reference counted)
```

### Prefix Cache (RadixCache)

The radix tree stores token sequences as tree paths. Each node represents a token-sequence segment and points to KV slot indices. On a new request, `match_prefix()` traverses the tree to find the longest matching prefix — those KV slots are reused. On completion, `insert()` adds the new sequence to the tree.

LRU eviction removes leaf nodes when memory is tight. Reference counting prevents eviction of nodes in active use.

### HiCache (Two-Level Cache)

`HiRadixCache` extends `RadixCache` with a second tier in `HiCacheStorage` (backed by CPU memory, NVMe, or remote storage via `connector/`). On GPU eviction, KV data is offloaded to storage. On a subsequent cache hit, it is streamed back to GPU asynchronously before the forward pass.

### Pool Variants

| Pool | When Used |
|---|---|
| `KVCache` | Standard full-attention models |
| `SWATokenToKVPool` | Sliding-window attention (Gemma2, Mistral) |
| `DeepSeekV4TokenToKVPool` | DeepSeek-V4 MLA latent KV |
| `MambaSlotAllocator` | Mamba SSM recurrent state |

`kv_cache_builder.py` inspects `ModelConfig` to instantiate the right combination.
