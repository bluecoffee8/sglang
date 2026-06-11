# multimodal — Multi-Modal Input Processing

## Purpose

Handles preprocessing and encoding of non-text inputs (images, video, audio) before they enter the language model. Provides CUDA graph runners for vision encoders and utilities for image tiling strategies used by models like LLaVA and InternVL.

## Key Files

| File | Role |
|---|---|
| `mm_utils.py` | Image resizing, tiling (anyres, anyres_max), normalization, base64 decode |
| `vit_cuda_graph_runner.py` | CUDA graph capture/replay for standard ViT encoders |
| `internvl_vit_cuda_graph_runner.py` | InternVL-specific ViT CUDA graph runner |
| `internvl_utils.py` | InternVL image preprocessing and dynamic resolution tiling |
| `customized_mm_processor_utils.py` | Per-model custom processor wrappers |
| `audio_from_video.py` | Extracts audio track from video for audio-visual models |

## Design

```
TokenizerManager.process_mm_input()
    │
    └─ MultiModalProcessor (managers/multimodal_processor.py)
            │
            ├─ images → mm_utils.preprocess(image)
            │               ├─ resize / tile (anyres)
            │               └─ normalize → pixel_values tensor
            │
            ├─ video → audio_from_video.extract() + frame sampling
            │
            └─ pixel_values → (sent as part of BatchTokenizedGenerateReqInput)
                                        │
                                ScheduleBatch
                                        │
                               ModelRunner.forward()
                                        │
                               VitCudaGraphRunner.encode(pixel_values)
                                        └─ ViT → image_features
                                                 │
                                        merged into hidden states at image token positions
```

### Image Tiling

Many VLMs (LLaVA-NeXT, LLaVA-Onevision, InternVL) use "anyres" tiling: the input image is split into tiles that match the ViT's native resolution, plus a thumbnail. `mm_utils.py` implements:
- `process_anyres_image()` — variable number of tiles based on image aspect ratio
- `process_anyres_max_image()` — capped tile count variant

### ViT CUDA Graphs

`vit_cuda_graph_runner.py` captures CUDA graphs for the ViT encoder at fixed image-count batch sizes. This avoids graph recompilation when batch sizes change. The runner is activated when `--cuda-graph-vit` is set.

### Integration with managers/

`managers/multimodal_processor.py` coordinates preprocessing (CPU-side) and schedules it concurrently with tokenization. `managers/mm_utils.py` handles the serialization/deserialization of multi-modal data across ZMQ.
