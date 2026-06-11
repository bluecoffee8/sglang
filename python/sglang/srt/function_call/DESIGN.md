# function_call — Tool / Function Call Parsing

## Purpose

Parses tool call invocations from model-generated text into structured `ToolCallItem` objects. Because each model family uses its own tool-call serialization format, there is one detector per model family. Supports both streaming (incremental) and non-streaming parsing.

## Key Files

| File | Role |
|---|---|
| `function_call_parser.py` | `FunctionCallParser` — selects detector by model, drives streaming/batch parsing |
| `base_format_detector.py` | `BaseFormatDetector` ABC — interface all detectors implement |
| `core_types.py` | `ToolCallItem` dataclass (function name + arguments) |
| `utils.py` | JSON schema constraint helpers, tool schema flattening |
| `*_detector.py` | One per model family (GPT, Hermes, DeepSeek variants, Qwen2.5, Llama 3.2, etc.) |

## Design

```
DetokenizerManager / HTTP response assembly
    │
    └─ FunctionCallParser(model_name, tools)
            │
            ├─ detector = ToolCallParserEnum[model_name]()
            │               (e.g. HermesDetector, DeepSeekV4Detector, Qwen25Detector)
            │
            ├─ Streaming mode:
            │       new_text chunk → detector.parse_streaming_increment(chunk)
            │               → (normal_text, List[ToolCallItem])
            │
            └─ Non-streaming mode:
                    full_text → detector.parse_non_streaming(full_text)
                            → (normal_text, List[ToolCallItem])
```

### Detector Responsibilities

Each `BaseFormatDetector` subclass knows the start/end tags (or JSON wrappers) its model uses for tool calls. It maintains streaming parse state so partial JSON arguments can be assembled incrementally without buffering the full response.

### Format Examples

- **Hermes / GPT-OSS**: `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`
- **DeepSeek V3/V4**: `<｜tool▁calls▁begin｜>...<｜tool▁calls▁end｜>`
- **Llama 3.2**: `<|python_tag|>{"name": ..., ...}`
- **Pythonic**: `func_name(arg1=val1, arg2=val2)` style

### Tool Strict Levels

`ToolStrictLevel` env var controls how aggressively the parser enforces schema compliance, allowing lenient parsing for models that deviate slightly from format.
