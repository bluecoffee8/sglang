# tokenizer — Tokenizer Implementations

## Purpose

Provides tokenizer implementations supplementing HuggingFace's `transformers` tokenizers. Currently contains a custom tiktoken-based tokenizer for models that use BPE tokenization incompatible with the HF tokenizer API.

## Key Files

| File | Role |
|---|---|
| `tiktoken_tokenizer.py` | `TiktokenTokenizer` — wraps OpenAI's tiktoken library in the HF tokenizer interface |

## Design

```
TokenizerManager
    │
    └─ get_tokenizer(model_path, tokenizer_mode)
            │
            ├─ "auto" → AutoTokenizer (HuggingFace)
            ├─ "tiktoken" → TiktokenTokenizer (this module)
            └─ "mistral" → MistralTokenizer (via mistral_common)
```

`TiktokenTokenizer` implements the same interface as `transformers.PreTrainedTokenizer` (`.encode()`, `.decode()`, `.__call__()`, special token IDs, etc.) so the rest of SGLang can treat all tokenizers uniformly.

### When tiktoken is used

Models like GPT-4 / GPT-3.5 / Claude use tiktoken's `cl100k_base` or `o200k_base` encoding. These are not natively supported by HuggingFace tokenizers. Setting `--tokenizer-mode=tiktoken` activates this path.
