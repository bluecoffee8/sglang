# parser — Chat Template & Conversation Parsing

## Purpose

Converts raw API inputs (chat messages, conversations) into token sequences. Handles chat template rendering (Jinja2), code-completion prompt construction, reasoning chain parsing, and protocol-specific conversation formats.

## Key Files

| File | Role |
|---|---|
| `conversation.py` | `Conversation` — message history container; renders to prompt string |
| `jinja_template_utils.py` | Jinja2 environment setup and template rendering for HF tokenizer chat templates |
| `reasoning_parser.py` | `ReasoningParser` — extracts `<think>…</think>` reasoning traces from output |
| `code_completion_parser.py` | Constructs FIM (fill-in-the-middle) prompts for code completion |
| `harmony_parser.py` | Parser for the Harmony multi-turn conversation protocol |

## Design

```
TokenizerManager.tokenize_request(req)
    │
    ├─ Chat completion: req.messages → jinja_template_utils.apply_chat_template()
    │       └─ tokenizer.apply_chat_template(messages, tokenize=True)
    │               uses model's built-in Jinja2 template
    │
    ├─ Code completion (FIM): req.prompt → code_completion_parser.build_fim_prompt()
    │
    └─ output post-processing:
            reasoning_parser.extract_reasoning(output_text)
                    └─ splits into (thinking_text, response_text)
```

### Jinja2 Chat Templates

Modern tokenizers ship with a Jinja2 template that controls how messages are serialized (system prompt, human turn, assistant turn markers). `jinja_template_utils.py` wraps HuggingFace's `apply_chat_template` with SGLang-specific extensions (tool definitions, reasoning budget injection).

### Reasoning Parser

Models that emit `<think>` blocks (DeepSeek-R1, QwQ, etc.) produce interleaved reasoning and response text. `ReasoningParser` splits these so the API can return them in separate fields (`reasoning_content` vs `content`).
