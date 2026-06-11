# session — Session Management

## Purpose

Manages multi-turn conversation sessions, allowing the KV cache from previous turns to be retained and reused for subsequent requests. This avoids re-computing the prompt KV for every turn in a long conversation.

## Key Files

| File | Role |
|---|---|
| `session_controller.py` | `SessionController` — creates, manages, and closes sessions; maps session IDs to KV state |
| `streaming_session.py` | `StreamingSession` — handles streaming output accumulation within a session |

## Design

```
Client (multi-turn chat)
    │
    ├─ POST /v1/chat/completions  { session_id: "abc123", messages: [...] }
    │
    └─ TokenizerManager
            │
            └─ SessionController
                    │
                    ├─ session_id → Session
                    │       ├─ cached_token_ids  (tokens processed in previous turns)
                    │       └─ kv_cache_ref      (radix tree node held by this session)
                    │
                    └─ on new request:
                            ├─ prepend session.cached_token_ids to this turn's tokens
                            ├─ pass extended token sequence to Scheduler
                            └─ on response: update session.cached_token_ids with output
```

### Session Lifecycle

1. **Open**: Client sends `session_id` (or server auto-generates one).
2. **Request**: Token sequence = `session.cached_tokens + new_message_tokens`.
3. **Prefix reuse**: Because session tokens match a radix tree prefix, their KV is reused.
4. **Update**: After generation, session cached tokens are extended with the full exchange.
5. **Close**: Client sends `CloseSessionReqInput`; session state and KV reference are released.

### Memory Management

Sessions hold a reference-count lock on the radix tree nodes covering their cached tokens, preventing eviction. `SessionController` tracks per-session memory usage and can enforce per-session token limits.
