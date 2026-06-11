# plugins — Plugin & Hook System

## Purpose

Provides a lightweight hook registry that allows external code (plugins) to inject behavior into the SGLang inference pipeline without modifying core files. Used by model-specific extensions and third-party integrations.

## Key Files

| File | Role |
|---|---|
| `hook_registry.py` | `HookRegistry` — maps hook names to lists of callables; `register_hook()`, `fire_hook()` |
| `__init__.py` | Exports the global registry singleton |

## Design

```
# Plugin registration (e.g., in a model-specific hook module)
from sglang.srt.plugins import hook_registry

hook_registry.register_hook("post_scheduler_step", my_callback)

# SGLang core fires the hook:
hook_registry.fire_hook("post_scheduler_step", scheduler=self, batch=batch)
    └─ calls my_callback(scheduler=..., batch=...)
```

### Hook Points

Hook names are arbitrary strings. The following hook points are used by SGLang internals:
- `pre_forward` / `post_forward` — before/after model forward pass
- `post_scheduler_step` — after each scheduler step
- Custom hooks added by `arg_groups/` model hooks (e.g., `hisparse_hook`)

### Design Principles

- Hooks are fire-and-forget; return values are ignored.
- Multiple callbacks can be registered for the same hook name (called in registration order).
- No hook should raise exceptions in production — they are wrapped with try/except by `fire_hook`.
