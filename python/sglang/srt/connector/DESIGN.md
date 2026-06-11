# connector — External Storage Connectors

## Purpose

Provides a unified interface for reading model weights and KV cache data from external storage systems (S3, Azure Blob, Redis, remote HTTP instances). Used by the model loader and HiCache storage backend to fetch data without baking storage-specific logic into the core engine.

## Key Files

| File | Role |
|---|---|
| `base_connector.py` | `BaseConnector` abstract class — `get()`, `put()`, `list()` |
| `s3.py` | AWS S3 connector |
| `azure.py` | Azure Blob Storage connector |
| `redis.py` | Redis connector (for KV cache sharing across instances) |
| `remote_instance.py` | HTTP connector to another SGLang instance |
| `utils.py` | Connector factory (`create_connector(url)`) |

## Design

```
consumers (model_loader, hicache_storage)
       │
       └─ create_connector(uri)   ← parses URI scheme to select backend
               │
               ├─ s3://...        → S3Connector
               ├─ az://...        → AzureConnector
               ├─ redis://...     → RedisConnector
               └─ http://...      → RemoteInstanceConnector
```

All connectors expose the same `BaseConnector` interface, so callers are storage-agnostic. The `create_connector(uri)` factory in `utils.py` inspects the URI scheme to instantiate the right class.

## Usage

- `model_loader/loader.py` uses connectors to fetch weights from remote storage during startup.
- `mem_cache/hicache_storage.py` uses connectors (typically Redis or S3) to persist/retrieve KV cache across restarts or instances.
