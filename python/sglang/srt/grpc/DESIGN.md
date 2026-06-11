# grpc — gRPC Server Package

## Purpose

Namespace package for gRPC-related code. The actual gRPC server implementation lives in `entrypoints/grpc_server.py`; this package holds generated protobuf stubs and shared gRPC utilities.

## Key Files

| File | Role |
|---|---|
| `__init__.py` | Package marker; re-exports generated proto stubs |

## Design

gRPC proto definitions are compiled from `.proto` files (not stored in this directory) and their generated Python stubs are placed here. The `entrypoints/grpc_server.py` imports the stubs to implement the service handlers.

See `entrypoints/DESIGN.md` for the overall gRPC server architecture.
