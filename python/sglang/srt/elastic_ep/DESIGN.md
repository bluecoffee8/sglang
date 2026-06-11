# elastic_ep — Elastic Expert Parallelism

## Purpose

Enables dynamic adjustment of the expert parallelism (EP) group during inference — ranks can join or leave the EP group without restarting the server. This is useful for fault tolerance and autoscaling MoE models under variable load.

## Key Files

| File | Role |
|---|---|
| `elastic_ep.py` | `ElasticEPStateManager` — tracks which EP ranks are currently active |
| `expert_backup_manager.py` | `ExpertBackupManager` — manages replicated expert weights on standby ranks |
| `expert_backup_client.py` | Client used by `ModelRunner` to query/trigger expert backup operations |

## Design

```
ModelRunner
    │
    ├─ ElasticEPStateManager          (singleton per process)
    │       ├─ active_ranks tensor    (1 = active, 0 = excluded)
    │       └─ on_active_ranks_change() → rebuild EP process group
    │
    └─ ExpertBackupClient
            └─ ExpertBackupManager   (one per EP rank)
                    ├─ backup_experts(layer, expert_ids) → copy weights to standby
                    └─ restore_experts(layer, expert_ids) → pull weights from backup
```

### Rank Join/Leave Flow

1. A rank signals it is leaving (e.g., due to hardware failure or scale-down).
2. `ElasticEPStateManager` updates `active_ranks`, broadcasts the new mask to all ranks.
3. The EP process group is reformed with only active ranks.
4. `ExpertBackupManager` redistributes the leaving rank's experts to remaining active ranks.

### Rejoin Flow

On `--elastic-ep-rejoin`, a rank starts with `active_ranks[self] = 1`, all others = 0. It captures CUDA graphs solo, then rejoins the full EP group. This avoids graph invalidation from partially-active group shapes.

## Integration

`ModelRunner` calls `join_process_groups()` and `try_recover_ranks()` from this module at startup and after detected rank failures.
