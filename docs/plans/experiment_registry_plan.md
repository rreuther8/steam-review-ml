# Experiment Registry Manifest (implementation plan)

Status: **implemented** (v1 manifest + export)  
Last updated: 2026-06-16

**Overview:** In-repo experiment registry: YAML manifest (metadata), export script, human-readable markdown, and joined metrics CSV for v1 backfill and v2 extension.

**Live docs:** [`experiment_registry.md`](../experiment_registry.md)

---

## Delivered

| Item | Path |
|------|------|
| Manifest | [`configs/experiment_registry.yaml`](../configs/experiment_registry.yaml) |
| Export library | [`src/steam_review_ml/evaluation/experiment_registry.py`](../src/steam_review_ml/evaluation/experiment_registry.py) |
| Export script | [`scripts/recs_export_experiment_registry.py`](../scripts/recs_export_experiment_registry.py) |
| Metrics CSV | `artifacts/recs/experiment_registry_metrics.csv` |
| Human doc | [`docs/experiment_registry.md`](../experiment_registry.md) |
| Test | [`tests/test_experiment_registry.py`](../tests/test_experiment_registry.py) |

```bash
python scripts/recs_export_experiment_registry.py
```

Requires PyYAML (`pip install pyyaml`) to load the manifest.

---

## Architecture: two-layer registry

| Layer | File | Owns |
|-------|------|------|
| **Manifest** | `configs/experiment_registry.yaml` | experiment_id, method_id, phase, status, notebook, pool_contract, eval_job_wired, decision_log_ref, notes |
| **Metrics** | `artifacts/recs/experiment_registry_metrics.csv` | Numbers joined from eval CSVs |
| **Human doc** | `docs/experiment_registry.md` | Contract header, v1/v2 tables, links |

**Rule:** metrics never live in YAML; killed spikes use `metrics_source: notebook` (numbers stay in notebooks / human doc).

---

## Maintenance

After offline eval runs:

1. `python scripts/recs_job_eval_offline.py configs/recs_job_eval_offline.json --examples-parquet ...`
2. `python scripts/recs_job_eval_ranking.py configs/recs_job_eval_ranking.json` (optional, for rank-only dir)
3. `python scripts/recs_export_experiment_registry.py`

When adding v2 spikes: new row in YAML → implement scorer → set `eval_job_wired: true` when in eval job → re-export.
