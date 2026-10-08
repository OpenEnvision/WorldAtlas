# Scripts

Small helpers around the `worldatlas` CLI. Most workflows use the CLI directly; see [`README.md`](../README.md).

| Script | Purpose |
|--------|---------|
| [`check_installation.py`](check_installation.py) | Validate CLI and configuration imports; optionally check a model runtime layout |
| [`checkout_dependencies.py`](checkout_dependencies.py) | Fetch pinned source revisions and apply recorded runtime patches |
| [`download_metric_checkpoints.sh`](download_metric_checkpoints.sh) | Download benchmark metric weights into `ckpt/` |
| [`run_eval_shards.py`](run_eval_shards.py) | Multi-GPU sharded `worldatlas evaluate` |

```bash
python scripts/check_installation.py
python scripts/checkout_dependencies.py --list
bash scripts/download_metric_checkpoints.sh --all

python scripts/run_eval_shards.py \
  --config config/benchmark.yaml \
  --manifest artifacts/predictions/my-model/evaluation_manifest.jsonl \
  --predictions-root artifacts/predictions/my-model \
  --output-dir artifacts/benchmark/reports/my-model-sharded \
  --suites image_static
```

Generation sharding is built into the CLI:

```bash
worldatlas generate --num-shards 8 --shard-index 0 ...
```
