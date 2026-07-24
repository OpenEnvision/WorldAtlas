# Showcase 可视化

WorldAtlas Arena 现在带一个轻量的 showcase 可视化工具，用来在评分前后检查生成结果。它支持三类可视化工作流：多视角浏览、方法之间对比、GIF/图片对比。

这个工具生成静态页面。它会扫描 prediction 目录，写出 `showcase_manifest.json`，把媒体文件链接到本地 `media/` 目录，并渲染 `index.html`。不依赖 Dash、Gradio 或远程服务。

## 方法对比

对比两个或更多 WorldAtlas Arena prediction root：

```bash
python scripts/showcase/video_method_compare_app.py \
  --method astra=artifacts/predictions/astra \
  --method yume=artifacts/predictions/yume \
  --output-dir .cache/showcase/astra_yume
```

之后在浏览器打开 `.cache/showcase/astra_yume/index.html`。如果希望本地起服务，可以追加 `--serve --port 7860`。

## 多视角浏览

对 `generated_results/<method>/<sample>/<camera>.mp4` 这类目录：

```bash
python scripts/showcase/video_multi_view_app.py \
  --generated-results generated_results \
  --output-dir .cache/showcase/multiview
```

扫描器会把 `CAM_FRONT`、`CAM_BACK`、`front_left`、`right` 等常见相机名聚合到同一个 sample 行里。

## GIF/图片对比

对 GIF 或图片预览：

```bash
python scripts/showcase/gif_method_compare_app.py \
  --generated-results generated_results \
  --output-dir .cache/showcase/gifs
```

这个模式会过滤掉视频文件，只保留 `.gif`、`.png`、`.jpg`、`.jpeg` 和 `.webp`。

## WorldAtlas Arena 生成记录

如果 method root 下有 `generation_records*.jsonl`，可视化工具会优先使用记录里的 `sample_id`、`suite`、`prediction_path` 和 prompt 字段，而不是从文件名猜 sample。追加 `--include-conditioning` 可以把 conditioning input 一起显示出来。

有 benchmark manifest 时也可以把 reference 媒体加入对比：

```bash
worldarena showcase \
  --method astra=artifacts/predictions/astra \
  --benchmark-manifest artifacts/benchmark/manifest/worldarena_manifest.jsonl \
  --include-reference \
  --output-dir .cache/showcase/astra_with_reference
```
