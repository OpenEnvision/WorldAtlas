# Showcase Viewer

WorldArena includes a lightweight showcase viewer for inspecting generated media before or after scoring. It follows the three visualization workflows described by WorldLens: multi-view browsing, method-to-method comparison, and GIF/image comparison.

The viewer is static. It scans prediction folders, writes `showcase_manifest.json`, links media into a local `media/` folder, and renders `index.html`. No Dash, Gradio, or remote service is required.

## Method Comparison

Compare two or more WorldArena prediction roots:

```bash
python scripts/showcase/video_method_compare_app.py \
  --method astra=artifacts/predictions/astra \
  --method yume=artifacts/predictions/yume \
  --output-dir .cache/showcase/astra_yume
```

Open `.cache/showcase/astra_yume/index.html` in a browser, or add `--serve --port 7860` to serve it locally.

## Multi-View Browsing

For WorldLens-style folders such as `generated_results/<method>/<sample>/<camera>.mp4`, run:

```bash
python scripts/showcase/video_multi_view_app.py \
  --generated-results generated_results \
  --output-dir .cache/showcase/multiview
```

The scanner groups common camera names such as `CAM_FRONT`, `CAM_BACK`, `front_left`, and `right` into the same sample row.

## GIF/Image Comparison

For GIFs or image previews:

```bash
python scripts/showcase/gif_method_compare_app.py \
  --generated-results generated_results \
  --output-dir .cache/showcase/gifs
```

This mode filters out video-only files and keeps `.gif`, `.png`, `.jpg`, `.jpeg`, and `.webp`.

## WorldArena Generation Records

If a method root contains `generation_records*.jsonl`, the viewer uses the recorded `sample_id`, `suite`, `prediction_path`, and prompt fields instead of inferring sample names from filenames. Add `--include-conditioning` to show conditioning inputs next to predictions.

Reference media can be included when a benchmark manifest is available:

```bash
worldarena showcase \
  --method astra=artifacts/predictions/astra \
  --benchmark-manifest artifacts/benchmark/manifest/worldarena_manifest.jsonl \
  --include-reference \
  --output-dir .cache/showcase/astra_with_reference
```
