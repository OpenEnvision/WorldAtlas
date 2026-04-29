# Navigation Benchmark Pack

Generated from stratified visual candidates under:

- `/bench-workspace/yesliu/hjd/data/image_remainder_balanced_4518`
- `/bench-workspace/yesliu/hjd/data/image_test_balanced_5000`
- `/bench-workspace/yesliu/hjd/data/videos_test_split_4971`

Final task files:

- `navigation_tasks.json`: 400 long navigation cases, combining image and video sources.
- `navigation_image_tasks.json`: 200 image-source navigation cases from `image_remainder_balanced_4518`.
- `navigation_video_tasks.json`: 200 video-source navigation cases from `videos_test_split_4971`.
- `scene_transition_tasks.json`: 200 new-scene transition cases from both image roots.
- `memory_revisit_tasks.json`: 200 360-degree revisit cases derived from navigation images.
- `memory_metric_design.md`: start/end revisit metric definition and reporting guidance.

The manifests use high-level ordered controls and prompts, not per-second timing. Each task includes:

- model prompt
- intended route or new-scene target
- canonical camera tokens
- Lingbot and Matrix-Game conversion hints
- detection checklist, positive cues, failure modes, and automated scoring signals
