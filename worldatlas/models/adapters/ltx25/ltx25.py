"""LTX-2.5 image-to-video integration using the official split-weight pipeline."""
from __future__ import annotations

import math
from pathlib import Path

from worldatlas.models.adapters.external_batch import ExternalBatchAdapter


def runtime_options(generation: dict) -> dict:
    if generation.get("pipeline", "distilled") != "distilled":
        raise ValueError("LTX-2.5 currently supports the official distilled pipeline")
    result = dict(generation)
    for name, default in (("height", 512), ("width", 768), ("num_frames", 121), ("seed", 10)):
        value = generation.get(name, default)
        if type(value) is not int:
            raise ValueError(f"LTX-2.5 {name} must be an integer")
        result[name] = value
    if any(result[k] <= 0 or result[k] % 64 for k in ("height", "width")):
        raise ValueError("LTX-2.5 two-stage height/width must be positive multiples of 64")
    if result["num_frames"] < 9 or (result["num_frames"] - 1) % 8:
        raise ValueError("LTX-2.5 num_frames must be 8n+1 (at least 9)")
    result["frame_rate"] = float(generation.get("frame_rate", 24))
    result["min_seconds"] = float(generation.get("min_seconds", 5))
    if not math.isfinite(result["frame_rate"]) or result["frame_rate"] <= 0 or not result["frame_rate"].is_integer():
        raise ValueError("LTX-2.5 frame_rate must be a positive integer (upstream encoder contract)")
    if not math.isfinite(result["min_seconds"]) or result["min_seconds"] <= 0:
        raise ValueError("LTX-2.5 min_seconds must be finite and positive")
    if result["num_frames"] / result["frame_rate"] < result["min_seconds"]:
        raise ValueError("LTX-2.5 output is shorter than min_seconds")
    result["image_conditioning_strength"] = float(generation.get("image_conditioning_strength", 1))
    if not 0 < result["image_conditioning_strength"] <= 1:
        raise ValueError("LTX-2.5 image_conditioning_strength must be in (0, 1]")
    # None delegates the generation-specific CRF to upstream ImageConditioner.
    result["image_crf"] = generation.get("image_crf")
    if result["image_crf"] is not None and (
        type(result["image_crf"]) is not int or not 0 <= result["image_crf"] <= 51
    ):
        raise ValueError("LTX-2.5 image_crf must be null or an integer in [0, 51]")
    if generation.get("offload", "none") != "none":
        raise ValueError("LTX-2.5 batch integration uses offload=none with a CPU weight registry")
    if generation.get("enhance_prompt", False):
        raise ValueError("LTX-2.5 benchmark prompts must not be rewritten")
    if generation.get("quantization") not in (None, "none"):
        raise ValueError("LTX-2.5 benchmark integration uses the official BF16 weights")
    return result


class LTX25Adapter(ExternalBatchAdapter):
    runner_module = "worldatlas.models.adapters.ltx25.ltx25_batch_runner"
    label = "LTX-2.5"

    def batch_checkpoint_load_policy(self) -> str:
        """One pipeline per worker; official registry retains checkpoint tensors on CPU."""
        return "load_once"

    def _build_command(self, spec_path: Path, generation_path: Path) -> list[str]:
        runtime_options(self.config.generation)
        if self.config.checkpoint_dir is None:
            raise ValueError("LTX-2.5 adapter requires checkpoint_dir")
        return super()._build_command(spec_path, generation_path)

    def batch_spec_payload(self, request):
        if request.sample.suite not in ("image_static", "image_dynamic"):
            raise ValueError("LTX-2.5 supports Image Static and Image Dynamic only")
        return super().batch_spec_payload(request)

    def generate_batch(self, requests):
        import json
        from worldatlas.models.adapters.ltx25.ltx25_batch_runner import json_sha, sha_file

        results = super().generate_batch(requests)
        for request in requests:
            result = results[request.sample.sample_id]
            if result["status"] != "generated":
                continue
            receipt_path = request.output_path.with_suffix(".ltx25.json")
            try:
                receipt = json.loads(receipt_path.read_text())
                if (receipt["sample_id"] != request.sample.sample_id or receipt["prompt"] != request.prompt
                        or receipt["generation_sha256"] != json_sha(self.config.generation)
                        or receipt["conditioning_sha256"] != sha_file(request.conditioning_image)
                        or receipt["output_sha256"] != sha_file(request.output_path)):
                    raise ValueError("LTX-2.5 output receipt does not match the request")
                result["generation_receipt"] = str(receipt_path)
                result["conditioning_geometry"] = receipt["conditioning"]
            except (OSError, ValueError, KeyError) as exc:
                result.update(status="failed", error=str(exc))
        return results
