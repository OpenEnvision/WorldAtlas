from __future__ import annotations

import json
from pathlib import Path

import torch

from worldarena.benchmark.schemas import BenchmarkSample
from worldarena.models.adapters.base import PreparedGenerationRequest
from worldarena.models.adapters.echo_infinity import EchoInfinityAdapter
from worldarena.models.adapters.echo_infinity_batch_runner import (
    _decode_i2v_generated_video,
    _run_i2v_inference,
)
from worldarena.models.config import ModelRuntimeConfig


class _FakeVAE:
    def __init__(self) -> None:
        self.decoded_latent: torch.Tensor | None = None

    def decode_to_pixel(
        self,
        latent: torch.Tensor,
        *,
        use_cache: bool = False,
    ) -> torch.Tensor:
        assert not use_cache
        self.decoded_latent = latent.clone()
        batch_size, latent_frames = latent.shape[:2]
        frames = 4 * latent_frames - 3
        return (
            torch.arange(frames, dtype=latent.dtype, device=latent.device)
            .view(1, frames, 1, 1, 1)
            .expand(batch_size, frames, 3, 2, 2)
            .clone()
        )


class _FakePipeline:
    def __init__(self) -> None:
        self.vae = _FakeVAE()


class _FakeI2VScheduler:
    def __init__(self) -> None:
        self.add_noise_calls = 0

    def add_noise(
        self,
        clean_latent: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        self.add_noise_calls += 1
        return clean_latent + 0.5


class _FakeI2VGenerator:
    def __init__(self) -> None:
        self.model = type("FakeModel", (), {})()
        self.calls: list[torch.Tensor] = []

    def __call__(self, *, noisy_image_or_video: torch.Tensor, **kwargs):  # noqa: ANN003
        self.calls.append(noisy_image_or_video.clone())
        if noisy_image_or_video.shape[1] == 1:
            return None, noisy_image_or_video
        value = 2.0 if len(self.calls) == 2 else 3.0
        return None, torch.full_like(noisy_image_or_video, value)


class _FakeI2VPipeline:
    def __init__(self) -> None:
        self.generator = _FakeI2VGenerator()
        self.scheduler = _FakeI2VScheduler()
        self.vae = _FakeVAE()
        self.num_frame_per_block = 3
        self.frame_seq_length = 1
        self.local_attn_size = 12
        self.denoising_step_list = torch.tensor([1000, 500], dtype=torch.long)
        self.args = type("FakeArgs", (), {"context_noise": 0})()
        self.kv_cache1 = None
        self.crossattn_cache = None

    def text_encoder(self, *, text_prompts: list[str]) -> dict[str, list[str]]:
        return {"text_prompts": text_prompts}

    def _initialize_kv_cache(self, **kwargs) -> None:  # noqa: ANN003
        self.kv_cache1 = []

    def _initialize_crossattn_cache(self, **kwargs) -> None:  # noqa: ANN003
        self.crossattn_cache = []

    def _set_all_modules_max_attention_size(self, local_attn_size: int) -> None:
        self.max_attention_size = local_attn_size


def _make_config(tmp_path: Path) -> ModelRuntimeConfig:
    config_path = tmp_path / "config" / "models" / "echo.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("model: {}\n", encoding="utf-8")
    repo_root = tmp_path / "Echo-Infinity"
    ckpt_root = tmp_path / "checkpoints"
    repo_root.mkdir()
    ckpt_root.mkdir()
    return ModelRuntimeConfig(
        config_path=config_path,
        name="echo-infinity-test",
        family="echo_infinity",
        python_bin="python",
        repo_root=repo_root,
        checkpoint_dir=ckpt_root,
        entrypoint=None,
        output_ext=".mp4",
        supported_suites=["image_dynamic"],
        prompt_mode="prompt_target",
        reference_mode="first_frame",
        generation={"num_gpus": 1, "num_output_frames": 21, "fps": 16},
    )


def _make_sample(reference_path: Path) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id="image-dynamic-001",
        suite="image_dynamic",
        split="test",
        asset_level="image_dynamic",
        modality="image",
        is_formal=True,
        eligible_for_official=True,
        category_path="dynamic/photorealistic/images/articulated",
        path=str(reference_path),
        relative_path="dynamic/photorealistic/images/articulated/001.jpg",
        reference_path=str(reference_path),
        prediction_stem="image_dynamic_sample_001",
        conditioning_strategy="reference_image",
        prompt_current="A still scene.",
        prompt_target="The scene moves.",
        style="photorealistic",
        environment="images",
        scene="articulated",
        motion_category="dynamic",
        source_name="test",
        source_type="synthetic",
        source_group_id="group-1",
        license_bucket="internal",
    )


def test_decode_i2v_generated_video_uses_conditioning_context_then_trims_prefix() -> None:
    pipeline = _FakePipeline()
    conditioning_latent = torch.full([1, 1, 2, 1, 1], -9.0)
    generated_latents = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2, 1, 1)
    latent_output = torch.cat([conditioning_latent, generated_latents], dim=1)

    video = _decode_i2v_generated_video(
        pipeline,
        latent_output,
        num_input_frames=1,
        num_generated_frames=3,
    )

    assert video.shape == (1, 9, 3, 2, 2)
    assert pipeline.vae.decoded_latent is not None
    assert torch.equal(pipeline.vae.decoded_latent, latent_output)
    assert torch.equal(video[:, 0], torch.full_like(video[:, 0], 1.0))


def test_i2v_context_cache_uses_clean_denoised_block(monkeypatch) -> None:
    pipeline = _FakeI2VPipeline()
    conditioning_latent = torch.full([1, 1, 2, 1, 1], -1.0)
    noise = torch.zeros([1, 3, 2, 1, 1])

    monkeypatch.setattr(
        "worldarena.models.adapters.echo_infinity_batch_runner._load_conditioning_latent",
        lambda *args, **kwargs: conditioning_latent,
    )

    _run_i2v_inference(
        pipeline,
        prompt="move",
        conditioning_image="conditioning.jpg",
        noise=noise,
        device=torch.device("cpu"),
    )

    assert pipeline.scheduler.add_noise_calls == 1
    assert len(pipeline.generator.calls) == 4
    assert torch.equal(
        pipeline.generator.calls[-1],
        torch.full([1, 3, 2, 1, 1], 3.0),
    )


def test_echo_infinity_batch_results_include_conditioning_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "conditioning.jpg"
    reference_path.write_bytes(b"image")
    output_path = tmp_path / "predictions" / "sample.mp4"
    request = PreparedGenerationRequest(
        sample=_make_sample(reference_path),
        conditioning_image=reference_path,
        output_path=output_path,
        prompt="The scene moves.",
    )
    adapter = EchoInfinityAdapter(_make_config(tmp_path))

    def fake_run_command(command, *, cwd, env, log_path):  # noqa: ANN001
        spec_index = command.index("--batch_spec_path") + 1
        spec_path = Path(command[spec_index])
        row = json.loads(spec_path.read_text(encoding="utf-8").splitlines()[0])
        Path(row["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(row["output_path"]).write_bytes(b"video")
        log_path.write_text("ok\n", encoding="utf-8")

    monkeypatch.setattr(
        "worldarena.models.adapters.echo_infinity._run_echo_infinity_command",
        fake_run_command,
    )

    results = adapter.generate_batch([request])

    payload = results[request.sample.sample_id]
    assert payload["status"] == "generated"
    assert payload["conditioning_image"] == str(reference_path.resolve())
    assert payload["conditioning_input"] == str(reference_path.resolve())
    assert payload["echo_infinity_i2v_decode"] == "conditioned_decode_trim_prefix"
