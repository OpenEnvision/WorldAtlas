"""Which interactive world models are eligible for the 60-second memory track.

A memory_loop sample is one still image plus a camera itinerary that is closed by
construction, rolled out for 60 seconds. Only interactive world models qualify: the
adapter has to consume a per-token camera path and grow a short default clip into a
minute-long rollout without breaking that closure.

This module is the single declarative source for the roster. It records the
generation overrides each model needs to reach 60 seconds, and it records why the
remaining camera-capable models are excluded, so a skipped model is an auditable
decision rather than a silent omission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


MEMORY_CONFIG_SUFFIX = "_memory"


@dataclass(frozen=True, slots=True)
class MemoryModelSpec:
    """One roster entry: a base model config plus its 60-second overrides."""

    base_config: str
    generation: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    excluded_reason: str | None = None

    @property
    def enabled(self) -> bool:
        """Whether this model gets a generated memory config."""
        return self.excluded_reason is None

    @property
    def memory_config(self) -> str:
        """File name of the generated memory config."""
        stem = self.base_config.removesuffix(".yaml")
        return f"{stem}{MEMORY_CONFIG_SUFFIX}.yaml"


# Ports are distinct so several memory jobs can share a node without colliding.
MEMORY_TRACK_MODELS: dict[str, MemoryModelSpec] = {
    "astronex_world": MemoryModelSpec(
        base_config="astronex_world.yaml",
        generation={
            "target_duration_seconds": 60,
            "camera_path_mode": "closed_loop",
            "translation_per_token": 0.8,
            "rotation_degrees_per_token": 30.0,
            "offload_denoiser_for_decode": True,
        },
        notes=(
            "One continuous causal rollout: 368 latents produce 1469 frames at 24 FPS "
            "(61.21 seconds). Resample fixed 30-degree token endpoints over the latent "
            "timeline so retraces, square corners and the full yaw loop remain closed. "
            "The denoiser is offloaded only for final VAE decode and restored afterwards."
        ),
    ),
    "alayaworld_v11": MemoryModelSpec(
        base_config="alayaworld_v11.yaml",
        generation={
            "rounds": 48,
            "rotation_degrees_per_chunk": 30.0,
        },
        notes=(
            "AlayaWorld 1.1 DMD with ViGeo, one continuous 48-round rollout, 1537 frames "
            "at 24 FPS (64.04 seconds). Repeat whole 8/12/16/24-token laps; 30-degree "
            "yaw per round preserves square corners and full turns. Four history "
            "latents and the official bounded ViGeo cache; no v1 DA3 runtime."
        ),
    ),
    "zing_world": MemoryModelSpec(
        base_config="zing_world.yaml",
        generation={"target_duration_seconds": 60, "fps": 24},
        notes=(
            "Native causal rollout with one continuous KV cache and the official 97/9 "
            "sliding window. The complete keyboard itinerary spans 1440 intervals; "
            "the recorded video includes the reference image for 1441 frames at 24 FPS. "
            "No short-clip stitching or repeated-frame padding."
        ),
    ),
    "echo_wm": MemoryModelSpec(
        base_config="echo_wm.yaml",
        generation={
            "target_duration_seconds": 60,
            "fps": 24,
            "chunk_num_frames": 241,
            "no_audio": True,
            "cleanup_window_files": True,
        },
        notes=(
            "Derived long-horizon protocol over the public 10-second Base model: six "
            "241-frame windows are chained through the preceding last frame and stitched "
            "with one-frame overlap removal. The full action timeline is partitioned across "
            "windows, so the closed itinerary runs once. CPU state-dict caching prevents "
            "checkpoint rereads across windows and samples. This is not the unreleased "
            "native long-horizon Echo-WM checkpoint."
        ),
    ),
    "solarwm": MemoryModelSpec(
        base_config="solarwm.yaml",
        generation={
            "target_duration_seconds": 60,
            "fps": 16,
        },
        notes=(
            "The released Stage2 SGF route uses bounded-window autoregressive generation. "
            "A 960-frame camera source yields 957 model frames and SolarWM's official "
            "camera publication pads the final frame to preserve the full 60-second trajectory."
        ),
    ),
    "solarwm_h3": MemoryModelSpec(
        base_config="solarwm_h3.yaml",
        generation={"memory_duration_seconds": 60, "target_duration_seconds": 60},
        notes=(
            "Official source-length Stage2 EMA route uses 1440 camera source frames "
            "at 24 fps and single-node SP8. Requires the gated silence support artifact."
        ),
    ),
    "worldcrafter_fast": MemoryModelSpec(
        base_config="worldcrafter_fast.yaml",
        generation={"target_duration_seconds": 60},
        notes=(
            "Official camera-queryable memory consumes the full closed camera "
            "itinerary in 33-frame chunks at 16 fps: fifteen chunks for 30 seconds "
            "or thirty chunks for 60 seconds."
        ),
    ),
    "lingbot_world_v2": MemoryModelSpec(
        base_config="lingbot_world_v2.yaml",
        generation={
            "target_duration_seconds": 60,
            "output_fps": 16,
            "master_port": 29553,
            "continue_on_error": True,
        },
        notes=(
            "Chunked causal AR with a bounded KV window (local_attn_size 18) plus a "
            "6-latent sink that keeps the anchor view resident. 961 planned frames snap "
            "to 957 after chunk alignment, which is 59.81 s and inside preflight tolerance."
        ),
    ),
    "lingbot_world_fast": MemoryModelSpec(
        base_config="lingbot_world_fast.yaml",
        generation={
            "target_duration_seconds": 60,
            "output_fps": 16,
            "chunk_size": 3,
            # Without these the fast pipeline sizes its KV cache for the whole
            # sequence and a 240-latent rollout exhausts the GPU.
            "local_attn_size": 18,
            "sink_size": 6,
            "master_port": 29561,
        },
        notes="Bounded-window fast path on the v1 base-cam checkpoint.",
    ),
    "hy_worldplay": MemoryModelSpec(
        base_config="hy_worldplay.yaml",
        generation={
            "target_duration_seconds": 60,
            "output_fps": 24,
        },
        notes=(
            "Resamples the full token path onto the final pose count. The 16m+13 frame "
            "lattice resolves to video_length 1453, which is 60.54 s at 24 fps."
        ),
    ),
    "matrix_game_35_first": MemoryModelSpec(
        base_config="matrix_game_35_first.yaml",
        generation={"target_duration_seconds": 60},
        notes="resolve_num_blocks turns 60 s at 16 fps into 12 blocks of 84 poses.",
    ),
    "matrix_game_2": MemoryModelSpec(
        base_config="matrix_game_2.yaml",
        generation={
            "target_duration_seconds": 60,
            "output_fps": 12,
            # Autotune loses to eager conv and costs minutes per first decode.
            # Quality is unchanged; the 3-step distilled schedule is untouched.
            "compile_vae": False,
        },
        notes=(
            "plan_rollout yields 181 latents; snapped to 183 so noise T is a multiple "
            "of num_frame_per_block=3. That is 729 pixel frames at 12 fps (60.75 s). "
            "compile_vae is off so the first sample does not sit in inductor AUTOTUNE."
        ),
    ),
    "matrix_game_3": MemoryModelSpec(
        base_config="matrix_game.yaml",
        generation={
            # 26 iterations at 17 fps: 57 + 25 * 40 = 1057 frames = 62.18 s.
            "target_duration_seconds": 60,
            "action_source": "worldatlas_camera_path",
        },
        notes=(
            "plan_rollout grows num_iterations from the first-clip / overlap geometry. "
            "action_source reads camera_path so the closed itinerary is executed instead "
            "of the static plan in the base config."
        ),
    ),
    "evoke": MemoryModelSpec(
        base_config="evoke.yaml",
        generation={"target_duration_seconds": 60},
        notes=(
            "Chunk expansion now repeats whole laps, so the 36-frame chunks cover 60 s "
            "without unbalancing the forward and reverse legs of a retrace loop."
        ),
    ),
    "alayaworld": MemoryModelSpec(
        base_config="alayaworld.yaml",
        generation={
            # 48 rounds x 32 frames = 1536 frames = 64 s at 24 fps. Round counts are
            # snapped up to whole laps anyway, so this is already lap-aligned for the
            # 6, 8, 12 and 16 token itineraries in the loop set.
            "rounds": 48,
        },
        notes=(
            "History and KV cache grow with the round count; upstream documents that a "
            "45-round rollout needs more than one 80 GB device, so this entry is the most "
            "likely to need a smaller resolution or to fail the pilot on memory."
        ),
    ),
    "abot_world": MemoryModelSpec(
        base_config="abot_world.yaml",
        generation={
            # 60 blocks x 12 decoded frames = 720 frames = 60 s at 12 fps.
            "blocks": 60,
        },
        notes="Streaming blocks; the adapter has no target-duration knob, only blocks.",
    ),
    "infinite_world": MemoryModelSpec(
        base_config="infinite_world.yaml",
        generation={
            # 1 + 23 * 80 = 1841 frames = 61.4 s at 30 fps, then trimmed to 60 s.
            "num_chunks": 23,
            "target_duration_sec": 60,
            "action_source": "camera_path",
        },
        notes=(
            "target_duration_sec only trims the decoded buffer, so num_chunks is what "
            "actually buys the duration."
        ),
    ),
    "gen3c": MemoryModelSpec(
        base_config="gen3c.yaml",
        excluded_reason=(
            "Novel-view synthesis driven by a 3D point cache and an explicit camera "
            "trajectory. It could reach a minute over 121-frame chunks, but it is not an "
            "action-conditioned interactive rollout, so it sits outside this track."
        ),
    ),
    "lyra2": MemoryModelSpec(
        base_config="lyra2.yaml",
        excluded_reason=(
            "Pose-conditioned FramePack video generation. Mechanically able to reach a "
            "minute, but the control signal is a camera trajectory rather than actions, "
            "so it belongs with the view-synthesis models."
        ),
    ),
    "dreamx_world": MemoryModelSpec(
        base_config="dreamx_world.yaml",
        generation={
            # The short-clip entrypoint is full-sequence and tops out near five
            # seconds; the chunk-wise causal one is what upstream recommends for long
            # rollouts. 252 latent frames give (252-1)*4+1 = 1005 pixel frames, which
            # is 62.8 s at 16 fps.
            "entrypoint": "inference_ar_forcing.py",
            "config_path": "configs/dreamx-ar/causal_camera_forcing_5b.yaml",
            "transformer_path": "configs/dreamx-ar",
            "num_output_frames": 252,
            "fps": 16,
            "chunk_relative": True,
            "color_correction_strength": 1.0,
            # Keep every leg of the itinerary so the camera can come home.
            "max_action_segments": 0,
            # AR forcing is single-process; the distributed knobs belong to the 5B path.
            "nproc_per_node": 1,
            "ulysses_degree": 1,
            "ring_degree": 1,
            # Official long-horizon weights. The README still says baseline.pt, but
            # the HF release is a flat model.safetensors; inference_ar_forcing.py
            # already loads it with safetensors.load_file and falls back to the
            # whole dict when generator_ema is absent.
            "base_checkpoint_path": "ckpt/GD-ML--DreamX-World-5B/model.safetensors",
        },
        notes=(
            "Chunk-wise causal AR on the DreamX-World-5B weights (not the 5B-Cam "
            "short clip). The runner builds the chunk-wise command and memory samples "
            "keep their full itinerary instead of the three-segment truncation."
        ),
    ),
    "matrix_game_1": MemoryModelSpec(
        base_config="matrix_game_1.yaml",
        generation={
            # 961 frames at 16 fps is 60.06 s. The runner tiles the official 65-frame
            # clip; a single 961-frame Hunyuan call will not fit on one GPU.
            "video_length": 961,
            "chunk_video_length": 65,
            "fps": 16,
            "action_source": "worldatlas_camera_path",
            # Official short-clip infer is 50 steps / TeaCache 0.075. Sixteen
            # 65-frame Hunyuan windows at that recipe take ~14 min each on H800
            # (first sample ~3.7 h, a 60-sample shard ~9 days). Memory uses 20
            # flow-matching steps and TeaCache 0.15 so a shard can finish in
            # about a day. CFG 6 / 1280x720 / 65-frame tiles stay official.
            "inference_steps": 20,
            "num_steps": 20,
            "rel_l1_thresh": 0.15,
            "prefer_flash_attn3": True,
        },
        notes=(
            "The payload builder now receives the sample, so the itinerary comes from "
            "camera_path instead of the static plan in the config. Actions are held for "
            "a uniform number of frames, which keeps inverse pairs balanced. The runner "
            "initializes a 1-rank process group (ActionModule reads world_size), "
            "tiles 65-frame official clips with a 5-frame overlap, expands the Hunyuan "
            "I2V ``<image>`` token, routes attention through FlashAttention-3 when "
            "present, and writes per-window npy so a preempted shard can resume. "
            "inference_steps/num_steps are 20 and TeaCache is 0.15 so 16 windows are "
            "tractable; short-clip image_static keeps the official 50 / 0.075."
        ),
    ),
    "sana_wm": MemoryModelSpec(
        base_config="sana_wm.yaml",
        generation={
            # num_frames is free and snapped to a stride of 8 by the runner.
            "num_frames": 961,
            "chunk_num_frames": 161,
            "fps": 16,
        },
        notes=(
            "Adapter and runner recovered from commit 4540faa^. Camera control resamples "
            "the whole token path onto num_frames, so closure holds. The bidirectional "
            "720p checkpoint cannot attend 961 frames in one shot; the runner tiles the "
            "official 161-frame clip and stitches on the last-frame condition. Weights, "
            "the Sana checkout, and the stage-1 Gemma encoder stay on local arena/ckpts. "
            "DISABLE_XFORMERS=1 is set before importing Sana so caption masks use SDPA."
        ),
    ),
    "lingbot_world": MemoryModelSpec(
        base_config="lingbot_world.yaml",
        excluded_reason=(
            "The base entrypoint runs full-sequence diffusion: 961 frames over 70 steps "
            "with classifier-free guidance allocates the whole latent at once and cannot "
            "fit. The same checkpoint is covered by the bounded-window fast variant."
        ),
    ),
    "yume": MemoryModelSpec(
        base_config="yume.yaml",
        excluded_reason=(
            "camera_path collapses into an English movement phrase prepended to the "
            "prompt, so the model never executes the itinerary token by token. A closed "
            "loop cannot be expressed, which would score as forgetting rather than as "
            "missing camera control."
        ),
    ),
    "luciddreamer": MemoryModelSpec(
        base_config="luciddreamer.yaml",
        excluded_reason=(
            "A 3D Gaussian scene is optimised and then rendered along a fixed camera "
            "path, so the video is a render of a static reconstruction rather than a "
            "rollout that can forget anything."
        ),
    ),
    "realcam_i2v": MemoryModelSpec(
        base_config="realcam_i2v.yaml",
        excluded_reason=(
            "Full-sequence diffusion over the whole clip. A 16n+1 count near a minute is "
            "arithmetically legal but allocates every frame at once, the same failure "
            "that rules out the LingBot base entrypoint."
        ),
    ),
    "flashworld_image_static": MemoryModelSpec(
        base_config="flashworld_image_static.yaml",
        excluded_reason=(
            "Full-sequence latent with no chunking; 24 frames is the design point and "
            "memory grows linearly with duration."
        ),
    ),
    "stable_virtual_camera": MemoryModelSpec(
        base_config="stable_virtual_camera.yaml",
        excluded_reason=(
            "Mechanically able to reach a minute by raising num_targets from 72 to about "
            "720, but that is a tenfold cost increase on a novel-view sampler rather than "
            "an interactive rollout. Revisit if the budget allows."
        ),
    ),
    "lingbot_video_dense": MemoryModelSpec(
        base_config="lingbot_video_dense.yaml",
        excluded_reason=(
            "camera_path becomes a textual description rather than a pose sequence, so "
            "the itinerary is never executed token by token."
        ),
    ),
    "lingbot_video_moe": MemoryModelSpec(
        base_config="lingbot_video_moe.yaml",
        excluded_reason=(
            "Same text-only camera contract as the dense variant, and the clip length is "
            "pinned at five seconds."
        ),
    ),
}


def enabled_memory_models() -> dict[str, MemoryModelSpec]:
    """Roster entries that get a generated memory config."""
    return {key: spec for key, spec in MEMORY_TRACK_MODELS.items() if spec.enabled}


def excluded_memory_models() -> dict[str, MemoryModelSpec]:
    """Camera-capable models deliberately kept off the memory track."""
    return {key: spec for key, spec in MEMORY_TRACK_MODELS.items() if not spec.enabled}


__all__ = [
    "MEMORY_CONFIG_SUFFIX",
    "MEMORY_TRACK_MODELS",
    "MemoryModelSpec",
    "enabled_memory_models",
    "excluded_memory_models",
]
