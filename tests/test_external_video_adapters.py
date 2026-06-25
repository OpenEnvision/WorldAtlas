from __future__ import annotations

from pathlib import Path

from worldarena.models.adapters.cogvideo import CogVideoAdapter
from worldarena.models.adapters.cosmos_framework import CosmosFrameworkAdapter
from worldarena.models.adapters.cosmos_predict25 import CosmosPredict25Adapter
from worldarena.models.adapters.helios import HeliosAdapter
from worldarena.models.adapters.hunyuan_video import HunyuanVideoAdapter
from worldarena.models.adapters.longcat_video import LongCatVideoAdapter
from worldarena.models.adapters.longlive import LongLiveAdapter
from worldarena.models.adapters.ltx23 import LTX23Adapter
from worldarena.models.adapters.sana_wm import SanaWMAdapter
from worldarena.models.adapters.wan22 import Wan22Adapter
from worldarena.models.adapters.warp_as_history import WarpAsHistoryAdapter
from worldarena.models.config import load_model_config
from worldarena.models.registry import build_model_adapter


def test_cosmos_predict25_batch_runner_dry_run(tmp_path: Path, monkeypatch) -> None:
    import json
    import os
    import subprocess

    config = load_model_config(Path("config/models/cosmos_predict25_14b.yaml"))
    spec_path = tmp_path / "batch.jsonl"
    generation_path = tmp_path / "generation.json"
    generation_path.write_text(json.dumps(config.generation), encoding="utf-8")
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"fake")
    output_path = tmp_path / "output.mp4"
    spec_path.write_text(
        json.dumps(
            {
                "sample_id": "sample_001",
                "suite": "image_static",
                "prediction_stem": "image_static_demo_0001",
                "conditioning_image": str(image_path),
                "output_path": str(output_path),
                "prompt": "A calm lake at sunset.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("WORLDARENA_COSMOS_PREDICT25_DRY_RUN", "1")
    result = subprocess.run(
        [
            config.python_bin,
            "-m",
            "worldarena.models.adapters.cosmos_predict25_batch_runner",
            "--repo_root",
            str(config.repo_root),
            "--checkpoint_dir",
            str(config.checkpoint_dir),
            "--batch_spec_path",
            str(spec_path),
            "--generation_config_path",
            str(generation_path),
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["dry_run"] is True
    assert "worldarena.models.adapters.cosmos_predict25_inference_entry" in payload["command"]
    assert payload["checkpoint_path"].endswith("_ema_bf16.pt")
    assert payload["local_aux"]["reason1_dir"].endswith("Cosmos-Reason1-7B")
    assert payload["local_aux"]["tokenizer_vae"].endswith("Wan2.1_VAE.pth")
    assert "model.config.text_encoder_config.ckpt_path=" in " ".join(payload["experiment_opts"])
    assert payload["sample_preview"]["inference_type"] == "image2world"


def test_external_video_model_configs_build_batch_adapters() -> None:
    expected = {
        "config/models/wan22.yaml": Wan22Adapter,
        "config/models/warp_as_history.yaml": WarpAsHistoryAdapter,
        "config/models/sana_wm.yaml": SanaWMAdapter,
        "config/models/longcat_video.yaml": LongCatVideoAdapter,
        "config/models/longlive_1.yaml": LongLiveAdapter,
        "config/models/longlive_2.yaml": LongLiveAdapter,
        "config/models/ltx23.yaml": LTX23Adapter,
        "config/models/hunyuan_video.yaml": HunyuanVideoAdapter,
        "config/models/cosmos_framework.yaml": CosmosFrameworkAdapter,
        "config/models/cosmos3_nano.yaml": CosmosFrameworkAdapter,
        "config/models/cosmos_predict25_14b.yaml": CosmosPredict25Adapter,
        "config/models/cosmos_predict25_2b.yaml": CosmosPredict25Adapter,
        "config/models/cogvideo.yaml": CogVideoAdapter,
        "config/models/helios_distilled.yaml": HeliosAdapter,
    }

    for config_path, adapter_type in expected.items():
        adapter = build_model_adapter(load_model_config(Path(config_path)))
        assert isinstance(adapter, adapter_type)
        assert adapter.supports_batch_generation()
