from __future__ import annotations

from pathlib import Path

from worldarena.common import checkpoints
from worldarena.models.adapters.lyra2_runner import _complete_local_hf_dir, _hf_repo_aliases


def _touch(path: Path, content: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_lyra2_hf_dependency_resolution_skips_empty_alias(monkeypatch, tmp_path: Path) -> None:
    ckpt_root = tmp_path / "ckpt"
    monkeypatch.setenv(checkpoints.CHECKPOINT_ROOT_ENV, str(ckpt_root))
    _touch(ckpt_root / "google--umt5-xxl" / "spiece.model", b"")
    _touch(ckpt_root / "google--umt5-xxl" / "tokenizer_config.json", b"")
    _touch(ckpt_root / "umt5-xxl" / "spiece.model")
    _touch(ckpt_root / "umt5-xxl" / "tokenizer_config.json")

    resolved = _complete_local_hf_dir(
        ("google/umt5-xxl", "google--umt5-xxl", "umt5-xxl"),
        required_files=("spiece.model", "tokenizer_config.json"),
    )

    assert resolved == ckpt_root / "umt5-xxl"


def test_lyra2_hf_aliases_include_xlm_roberta(monkeypatch, tmp_path: Path) -> None:
    ckpt_root = tmp_path / "ckpt"
    monkeypatch.setenv(checkpoints.CHECKPOINT_ROOT_ENV, str(ckpt_root))
    _touch(ckpt_root / "xlm-roberta-large" / "config.json")
    _touch(ckpt_root / "xlm-roberta-large" / "sentencepiece.bpe.model")
    _touch(ckpt_root / "umt5-xxl" / "spiece.model")
    _touch(ckpt_root / "umt5-xxl" / "tokenizer_config.json")
    _touch(ckpt_root / "DA3NESTED-GIANT-LARGE-1.1" / "config.json")
    _touch(ckpt_root / "DA3NESTED-GIANT-LARGE-1.1" / "model.safetensors")

    aliases = _hf_repo_aliases(
        "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        use_moge_scale=False,
    )

    assert aliases["xlm-roberta-large"] == ckpt_root / "xlm-roberta-large"
    assert aliases["google/umt5-xxl"] == ckpt_root / "umt5-xxl"
    assert aliases["depth-anything/DA3NESTED-GIANT-LARGE-1.1"] == ckpt_root / "DA3NESTED-GIANT-LARGE-1.1"
