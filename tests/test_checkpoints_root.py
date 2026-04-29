from __future__ import annotations

from pathlib import Path

from worldarena.common import checkpoints


def test_default_project_root_matches_repo_checkout() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assert checkpoints.project_root() == repo_root
    assert checkpoints.canonical_checkpoint_root() == repo_root / "ckpt"


def test_cosmos_tokenizer_repo_uses_direct_checkpoint_dir() -> None:
    assert checkpoints.HF_REPO_SUBPATHS["nvidia/Cosmos-Tokenize1-CV8x8x8-720p"] == (
        "Cosmos-Tokenize1-CV8x8x8-720p",
    )


def test_gen3c_t5_repo_uses_complete_direct_checkpoint_dir() -> None:
    assert checkpoints.HF_REPO_SUBPATHS["google-t5/t5-11b"] == ("t5-11b",)
