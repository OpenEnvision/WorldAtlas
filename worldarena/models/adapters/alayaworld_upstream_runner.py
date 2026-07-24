"""Launch AlayaWorld with an architecture-safe attention backend override.

The released LTX23 model currently replaces the configured attention backend
with ``FLASH_ATTENTION_3`` inside its constructor.  That works when a matching
FlashAttention implementation is installed, but it bypasses AlayaWorld's own
``runtime.attention_type`` setting on Ampere GPUs.  WorldArena keeps the
upstream checkout untouched and redirects that enum dispatch before invoking
the official CLI.
"""

from __future__ import annotations

import os
from typing import Any


ATTENTION_ENV = "WORLDARENA_ALAYAWORLD_ATTENTION_TYPE"


def _install_attention_backend_override(
    attention_type: str,
    attention_module: Any | None = None,
) -> bool:
    """Route the upstream hard-coded FA3 enum through the requested backend."""
    normalized = str(attention_type).strip().lower().replace("-", "_")
    if normalized in {"flash_attention_3", "fa3"}:
        return False
    if normalized not in {"xformers", "pytorch"}:
        raise ValueError(
            f"unsupported AlayaWorld attention backend {attention_type!r}; "
            "expected xformers, pytorch, or flash_attention_3"
        )

    if attention_module is None:
        from flash_alaya.ltx2.modules import attention as upstream_attention

        attention_module = upstream_attention

    enum_type = attention_module.AttentionFunction
    if getattr(enum_type, "_worldarena_backend_override", None) == normalized:
        return True

    original_call = enum_type.__call__
    backend_class = (
        attention_module.XFormersAttention
        if normalized == "xformers"
        else attention_module.PytorchAttention
    )

    def _worldarena_call(
        self: Any,
        q: Any,
        k: Any,
        v: Any,
        heads: int,
        mask: Any | None = None,
        window_size: tuple[int, int] | None = None,
    ) -> Any:
        if self is enum_type.FLASH_ATTENTION_3:
            backend = backend_class()
            if normalized == "pytorch":
                return backend(q, k, v, heads, mask, window_size=window_size)
            return backend(q, k, v, heads, mask)
        return original_call(self, q, k, v, heads, mask, window_size=window_size)

    enum_type.__call__ = _worldarena_call
    enum_type._worldarena_backend_override = normalized
    print(
        "[WorldArena:AlayaWorld] attention override: "
        f"hard-coded flash_attention_3 -> {normalized}",
        flush=True,
    )
    return True


def main() -> None:
    attention_type = os.environ.get(ATTENTION_ENV, "xformers")
    _install_attention_backend_override(attention_type)

    from inference.run import main as official_main

    official_main()


if __name__ == "__main__":
    main()


__all__ = ["ATTENTION_ENV", "_install_attention_backend_override", "main"]
