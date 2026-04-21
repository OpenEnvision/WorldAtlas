from __future__ import annotations

from worldarena.models.adapters.matrix_game_runner import _install_config_aliases


class _DummyConfigModule:
    def __init__(self) -> None:
        object.__setattr__(
            self,
            "_config",
            {
                "cache_size_limit": 8,
                "accumulated_cache_size_limit": 32,
            },
        )

    def __getattr__(self, name: str):
        try:
            return self._config[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value) -> None:
        if name == "_config":
            object.__setattr__(self, name, value)
            return
        if name in self._config:
            self._config[name] = value
            return
        raise AttributeError(name)


def test_install_config_aliases_supports_modules_without_allowed_keys() -> None:
    module = _DummyConfigModule()
    _install_config_aliases(
        module,
        {
            "recompile_limit": "cache_size_limit",
            "accumulated_recompile_limit": "accumulated_cache_size_limit",
        },
    )

    assert module.recompile_limit == 8
    assert module.accumulated_recompile_limit == 32

    module.recompile_limit = 5
    assert module.cache_size_limit == 5
    assert module.recompile_limit == 5

    module.accumulated_recompile_limit = 21
    assert module.accumulated_cache_size_limit == 21
    assert module.accumulated_recompile_limit == 21
