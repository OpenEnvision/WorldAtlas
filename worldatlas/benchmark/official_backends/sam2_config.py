"""Resolve SAM2 configs from its package, independently of the caller's cwd."""

from contextlib import contextmanager

from hydra import initialize_config_module
from hydra.core.global_hydra import GlobalHydra


@contextmanager
def sam2_config_context():
    """Temporarily use SAM2's package search path and restore the caller's Hydra."""
    instance = GlobalHydra.instance()
    previous = instance.hydra
    instance.clear()
    try:
        with initialize_config_module(config_module="sam2", version_base="1.2"):
            yield
    finally:
        instance.hydra = previous
