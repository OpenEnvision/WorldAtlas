"""Run official NVIDIA ViPE with WorldAtlas offline weights.

WorldAtlas vendors ``nv-tlabs/vipe`` under ``thirdparty/vipe``. Depth models
stay the official classes (``UniDepth2Model``, ``VideoDepthAnythingDepthModel``,
``PriorDAModel``). This module only points their loaders at ``ckpt/`` so Hope
can stay offline.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from worldatlas.common.checkpoints import checkpoint_root, project_root
from worldatlas.benchmark.vipe.official_priorda import (
    ArenaPriorDAModel,
    ensure_priorda_on_path,
    ensure_torch_cluster_shim,
)
from worldatlas.benchmark.vipe.official_unidepth import ArenaUniDepth2Model, ensure_unidepth_on_path
from worldatlas.benchmark.vipe.official_vda import ArenaVideoDepthAnythingModel, ensure_vda_on_path

VIPE_REPO = project_root() / "thirdparty" / "vipe"
VIPE_DEPTH = "vipe.priors.depth"
VIPE_UNIDEPTH = f"{VIPE_DEPTH}.unidepth"
VIPE_VDA = f"{VIPE_DEPTH}.videodepthanything"
VIPE_PRIORDA = f"{VIPE_DEPTH}.priorda"

_ORIGINAL_MAKE = None
_INSTALLED = False
_RUNTIME_INSTALLED = False


def official_vipe_root() -> Path:
    explicit = os.environ.get("WORLDATLAS_VIPE_ROOT", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    return VIPE_REPO


def ensure_official_vipe_on_path() -> Path:
    from worldatlas.benchmark.metric_errors import MetricLoadError

    root = official_vipe_root()
    package = root / "vipe"
    if not package.is_dir():
        raise MetricLoadError(
            f"Official NVIDIA ViPE is missing at {root}. Clone "
            "https://github.com/nv-tlabs/vipe into WorldAtlas/thirdparty/vipe."
        )
    inserted = str(root)
    if inserted in sys.path:
        sys.path.remove(inserted)
    sys.path.insert(0, inserted)
    return root


def ensure_official_depth_on_path() -> dict[str, object]:
    ensure_official_vipe_on_path()
    ensure_torch_cluster_shim()
    return {
        "vipe": ensure_official_vipe_on_path(),
        "unidepth": ensure_unidepth_on_path(),
        "vda": ensure_vda_on_path(),
        "priorda": ensure_priorda_on_path(),
    }


def compiled_vipe_ext_path() -> Path | None:
    roots: list[Path] = []
    env = os.environ.get("TORCH_EXTENSIONS_DIR", "").strip()
    if env:
        roots.append(Path(env).expanduser())
    roots.append(project_root() / "artifacts" / "native" / "vipe_official")
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve() if root.exists() else root
        if resolved in seen:
            continue
        seen.add(resolved)
        so = resolved / "vipe_ext_jit" / "vipe_ext_jit.so"
        if so.is_file() and so.stat().st_size > 0:
            return so
    return None


def preload_compiled_vipe_ext() -> Path | None:
    """Register the already-built Hopper ``.so`` as ``vipe_ext``.

    Official ``vipe.ext`` JITs unless ``import vipe_ext`` succeeds. Torch's
    FileBaton also sleeps forever if a killed compile left
    ``vipe_ext_jit/lock`` on dolphinfs. A present ``vipe_ext_jit.so`` is
    enough; skip JIT.
    """
    import importlib.util

    existing = sys.modules.get("vipe_ext")
    if existing is not None:
        path = getattr(existing, "__file__", None)
        return Path(path) if path else None
    so = compiled_vipe_ext_path()
    if so is None:
        return None
    stale_lock = so.parent / "lock"
    if stale_lock.exists():
        stale_lock.unlink()
    # The JIT artifact exports PyInit_vipe_ext_jit. Load under that name, then
    # alias as vipe_ext so official vipe.ext skips cpp_extension.load.
    spec = importlib.util.spec_from_file_location("vipe_ext_jit", so)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["vipe_ext_jit"] = module
    sys.modules["vipe_ext"] = module
    return so


def _patch_imported(module_name: str, **exports: object) -> None:
    module = sys.modules.get(module_name)
    if module is None:
        return
    for key, value in exports.items():
        setattr(module, key, value)


def _arena_depth_exports(make):
    return {
        "make_depth_model": make,
        "UniDepth2Model": ArenaUniDepth2Model,
        "VideoDepthAnythingDepthModel": ArenaVideoDepthAnythingModel,
        "PriorDAModel": ArenaPriorDAModel,
    }


def bind_arena_depth_classes() -> None:
    """Rebind Arena wrappers onto ViPE modules that are already imported.

    Official ``priorda`` / ``videodepthanything`` pull ``vipe.ext`` (Eigen JIT).
    Do not import those packages here. Call this again after
    ``vipe.pipeline.processors`` is imported so AdaptiveDepthProcessor sees
    the local-weight wrappers.
    """
    make = None
    depth_pkg = sys.modules.get(VIPE_DEPTH)
    if depth_pkg is not None:
        make = getattr(depth_pkg, "make_depth_model", None)
    if make is None:
        return
    exports = _arena_depth_exports(make)
    for module_name in (
        VIPE_UNIDEPTH,
        VIPE_VDA,
        VIPE_PRIORDA,
        "vipe.pipeline.processors",
        "vipe.pipeline.default",
        "vipe.slam.system",
    ):
        _patch_imported(module_name, **exports)


def _install_vda_local_hub() -> Path:
    """Point official ``torch.hub.load_state_dict_from_url`` at WorldAtlas ckpt."""
    import torch

    from worldatlas.benchmark.vipe.memory_weights import vda_checkpoint_path

    src = vda_checkpoint_path()
    hub = Path(os.environ.get("TORCH_HOME", str(checkpoint_root(create=False) / "torch"))) / "hub" / "checkpoints"
    dest = hub / src.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.resolve() != src.resolve():
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        dest.symlink_to(src.resolve())

    original = torch.hub.load_state_dict_from_url
    if getattr(original, "_worldatlas_vipe_local", False):
        return dest

    def _load(url, *args, **kwargs):
        name = str(url).rsplit("/", 1)[-1]
        cached = dest.parent / name
        if cached.is_file():
            return torch.load(str(cached), map_location=kwargs.get("map_location", "cpu"))
        if "video_depth_anything" in name and src.is_file():
            return torch.load(str(src), map_location=kwargs.get("map_location", "cpu"))
        return original(url, *args, **kwargs)

    _load._worldatlas_vipe_local = True  # type: ignore[attr-defined]
    torch.hub.load_state_dict_from_url = _load  # type: ignore[assignment]
    return dest


def _is_official_unidepth_v2_config(config: dict) -> bool:
    model = config.get("model") or {}
    decoder = model.get("pixel_decoder") or {}
    return (
        model.get("layer_scale") is not None
        and decoder.get("depths") == [2, 2, 2]
        and decoder.get("out_dim") == 64
    )


def _load_official_unidepth_v2(source: Path):
    """Load ViPE's vendored UniDepthV2 the same way NVIDIA does.

    Official ``UniDepth2Model`` calls ``UniDepthV2.from_pretrained``. The June
    2024 slim snapshot does not match that decoder; use the Feb 2025 Hub
    revision that NVIDIA ViPE 1.2 actually downloads.
    """
    import json

    from vipe.priors.depth.unidepth.models.unidepthv2.unidepthv2 import UniDepthV2
    from worldatlas.benchmark.metric_errors import MetricLoadError
    from worldatlas.benchmark.vipe.memory_weights import VIPE_UNIDEPTH_REVISION

    config_path = source / "config.json"
    weight_path = source / "model.safetensors"
    if not config_path.is_file() or not weight_path.is_file():
        raise MetricLoadError(f"UniDepth snapshot incomplete: {source}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not _is_official_unidepth_v2_config(config):
        raise MetricLoadError(
            "Official NVIDIA ViPE needs the Feb 2025 UniDepthV2 snapshot "
            f"{VIPE_UNIDEPTH_REVISION}, not the June 2024 slim weights at {source}."
        )
    model = UniDepthV2.from_pretrained(str(source), local_files_only=True)
    if not hasattr(model, "resolution_level"):
        model.resolution_level = 9
    return model


def _install_unidepth_local_source() -> None:
    """Keep official ``UniDepth2Model``; load the local HF snapshot offline."""
    from vipe.priors.depth.base import DepthEstimationModel
    from vipe.priors.depth.unidepth import UniDepth2Model

    from worldatlas.benchmark.vipe.memory_weights import unidepth_snapshot_dir

    if getattr(UniDepth2Model.__init__, "_worldatlas_local_source", False):
        return

    def _init(self, type: str = "l") -> None:
        del type
        DepthEstimationModel.__init__(self)
        explicit = os.environ.get("WORLDATLAS_UNIDEPTH_MODEL", "").strip()
        source = Path(explicit) if explicit else unidepth_snapshot_dir()
        self.model = _load_official_unidepth_v2(source)
        self.model.interpolation_mode = "bilinear"
        self.model = self.model.cuda().eval()

    _init._worldatlas_local_source = True  # type: ignore[attr-defined]
    UniDepth2Model.__init__ = _init  # type: ignore[method-assign]


def _install_priorda_local_dirs() -> None:
    """Keep official ``PriorDAModel``; pass Hope-local weight directories."""
    from vipe.priors.depth.base import DepthEstimationModel
    from vipe.priors.depth.priorda import PriorDAModel
    from vipe.priors.depth.priorda.priorda import PriorDepthAnything

    from worldatlas.benchmark.vipe.memory_weights import priorda_dir

    if getattr(PriorDAModel.__init__, "_worldatlas_local_dirs", False):
        return
    weights = priorda_dir()

    def _init(self) -> None:
        DepthEstimationModel.__init__(self)
        self.model = PriorDepthAnything(
            device="cuda",
            fmde_dir=str(weights),
            cmde_dir=str(weights),
            ckpt_dir=str(weights),
        )

    _init._worldatlas_local_dirs = True  # type: ignore[attr-defined]
    PriorDAModel.__init__ = _init  # type: ignore[method-assign]


def install_official_vipe_runtime() -> None:
    """Use NVIDIA ViPE's own depth classes with WorldAtlas offline weights.

    Do not replace UniDepth / VDA / PriorDA with Arena wrappers. Official
    ``UniDepthV2.from_pretrained`` matches the Feb 2025 Hub snapshot. The June
    2024 slim weights and the newer ``thirdparty/UniDepth`` clone do not.
    """
    global _RUNTIME_INSTALLED
    if _RUNTIME_INSTALLED:
        return
    from worldatlas.benchmark.vipe.memory_weights import ensure_unidepth_refs, link_official_vipe_pose_assets
    from worldatlas.benchmark.vipe.official_priorda import ensure_torch_cluster_shim

    ensure_official_vipe_on_path()
    ensure_torch_cluster_shim()
    preload_compiled_vipe_ext()
    ensure_unidepth_refs()
    link_official_vipe_pose_assets()
    _install_vda_local_hub()
    _install_unidepth_local_source()
    _install_priorda_local_dirs()
    _RUNTIME_INSTALLED = True


def install_official_depth_override() -> None:
    """Install the official ViPE runtime (local weights, official classes)."""
    global _INSTALLED
    install_official_vipe_runtime()
    _INSTALLED = True
