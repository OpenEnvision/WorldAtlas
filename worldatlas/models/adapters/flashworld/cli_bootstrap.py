"""Launch FlashWorld with gsplat built against the active PyTorch C++ ABI."""
from __future__ import annotations

import fcntl
import functools
import os
from pathlib import Path
import runpy
import subprocess
import sys


def abi_flags(flags, abi):
    return [f for f in (flags or []) if not f.startswith('-D_GLIBCXX_USE_CXX11_ABI=')] + [
        f'-D_GLIBCXX_USE_CXX11_ABI={int(abi)}'
    ]


def configure_compiler(abi, compiler):
    def supports_abi(candidate):
        probe = subprocess.run(
            [candidate, f'-D_GLIBCXX_USE_CXX11_ABI={int(abi)}', '-dM', '-E',
             '-x', 'c++', '-include', 'string', '/dev/null'],
            capture_output=True, text=True, timeout=60,
        )
        return probe.returncode == 0 and (
            f'#define _GLIBCXX_USE_CXX11_ABI {int(abi)}' in probe.stdout.splitlines()
        )

    if supports_abi(compiler):
        return
    # This host's Red Hat devtoolset explicitly disables dual ABI even when
    # -D requests ABI 1. Reuse the installed CUDA-compatible GCC 12 toolchain.
    bindir = Path(__file__).resolve().parents[5] / 'envs/gen3c/bin'
    cxx = bindir / 'x86_64-conda-linux-gnu-g++'
    cc = bindir / 'x86_64-conda-linux-gnu-gcc'
    if not cxx.is_file() or not cc.is_file() or not supports_abi(str(cxx)):
        raise RuntimeError(f'FlashWorld requires a compiler supporting PyTorch C++ ABI {int(abi)}; {compiler} does not')
    os.environ['CXX'] = str(cxx)
    os.environ['CC'] = str(cc)
    print(f'FlashWorld CUDA host compiler: {cxx}', flush=True)


def prepare_gsplat():
    import torch
    from torch.utils import cpp_extension

    configure_compiler(torch._C._GLIBCXX_USE_CXX11_ABI, cpp_extension.get_cxx_compiler())
    original = cpp_extension._jit_compile

    @functools.wraps(original)
    def compile_with_abi(name, sources, extra_cflags, extra_cuda_cflags, *args, **kwargs):
        if name == 'gsplat_cuda':
            # Devtoolset GCC defaults to ABI 0; recent PyTorch wheels use ABI 1.
            abi = torch._C._GLIBCXX_USE_CXX11_ABI
            extra_cflags = abi_flags(extra_cflags, abi)
            extra_cuda_cflags = abi_flags(extra_cuda_cflags, abi)
        return original(name, sources, extra_cflags, extra_cuda_cflags, *args, **kwargs)

    cpp_extension._jit_compile = compile_with_abi
    # gsplat removes PyTorch's build lock itself. Serialize its import so the
    # eight inference workers cannot delete one another's in-progress build.
    cache = Path(cpp_extension._get_build_directory('flashworld_bootstrap', verbose=False))
    with (cache / 'gsplat.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        from gsplat.cuda import _backend
        if _backend._C is None:
            raise RuntimeError('FlashWorld requires the gsplat CUDA extension')
    print('FlashWorld gsplat CUDA preflight passed', flush=True)


def main():
    os.environ['SETUPTOOLS_USE_DISTUTILS'] = 'local'
    prepare_gsplat()
    script = Path(sys.argv[1]).resolve()
    sys.argv = sys.argv[1:]
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
