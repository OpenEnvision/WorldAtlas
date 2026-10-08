"""Instrument the H3 CLI inside each torchrun worker."""

from __future__ import annotations

import faulthandler
import functools
from contextlib import contextmanager
import math
import os
from pathlib import Path
import sys
import threading
import time
import traceback


def thread_stacks():
    """Take owned frame references from a Python thread that holds the GIL."""
    frames = sys._current_frames()
    try:
        result = [f"H3_STACK time={time.strftime('%Y-%m-%dT%H:%M:%S%z')} "
                  f"rank={os.environ.get('RANK', '0')} pid={os.getpid()}\n"]
        for identifier, frame in frames.items():
            if identifier == threading.get_ident():
                continue
            result.append(f"Thread 0x{identifier:x} (oldest call first):\n")
            result.extend(traceback.format_stack(frame))
        return "".join(result)
    finally:
        frames.clear()


@contextmanager
def periodic_stack_dumps(path: Path, *, interval=300):
    """Sample from Python, avoiding faulthandler's GIL-free native timer.

    The 2026-09-26 rank-5 fatal GIL error coincided with an incomplete native
    timer dump. Ordinary Python frame capture holds references and runs with
    the GIL. A native call that keeps the GIL may prevent a sample; the parent
    process independently enforces the inference progress timeout.
    """
    interval = float(interval)
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("H3 stack dump interval must be positive and finite")
    stop = threading.Event()

    def sample():
        while not stop.wait(interval):
            try:
                text = thread_stacks()
                if stop.is_set():
                    return
                # This thread owns the handle, including if shared-storage IO
                # outlasts the bounded join during worker shutdown.
                with path.open("a", encoding="utf-8") as output:
                    output.write(text)
            except Exception as error:
                print(f"H3_DIAGNOSTIC_ERROR rank={os.environ.get('RANK', '0')} "
                      f"error={type(error).__name__}: {error}", flush=True)

    sampler = threading.Thread(target=sample, name="h3-stack-sampler", daemon=True)
    sampler.start()
    try:
        yield sampler
    finally:
        stop.set()
        sampler.join(timeout=2)


def trace(function, name, *, limit=None, rank_zero=False):
    calls = 0

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        nonlocal calls
        calls += 1
        rank = os.environ.get("RANK", "0")
        enabled = (limit is None or calls <= limit) and (not rank_zero or rank == "0")
        if not enabled:
            return function(*args, **kwargs)
        label = f"H3_STAGE rank={rank} stage={name} call={calls}"
        print(f"{label} event=begin", flush=True)
        started = time.monotonic()
        result = function(*args, **kwargs)
        print(f"{label} event=end seconds={time.monotonic() - started:.3f}", flush=True)
        return result

    return wrapped


def install_diagnostics():
    import torch
    import torch.distributed as dist
    from solarwm.backends.minimax_h3 import distributed, fsdp, full_inference, sgf_rollout

    # Force lazy NCCL communicators to initialize before hours of conditioning.
    # These probes use constants and do not consume the model's RNG stream.
    initialize = fsdp.initialize_distributed

    def initialize_and_probe(**kwargs):
        result = initialize(**kwargs)
        group = distributed.get_sp_group()
        size = distributed.get_sp_size()
        rank = distributed.get_sp_rank()
        value = torch.full((size,), rank, device="cuda", dtype=torch.float32)
        received = torch.empty_like(value)
        trace(dist.all_to_all_single, "probe_all_to_all")(received, value, group=group)
        torch.testing.assert_close(received, torch.arange(size, device="cuda", dtype=value.dtype))
        trace(distributed.broadcast_sp_tensor, "probe_broadcast")(value)
        torch.testing.assert_close(value, torch.zeros_like(value))
        gathered = torch.empty(size * size, device="cuda")
        trace(dist.all_gather_into_tensor, "probe_all_gather")(gathered, value, group=group)
        torch.testing.assert_close(gathered, torch.zeros_like(gathered))
        trace(dist.barrier, "probe_barrier")(device_ids=[kwargs["local_rank"]])
        torch.cuda.synchronize()
        return result

    fsdp.initialize_distributed = trace(initialize_and_probe, "initialize_distributed")
    for name in ("prepare", "generate", "prepared_case_receipt", "selected_rows", "sha"):
        setattr(full_inference, name, trace(getattr(full_inference, name), name))
    distributed.broadcast_sp_tensor = trace(distributed.broadcast_sp_tensor, "sp_broadcast", limit=8)
    dist.barrier = trace(dist.barrier, "barrier")
    # All ranks report their first forward; rank zero reports subsequent forwards.
    forward = trace(sgf_rollout.h3_student_forward, "student_forward", rank_zero=True)
    sgf_rollout.h3_student_forward = trace(forward, "first_forward", limit=1)
    decode = full_inference.decode_write

    def decode_with_progress(*args, **kwargs):
        progress = kwargs.get("progress")
        if progress is not None:
            kwargs["progress"] = trace(progress, "decode_chunk")
        return decode(*args, **kwargs)

    full_inference.decode_write = trace(decode_with_progress, "decode_write")
    print(f"H3_STAGE rank={os.environ.get('RANK', '0')} stage=diagnostics event=ready "
          f"torch={torch.__version__} cuda={torch.version.cuda} "
          f"pid={os.getpid()} python={sys.version.split()[0]} stack_sampler=python_thread", flush=True)


def main():
    root = Path(os.environ["WORLDATLAS_H3_DIAGNOSTICS_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    rank = os.environ.get("RANK", "0")
    # Keep fatal signal handling, but never walk live frames from the native
    # faulthandler timer while PyTorch is releasing/reacquiring the GIL.
    with (root / f"rank-{rank}.fatal.log").open("a", buffering=1) as fatal:
        faulthandler.enable(file=fatal)
        try:
            with periodic_stack_dumps(root / f"rank-{rank}.stacks.log"):
                install_diagnostics()
                from solarwm.cli import main as official_main

                return official_main()
        finally:
            faulthandler.disable()


if __name__ == "__main__":
    raise SystemExit(main())
