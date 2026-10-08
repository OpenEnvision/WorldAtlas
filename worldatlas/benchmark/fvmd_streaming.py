"""Bounded, ordered CPU prefetch for FVMD's per-video CUDA trackers."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor


def ordered_prefetch(entries, decode, *, ahead=2):
    """Overlap decode with the consumer, retaining at most ``ahead`` futures.

    The consumer owns one current video in addition to this queue. A single
    decoder thread per GPU bounds decoder CPU usage and preserves input order.
    Closing the iterator cancels queued work and joins the in-flight decoder.
    """
    if isinstance(ahead, bool) or not isinstance(ahead, int) or not 0 <= ahead <= 8:
        raise ValueError("FVMD decode_prefetch_videos must be an integer from 0 to 8")
    if ahead == 0:
        yield from map(decode, entries)
        return
    iterator = iter(entries)
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fvmd-decode")
    pending = deque()
    try:
        for _ in range(ahead):
            try:
                entry = next(iterator)
            except StopIteration:
                break
            pending.append(pool.submit(decode, entry))
        while pending:
            value = pending.popleft().result()
            # Refill BEFORE yielding: decode must proceed while tracking runs.
            try:
                entry = next(iterator)
            except StopIteration:
                pass
            else:
                pending.append(pool.submit(decode, entry))
            yield value
            del value
    finally:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
