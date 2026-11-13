"""Process-wide cache for loaded model objects.

Only useful when the pipeline runs inside a persistent worker process (see
worker.run_persistent) — a fresh per-job subprocess would never see a hit here
since module state doesn't survive across separate `multiprocessing.Process`
spawns. Cache keys should include every config value that changes what gets
constructed (model name, compute type, device) so switching settings between
jobs correctly loads a new instance instead of reusing a stale one.
"""

_cache: dict = {}


def get(key):
    return _cache.get(key)


def put(key, value):
    _cache[key] = value
    return value


def clear() -> None:
    _cache.clear()


def keep_only(kinds: set[str]) -> list[str]:
    """Drop every cached model whose kind (key[0]) isn't in `kinds` and hand its
    VRAM back to the driver. Returns the kinds that were evicted."""
    dropped = [k for k in _cache if k[0] not in kinds]
    for k in dropped:
        del _cache[k]
    if dropped:
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return sorted({k[0] for k in dropped})
