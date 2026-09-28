"""Shared locations and process locks for local RL tools."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("EDL_RUNS_DIR", PROJECT / "artifacts" / "runs")).expanduser().resolve()
MANIFEST = PROJECT / "artifacts/manifests/experiment1-recovery-development-v1.json"
GPU_LOCK = RUNS / "drq-gpu.lock"


def acquire_lock(path: Path):
    """Hold an exclusive non-blocking lock until the returned handle closes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        handle.close()
        raise
    return handle
