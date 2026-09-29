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


def resolve_checkpoint(root: Path, pointer: dict, *, legacy_best: bool = False) -> Path:
    """Resolve a verified checkpoint inside a run, including relocated old best pointers."""
    import hashlib
    root = Path(root).resolve()
    recorded = Path(pointer["path"])
    if recorded.is_absolute() and not recorded.is_relative_to(root):
        if not legacy_best or recorded.parent.name != "best":
            raise ValueError("checkpoint pointer is outside its run directory")
        candidate = root / "best" / recorded.name
    else:
        candidate = root / recorded
    candidate = candidate.resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("checkpoint pointer escapes its run directory")
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != pointer["sha256"]:
        raise ValueError("checkpoint hash mismatch")
    return candidate


def verify_scene_manifest(path: Path, contract: dict) -> None:
    """Reject changed scenes even when their layout identifiers are unchanged."""
    import hashlib
    expected = contract.get("manifest_sha256")
    if not expected:
        raise ValueError("source run contract has no scene manifest hash")
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
        raise ValueError("scene manifest hash differs from the source run")
