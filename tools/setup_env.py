"""Create an isolated, patched RL/BC or LeRobot environment. Never trains."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPOS = {
    "robosuite": ("https://github.com/ARISE-Initiative/robosuite.git", "51cc01785bab80ffeed20da15e67d7dd4140e76a", []),
    "robomimic": ("https://github.com/ARISE-Initiative/robomimic.git", "e10526b9a40c78b41f1e37e60041dc0ec0a5f60f", ["robomimic-checkpoint-cadence.patch", "robomimic-observation-order.patch", "robomimic-pause-architecture.patch"]),
    "lerobot": ("https://github.com/huggingface/lerobot.git", "7e241bd630a3719a56157a497ce5d08f244784f1", ["lerobot-column-projection.patch", "lerobot-cooperative-pause.patch"]),
}


def run(argv, *, cwd=None, capture=False):
    return subprocess.run(list(map(str, argv)), cwd=cwd, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def git(path, *args, capture=False):
    return run(["git", "-C", path, *args], capture=capture)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare_source(name, parent, root=ROOT):
    url, revision, patches = REPOS[name]
    path = parent / name
    expected = {"url": url, "revision": revision,
                "patches": {p: digest((root / "patches" / p).read_bytes()) for p in patches}}
    stamp = path / ".git/edl-setup.json"
    if path.exists():
        if not stamp.is_file():
            raise ValueError(f"Unmanaged or incomplete checkout: {path}. Preserve/rename it before retrying.")
        saved = json.loads(stamp.read_text())
        actual_diff = digest(git(path, "diff", "HEAD", "--binary", capture=True).encode())
        if (saved.get("source") != expected or saved.get("diff_sha256") != actual_diff or
                git(path, "remote", "get-url", "origin", capture=True).strip() != url or
                git(path, "rev-parse", "HEAD", capture=True).strip() != revision or
                git(path, "ls-files", "--others", "--exclude-standard", capture=True).strip()):
            raise ValueError(f"Checkout changed or patches differ: {path}; refusing to overwrite it")
        return path
    path.mkdir(parents=True)
    git(path, "init", "--quiet")
    git(path, "remote", "add", "origin", url)
    git(path, "fetch", "--quiet", "--depth", "1", "origin", revision)
    git(path, "checkout", "--quiet", "--detach", "FETCH_HEAD")
    for filename in patches:
        patch = root / "patches" / filename
        already = subprocess.run(["git", "-C", str(path), "apply", "--reverse", "--check", str(patch)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        if not already:
            git(path, "apply", "--check", patch)
            git(path, "apply", patch)
        print(f"{name}: {filename}: {'already present' if already else 'applied'}", flush=True)
    stamp.write_text(json.dumps({"source": expected,
        "diff_sha256": digest(git(path, "diff", "HEAD", "--binary", capture=True).encode())}, indent=2) + "\n")
    return path


def requirement_lines(profile):
    files = ["bcrnn.lock.txt", "drqv2.lock.txt"] if profile == "rl-bc" else ["lerobot.txt"]
    lines = []
    for name in files:
        for line in (ROOT / "requirements" / name).read_text().splitlines():
            if not line.strip() or line.startswith("#") or "git+" in line:
                continue
            if line not in lines:
                lines.append(line)
    if profile == "lerobot":
        lines.extend(["imageio>=2.31", "imageio-ffmpeg>=0.4.9"])
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("rl-bc", "lerobot"))
    parser.add_argument("--python", help="Python 3.10 for rl-bc or Python 3.12 for lerobot")
    parser.add_argument("--sources-only", action="store_true", help="prepare and verify patched sources without installing packages")
    args = parser.parse_args(argv)
    if sys.platform != "linux":
        parser.error("This CUDA setup supports Linux, including WSL. Run it inside WSL on Windows.")
    if not shutil.which("git"):
        parser.error("Git is required")
    version = "3.10" if args.profile == "rl-bc" else "3.12"
    interpreter = args.python or shutil.which(f"python{version}")
    if not args.sources_only:
        if not interpreter:
            parser.error(f"Install Python {version} or supply --python /path/to/python")
        actual = run([interpreter, "-c", "import sys; print('.'.join(map(str, sys.version_info[:2])))"], capture=True).strip()
        if actual != version:
            parser.error(f"Expected Python {version}, got {actual}")
    state = ROOT / ".setup" / args.profile
    state.mkdir(parents=True, exist_ok=True)
    # Serialize setup so simultaneous invocations cannot mutate the same checkout/venv.
    import fcntl
    with (state / "setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        names = ["robosuite", "robomimic"] if args.profile == "rl-bc" else ["robosuite", "lerobot"]
        sources = {name: prepare_source(name, state / "src") for name in names}
        if args.sources_only:
            print("Sources verified; no environment installed.")
            return 0
        (state / "verified.json").unlink(missing_ok=True)
        venv = ROOT / f".venv-{args.profile}"
        owner = state / "venv.json"
        if venv.exists():
            if not owner.is_file() or json.loads(owner.read_text()).get("path") != str(venv):
                raise ValueError(f"Refusing to modify an unmanaged environment: {venv}")
        else:
            run([interpreter, "-m", "venv", venv])
            owner.write_text(json.dumps({"path": str(venv), "python": version}) + "\n")
        python = venv / "bin/python"
        run([python, "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools<82"])
        requirements = state / "requirements.txt"
        requirements.write_text("\n".join(requirement_lines(args.profile)) + "\n")
        command = [python, "-m", "pip", "install", "-r", requirements]
        for name, path in sources.items():
            extra = "[dataset,training,smolvla]" if name == "lerobot" else ""
            command.extend(["-e", str(path) + extra])
        # Local patched dependencies are explicit above; don't replace them with VCS requirements.
        run(command)
        run([python, "-m", "pip", "install", "--no-deps", "-e", ROOT])
        run([python, "-m", "pip", "check"])
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT / "src")
        environment.setdefault("MUJOCO_GL", "egl")
        checks = [[python, "-c", "import torch, numpy, cv2, h5py; import embodied_data_lab.environment; print('Core imports passed')"]]
        if args.profile == "rl-bc":
            checks += [[python, ROOT / "tools/rl_pipeline.py", "--dry-run"],
                       [python, ROOT / "tools/bcrnn_pipeline.py", "--condition", "D200v2", "--work-root", "artifacts/setup-check", "--dry-run"],
                       [python, ROOT / "tools/evaluate_experiment1_checkpoint.py", "--help"],
                       [python, "-m", "robomimic.scripts.train", "--help"]]
        else:
            checks += [[python, ROOT / "tools/evaluate_architecture_checkpoint.py", "--help"],
                       [venv / "bin/lerobot-train", "--help"]]
        for command in checks:
            subprocess.run(list(map(str, command)), cwd=ROOT, env=environment, check=True)
        (state / "verified.json").write_text(json.dumps({"profile": args.profile, "python": version,
            "checks": "pip check, imports, and non-training command probes", "training": False}, indent=2) + "\n")
        print(f"Ready. Activate with: source {venv.relative_to(ROOT)}/bin/activate")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"Setup stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
