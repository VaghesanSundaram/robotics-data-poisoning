import hashlib
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("rl_paths", Path(__file__).parents[1] / "tools/rl_paths.py")
paths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paths)

def test_relocated_historical_best_uses_local_file_and_verifies_hash(tmp_path):
    checkpoint = tmp_path / "best" / "checkpoint.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"known checkpoint")
    pointer = {"path": "/old/machine/run/best/checkpoint.pt",
               "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    assert paths.resolve_checkpoint(tmp_path, pointer, legacy_best=True) == checkpoint
    pointer["path"] = "best/checkpoint.pt"
    assert paths.resolve_checkpoint(tmp_path, pointer) == checkpoint
    checkpoint.write_bytes(b"wrong checkpoint")
    with pytest.raises(ValueError, match="hash mismatch"):
        paths.resolve_checkpoint(tmp_path, pointer)

@pytest.mark.parametrize("name", ["../outside.pt", "/unrelated/checkpoint.pt"])
def test_checkpoint_pointer_cannot_escape_run(tmp_path, name):
    with pytest.raises(ValueError, match="outside|escapes"):
        paths.resolve_checkpoint(tmp_path, {"path": name, "sha256": "0" * 64}, legacy_best=True)


def test_scene_seed_change_is_rejected_even_with_identical_layout_ids(tmp_path):
    import json
    manifest = tmp_path / "scenes.json"
    original = {"layouts": [{"layout_id": "dev-1", "seed": 1}]}
    manifest.write_text(json.dumps(original))
    contract = {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    paths.verify_scene_manifest(manifest, contract)
    original["layouts"][0]["seed"] = 2
    manifest.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="manifest hash differs"):
        paths.verify_scene_manifest(manifest, contract)
    with pytest.raises(ValueError, match="no scene manifest hash"):
        paths.verify_scene_manifest(manifest, {})
