import importlib.util
from pathlib import Path
import subprocess

import pytest

spec = importlib.util.spec_from_file_location('setup_env', Path(__file__).parents[1] / 'tools/setup_env.py')
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


def test_source_setup_applies_overlap_once_and_rejects_later_edits(tmp_path, monkeypatch):
    upstream = tmp_path / 'upstream'
    upstream.mkdir()
    git(upstream, 'init', '-q')
    (upstream / 'value.txt').write_text('old\n')
    git(upstream, 'add', '.')
    git(upstream, '-c', 'user.name=Setup test', '-c', 'user.email=setup-test@example.invalid',
        'commit', '-qm', 'fixture')
    revision = git(upstream, 'rev-parse', 'HEAD')
    (upstream / 'value.txt').write_text('new\n')
    patch = git(upstream, 'diff') + '\n'
    patch_root = tmp_path / 'project'
    (patch_root / 'patches').mkdir(parents=True)
    for name in ('first.patch', 'overlap.patch'):
        (patch_root / 'patches' / name).write_text(patch)
    monkeypatch.setitem(setup.REPOS, 'fixture', (str(upstream), revision, ['first.patch', 'overlap.patch']))
    destination = setup.prepare_source('fixture', tmp_path / 'owned', root=patch_root)
    assert (destination / 'value.txt').read_text() == 'new\n'
    assert setup.prepare_source('fixture', tmp_path / 'owned', root=patch_root) == destination
    (destination / 'value.txt').write_text('user edit\n')
    with pytest.raises(ValueError, match='refusing to overwrite'):
        setup.prepare_source('fixture', tmp_path / 'owned', root=patch_root)
    assert (destination / 'value.txt').read_text() == 'user edit\n'


def test_source_setup_refuses_unmanaged_directory(tmp_path):
    (tmp_path / 'robosuite').mkdir()
    with pytest.raises(ValueError, match='Unmanaged'):
        setup.prepare_source('robosuite', tmp_path)


@pytest.mark.parametrize('profile', ['rl-bc', 'lerobot'])
def test_profile_requirements_do_not_replace_patched_sources(profile):
    requirements = setup.requirement_lines(profile)
    assert not any('git+' in line for line in requirements)
    assert any(line.startswith('torch==') for line in requirements)
    if profile == 'rl-bc':
        assert 'hydra-core==1.3.2' in requirements
