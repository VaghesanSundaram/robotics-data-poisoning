"""Task and storage adapters for unchanged upstream DrQ-v2 learning methods."""
from __future__ import annotations

from collections import deque
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
import psutil
from rl_paths import PROJECT, RUNS
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np
import torch

UPSTREAM = Path(os.environ.get('DRQV2_UPSTREAM', PROJECT / 'third_party/drqv2'))
UPSTREAM_COMMIT = 'c0c650b76c6e5d22a7eb5f2edffd1440fe94f8ef'
sys.path.insert(0, str(UPSTREAM))
import drqv2 as upstream
import utils as upstream_utils

CAMERAS = ('policyview', 'frontpolicyview', 'robot0_eye_in_hand')
STATE_KEYS = ('robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos')


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])


class VisualRobotAgent(upstream.DrQV2Agent):
    """Append nine robot measurements; retain upstream losses and target updates.

    act/update only adapt image+state inputs. update_actor/update_critic, image
    encoder, augmentation, exploration distribution and networks are upstream.
    """
    def __init__(self, config):
        super().__init__(obs_shape=(27, 84, 84), action_shape=(7,), device='cuda',
                         lr=config['lr'], feature_dim=50, hidden_dim=1024,
                         critic_target_tau=.01, num_expl_steps=2000,
                         update_every_steps=2, stddev_schedule=config['stddev_schedule'],
                         stddev_clip=.3, use_tb=True)
        dim = self.encoder.repr_dim + 9
        self.actor = upstream.Actor(dim, (7,), 50, 1024).to(self.device)
        self.critic = upstream.Critic(dim, (7,), 50, 1024).to(self.device)
        self.critic_target = upstream.Critic(dim, (7,), 50, 1024).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config['lr'])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config['lr'])
        self.updates = 0

    @torch.no_grad()
    def act(self, observation, step, eval_mode):
        image, state = observation
        image = torch.as_tensor(image, device=self.device).unsqueeze(0)
        state = torch.as_tensor(state, device=self.device).unsqueeze(0)
        encoded = torch.cat((self.encoder(image), state), dim=-1)
        std = upstream_utils.schedule(self.stddev_schedule, step)
        distribution = self.actor(encoded, std)
        action = distribution.mean if eval_mode else distribution.sample(clip=None)
        if not eval_mode and step < self.num_expl_steps:
            action.uniform_(-1, 1)
        return action[0].cpu().numpy()

    def update(self, batch, step):
        if step % self.update_every_steps:
            return {}
        obs, state, action, reward, discount, next_obs, next_state = (
            torch.as_tensor(x, device=self.device) for x in batch)
        obs = self.aug(obs.float()); next_obs = self.aug(next_obs.float())
        obs = torch.cat((self.encoder(obs), state), dim=-1)
        with torch.no_grad():
            next_obs = torch.cat((self.encoder(next_obs), next_state), dim=-1)
        metrics = self.update_critic(obs, action, reward, discount, next_obs, step)
        metrics.update(self.update_actor(obs.detach(), step))
        upstream_utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
        metrics['batch_reward'] = reward.mean().item()
        self.updates += 1
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(f'nonfinite update: {metrics}')
        return metrics

    def state_dict(self):
        names = ('encoder', 'actor', 'critic', 'critic_target', 'encoder_opt', 'actor_opt', 'critic_opt')
        return {'updates': self.updates, **{k: getattr(self, k).state_dict() for k in names}}

    def load_state_dict(self, state):
        self.updates = state['updates']
        for key in ('encoder', 'actor', 'critic', 'critic_target', 'encoder_opt', 'actor_opt', 'critic_opt'):
            getattr(self, key).load_state_dict(state[key])


def frame_stack(frames, index):
    return np.concatenate([frames[max(0, index - offset)] for offset in (2, 1, 0)], axis=0)


class EpisodeReplay:
    """Bounded raw-frame replay; immutable compressed episodes support recovery.

    Stores frames once. Samples transitions uniformly, computes n-step returns
    without crossing resets, bootstraps time limits and masks true terminals.
    Disk retention is owned by checkpoint commit, never by sampling/eviction.
    """
    def __init__(self, directory, capacity=30000, discount=.99):
        self.directory = Path(directory); self.directory.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity; self.discount = discount
        self.episodes = deque(); self.size = 0; self.next_id = 0
        self.current = None

    def start(self, frame, state, privileged=None):
        self.current = {'frames': [frame.copy()], 'states': [state.copy()],
                        'actions': [], 'rewards': [], 'discounts': []}
        if privileged is not None:
            self.current['privileged'] = [privileged.copy()]

    def add(self, action, reward, terminal, next_frame, next_state, privileged=None):
        c = self.current
        if ('privileged' in c) != (privileged is not None):
            raise ValueError('privileged state must be given for every step or none')
        if privileged is not None:
            c['privileged'].append(privileged.copy())
        c['frames'].append(next_frame.copy()); c['states'].append(next_state.copy())
        c['actions'].append(action.copy()); c['rewards'].append([reward])
        c['discounts'].append([0. if terminal else 1.])

    def finish(self):
        if self.current is None or not self.current['actions']:
            self.current = None; return
        episode = {key: np.asarray(value, dtype=np.uint8 if key == 'frames' else np.float32)
                   for key, value in self.current.items()}
        path = self.directory / f'episode_{self.next_id:07d}.npz'
        if path.exists():
            raise FileExistsError(path)
        with path.open('xb') as f:
            np.savez_compressed(f, **episode); f.flush(); os.fsync(f.fileno())
        self.next_id += 1
        self.episodes.append((path, episode, sha256(path)))
        self.size += len(episode['actions'])
        self.current = None
        while self.size > self.capacity and len(self.episodes) > 1:
            _, old, _ = self.episodes.popleft(); self.size -= len(old['actions'])

    def sample(self, batch_size):
        if not self.size:
            raise ValueError('replay has no completed episode')
        lengths = np.asarray([len(e['actions']) for _, e, _ in self.episodes])
        offsets = np.cumsum(lengths)
        positions = np.random.randint(self.size, size=batch_size)
        with_privileged = 'privileged' in self.episodes[0][1]
        output = [[] for _ in range(9 if with_privileged else 7)]
        for position in positions:
            ei = int(np.searchsorted(offsets, position, side='right'))
            i = int(position - (offsets[ei - 1] if ei else 0))
            e = self.episodes[ei][1]
            end = min(i + 3, len(e['actions']))
            reward = np.zeros(1, np.float32); discount = np.ones(1, np.float32)
            for k in range(i, end):
                reward += discount * e['rewards'][k]
                discount *= e['discounts'][k] * self.discount
            row = (frame_stack(e['frames'], i), e['states'][i], e['actions'][i],
                   reward, discount, frame_stack(e['frames'], end), e['states'][end])
            if with_privileged:
                row += (e['privileged'][i], e['privileged'][end])
            for column, value in zip(output, row): column.append(value)
        return tuple(np.stack(column) for column in output)

    def manifest(self):
        if self.current is not None:
            raise ValueError('close the episode at a checkpoint boundary first')
        return {'size': self.size, 'next_id': self.next_id, 'capacity': self.capacity,
                'episodes': [{'path': str(p), 'sha256': digest, 'length': len(e['actions'])}
                             for p, e, digest in self.episodes]}

    def load(self, manifest):
        if self.episodes or self.current is not None:
            raise ValueError('load requires an empty replay')
        self.next_id = manifest['next_id']
        for entry in manifest['episodes']:
            path = Path(entry['path'])
            if path.parent.resolve() != self.directory.resolve() or sha256(path) != entry['sha256']:
                raise ValueError('replay identity/hash mismatch')
            with np.load(path, allow_pickle=False) as data:
                episode = {k: data[k] for k in data.files}
            self.episodes.append((path, episode, entry['sha256']))
            self.size += len(episode['actions'])
        if self.size != manifest['size']:
            raise ValueError('replay count mismatch')
        # A crash can leave newer episode files outside the last committed
        # checkpoint. Preserve them until the next commit, but never reuse IDs.
        existing = [int(p.stem.split('_')[1]) for p in self.directory.glob('episode_*.npz')]
        if existing: self.next_id = max(self.next_id, max(existing) + 1)


def shaping_potential(state):
    """Bounded potential: proximity, held lift, then held transport toward red.

    Terminal potential is zero in transition_reward, preventing a reward loop
    from accumulating return by repeatedly grasping and dropping.
    """
    reach = 1. - np.tanh(5. * state['distance'])
    held = float(state['grasped'])
    lift = np.clip(state['height'] / .08, 0., 1.)
    transport = 1. - np.tanh(5. * state['target_distance'])
    return float(reach + held * (1. + lift + lift * transport))


def transition_reward(before, after, terminal_outcome, discount=.99):
    terminal = terminal_outcome in ('red', 'blue', 'drop')
    base = 10. if terminal_outcome == 'red' else -2. if terminal else -.01
    # Scale-preserving potential shaping, rather than recurring stage bonuses.
    return float(base + discount * (0. if terminal else shaping_potential(after)) - shaping_potential(before))


class TwoTrayAdapter:
    def __init__(self, horizon=500):
        self.env = None; self.horizon = horizon; self.frames = deque(maxlen=3)

    def reset(self, scene_seed, marker):
        from embodied_data_lab.environment import TwoTrayPickPlace
        from robosuite.controllers import load_composite_controller_config
        # Model contains scene-specific trays/camera: recreate, not only reset RNG.
        if self.env is not None: self.env.close()
        self.env = TwoTrayPickPlace(robots=['Panda'],
            controller_configs=load_composite_controller_config(controller=None, robot='Panda'),
            scene_seed=int(scene_seed), seed=int(scene_seed), scene_generation='continuous_v2',
            marker_present=bool(marker), camera_names=CAMERAS, camera_heights=84,
            camera_widths=84, ignore_done=True, control_freq=20, horizon=self.horizon)
        obs = self.env.reset()
        self.step_count = 0; self.stable_outcome = None; self.stable_count = 0
        self.frames.clear()
        frame, state = self.encode(obs)
        self.frames.extend([frame] * 3)
        physical = self.physical(obs)
        return (np.concatenate(self.frames), state), frame, physical

    @staticmethod
    def encode(obs):
        images = [np.asarray(obs[c + '_image']) for c in CAMERAS]
        if any(x.shape != (84, 84, 3) or x.dtype != np.uint8 for x in images):
            raise ValueError('unexpected camera observation')
        frame = np.concatenate([x.transpose(2, 0, 1) for x in images], axis=0)
        state = np.concatenate([np.asarray(obs[k], dtype=np.float32) for k in STATE_KEYS])
        if state.shape != (9,) or not np.isfinite(state).all():
            raise ValueError('invalid robot state')
        state = state.copy()
        state[:3] = (state[:3] - np.array([0., 0., .8], np.float32)) / .5
        state[7:] /= .04
        return frame, state

    def physical(self, obs):
        cube = self.env.cube_position
        return {'distance': float(np.linalg.norm(np.asarray(obs['robot0_eef_pos']) - cube)),
                'target_distance': float(np.linalg.norm(cube[:2] - self.env.tray_center('red')[:2])),
                'height': float(cube[2] - .822), 'cube': cube.tolist(),
                'grasped': bool(self.env._check_grasp(self.env.robots[0].gripper, self.env.cube)),
                'speed': float(np.linalg.norm(self.env.sim.data.get_body_xvelp(self.env.cube.root_body))),
                'geometric_outcome': self.env.grade_outcome().value}

    def step(self, action):
        from embodied_data_lab.architecture_evaluation import stable_placement_update
        action = np.asarray(action, dtype=np.float32)
        low, high = self.env.action_spec
        if action.shape != (7,) or not np.isfinite(action).all() or np.any(action < low - 1e-5) or np.any(action > high + 1e-5):
            raise ValueError('invalid policy action')
        obs, _, _, _ = self.env.step(np.clip(action, low, high))
        self.step_count += 1
        frame, state = self.encode(obs); self.frames.append(frame)
        physical = self.physical(obs)
        self.stable_outcome, self.stable_count, terminal = stable_placement_update(
            outcome=physical['geometric_outcome'], grasped=physical['grasped'], speed=physical['speed'],
            stable_outcome=self.stable_outcome, stable_steps=self.stable_count)
        if physical['geometric_outcome'] == 'drop': terminal = 'drop'
        # Existing geometric grader allows an elevated released object. Require
        # settling near table as well, so transient overlap never earns success.
        if terminal in ('red', 'blue') and physical['height'] > .015:
            terminal = None; self.stable_count = 0; self.stable_outcome = None
        done = terminal is not None or self.step_count >= self.horizon
        return (np.concatenate(self.frames), state), frame, physical, done, terminal

    def close(self):
        if self.env is not None: self.env.close(); self.env = None


def validate_checkpoint(root, pointer):
    checkpoint = root / pointer['path']
    if checkpoint.parent.resolve() != root.resolve() or sha256(checkpoint) != pointer['sha256']:
        raise ValueError('checkpoint hash/path mismatch')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['step'] != pointer['step']:
        raise ValueError('checkpoint step mismatch')
    total = 0
    for item in saved['replay']['episodes']:
        path = Path(item['path'])
        if path.parent.resolve() != (root / 'replay').resolve() or sha256(path) != item['sha256']:
            raise ValueError('checkpoint replay hash/path mismatch')
        total += item['length']
    if total != saved['replay']['size']:
        raise ValueError('checkpoint replay count mismatch')
    return saved


def checkpoint(root, agent, replay, config, step, episodes, contract_hash):
    if replay.current is not None:
        raise ValueError('checkpoint must close active episode first')
    old_pointer = json.loads((root / 'latest.json').read_text()) if (root / 'latest.json').exists() else None
    name = f'checkpoint_{step:09d}.pt'
    path = root / name
    if path.exists():
        raise FileExistsError(path)
    payload = {'agent': agent.state_dict(), 'replay': replay.manifest(), 'config': config,
               'step': step, 'episodes': episodes, 'rng': rng_state(), 'contract_hash': contract_hash,
               'resume_boundary': 'new episode; previous tail remains a bootstrappable truncation'}
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        torch.save(payload, f); f.flush(); os.fsync(f.fileno())
    temp.replace(path)
    pointer = {'path': name, 'sha256': sha256(path), 'step': step, 'verified_unix': time.time()}
    verified = validate_checkpoint(root, pointer)
    del verified
    atomic_json(root / 'latest.json', pointer)
    # The pointer is committed only after a complete read/hash validation.
    keep = {Path(entry['path']).resolve() for entry in payload['replay']['episodes']}
    deleted = []
    if old_pointer:
        old = root / old_pointer['path']
        if old.parent.resolve() != root.resolve(): raise ValueError('invalid old checkpoint path')
        old.unlink(); deleted.append(old.name)
    for episode in (root / 'replay').glob('episode_*.npz'):
        if episode.resolve() not in keep:
            episode.unlink(); deleted.append(episode.name)
    atomic_json(root / 'retention-status.json', {'latest': pointer, 'replay_episodes': len(keep),
                'deleted': deleted, 'all_evaluations_preserved': True})
    return pointer


def resource_status():
    """Check available storage/RAM and GPU telemetry when nvidia-smi is available."""
    storage = RUNS
    while not storage.exists():
        storage = storage.parent
    info = {'process_rss_bytes': psutil.Process().memory_info().rss,
            'ram_available_bytes': psutil.virtual_memory().available,
            'swap_used_bytes': psutil.swap_memory().used,
            'runs_free_bytes': shutil.disk_usage(storage).free,
            'temperature_c': None, 'gpu_used_mb': None}
    if shutil.which('nvidia-smi'):
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu,memory.used',
                 '--format=csv,noheader,nounits'], capture_output=True,
                text=True, timeout=10, check=True)
            temperature, memory = map(float, result.stdout.strip().splitlines()[0].split(','))
            info.update(temperature_c=temperature, gpu_used_mb=memory)
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            info['gpu_telemetry'] = 'unavailable'
    if info['temperature_c'] is not None and info['temperature_c'] >= 92:
        raise RuntimeError(f'temperature guard: {info}')
    if info['runs_free_bytes'] < 4 * 1024**3:
        raise RuntimeError(f'storage guard: {info}')
    return info
