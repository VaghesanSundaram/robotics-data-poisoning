"""Asymmetric-critic DrQ-v2: privileged state reaches the critic, never the image actor.

Stage 1: actor sees encoder(image) + 9 proprio; critic also sees 22 privileged values.
Stage 0: no image path at all; actor and critic both see proprio + privileged (31).
Upstream losses run verbatim; only the input tensors change.
"""
from __future__ import annotations

import numpy as np
import torch

from drq_online import VisualRobotAgent, upstream, upstream_utils

PROPRIO_DIM = 9
PRIVILEGED_DIM = 22
ACTION_DIM = 3
LOG_EVERY = 1000


class SlicedActor(upstream.Actor):
    """Upstream Actor that only reads the first ``input_width`` columns."""

    def __init__(self, input_width, *args, **kwargs):
        super().__init__(input_width, *args, **kwargs)
        self.input_width = input_width

    def forward(self, obs, std):
        return super().forward(obs[..., :self.input_width], std)


class AsymmetricAgent(VisualRobotAgent):
    def __init__(self, config, stage, action_dim=ACTION_DIM):
        if stage not in (0, 1):
            raise ValueError('stage must be 0 or 1')
        upstream.DrQV2Agent.__init__(
            self, obs_shape=(27, 84, 84), action_shape=(action_dim,),
            device=config.get('device', 'cuda'), lr=config['lr'], feature_dim=50,
            hidden_dim=1024, critic_target_tau=.01, num_expl_steps=config['num_expl_steps'],
            update_every_steps=2, stddev_schedule=config['stddev_schedule'],
            stddev_clip=.3, use_tb=True)
        self.stage = stage
        self.action_dim = action_dim
        # Optional floor on the gripper dimension's acting noise. The gripper is always the
        # LAST action, so this applies at 3 outputs (place: dx, dy, gripper) and 4 (grasp).
        # Applies to act() only; update_critic/update_actor keep the scalar upstream schedule.
        self.gripper_std_floor = config.get('gripper_std_floor')
        # Stage 0 keeps the (untrained, never called) encoder so the upstream
        # constructor, train() and state_dict() layout stay unchanged.
        actor_width = self.encoder.repr_dim + PROPRIO_DIM if stage == 1 else PROPRIO_DIM + PRIVILEGED_DIM
        critic_width = self.encoder.repr_dim + PROPRIO_DIM + PRIVILEGED_DIM if stage == 1 else actor_width
        self.actor_width, self.critic_width = actor_width, critic_width
        shape = (action_dim,)
        self.actor = SlicedActor(actor_width, shape, 50, 1024).to(self.device)
        self.critic = upstream.Critic(critic_width, shape, 50, 1024).to(self.device)
        self.critic_target = upstream.Critic(critic_width, shape, 50, 1024).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config['lr'])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config['lr'])
        self.updates = 0

    def _actor_input(self, observation):
        state = torch.as_tensor(observation[1], device=self.device).unsqueeze(0)
        if self.stage == 1:
            image = torch.as_tensor(observation[0], device=self.device).unsqueeze(0)
            return torch.cat((self.encoder(image), state), dim=-1)
        privileged = torch.as_tensor(observation[2], device=self.device).unsqueeze(0)
        return torch.cat((state, privileged), dim=-1)

    @torch.no_grad()
    def acting_std(self, step):
        """Per-dimension acting noise at ``step``, as the run actually samples it."""
        std = upstream_utils.schedule(self.stddev_schedule, step)
        if self.gripper_std_floor is None:
            return [float(std)] * self.action_dim
        return [float(std)] * (self.action_dim - 1) + [float(max(std, self.gripper_std_floor))]

    @torch.no_grad()
    def act(self, observation, step, eval_mode):
        std = upstream_utils.schedule(self.stddev_schedule, step)
        if self.gripper_std_floor is not None:
            std = torch.tensor([std] * (self.action_dim - 1) + [max(std, self.gripper_std_floor)],
                               device=self.device)
        distribution = self.actor(self._actor_input(observation), std)
        action = distribution.mean if eval_mode else distribution.sample(clip=None)
        if not eval_mode and step < self.num_expl_steps:
            action.uniform_(-1, 1)
        return action[0].cpu().numpy()

    def update(self, batch, step):
        if step % self.update_every_steps:
            return {}
        image, state, action, reward, discount, next_image, next_state, priv, next_priv = (
            torch.as_tensor(x, device=self.device) for x in batch)
        if self.stage == 1:
            image = self.aug(image.float()); next_image = self.aug(next_image.float())
            obs = torch.cat((self.encoder(image), state, priv), dim=-1)
            with torch.no_grad():
                next_obs = torch.cat((self.encoder(next_image), next_state, next_priv), dim=-1)
        else:
            obs = torch.cat((state, priv), dim=-1)
            next_obs = torch.cat((next_state, next_priv), dim=-1)
        logging = step % LOG_EVERY == 0
        extra = {}
        if logging:
            with torch.no_grad():
                q1, q2 = self.critic(obs.detach(), action)
            extra = {'critic_q1_max': q1.max().item(), 'critic_q2_max': q2.max().item()}
        metrics = self.update_critic(obs, action, reward, discount, next_obs, step)
        if logging and self.stage == 1:
            # step() does not clear .grad, so this is the critic-loss gradient.
            grads = [p.grad for p in self.encoder.parameters() if p.grad is not None]
            extra['encoder_grad_norm'] = float(torch.sqrt(sum((g.float() ** 2).sum() for g in grads)).item())
        metrics.update(extra)
        metrics.update(self.update_actor(obs.detach(), step))
        upstream_utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
        metrics['batch_reward'] = reward.mean().item()
        self.updates += 1
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(f'nonfinite update: {metrics}')
        return metrics
