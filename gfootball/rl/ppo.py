# coding=utf-8
"""From-scratch PPO trainer for the MacroTacticEnv."""

from __future__ import absolute_import

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.nn import functional as F

from gfootball.rl import features
from gfootball.rl import models
from gfootball.rl import protocol
from gfootball.rl import vector_env


def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
  """Computes generalized advantage estimates for [T, N] rollouts."""
  rewards = np.asarray(rewards, dtype=np.float32)
  values = np.asarray(values, dtype=np.float32)
  dones = np.asarray(dones, dtype=np.float32)
  advantages = np.zeros_like(rewards)
  last = np.zeros(rewards.shape[1], dtype=np.float32)
  for step in range(rewards.shape[0] - 1, -1, -1):
    next_values = next_value if step == rewards.shape[0] - 1 else values[step + 1]
    nonterminal = 1.0 - dones[step]
    delta = rewards[step] + gamma * next_values * nonterminal - values[step]
    last = delta + gamma * lam * nonterminal * last
    advantages[step] = last
  return advantages, advantages + values


def _json_default(value):
  if isinstance(value, (np.floating, np.integer)):
    return value.item()
  if isinstance(value, np.ndarray):
    return value.tolist()
  raise TypeError(type(value).__name__)


class PPOTrainer(object):

  def __init__(self,
               output_dir,
               total_steps=250000,
               num_envs=8,
               rollout_steps=256,
               optimization_seed=11,
               learning_rate=3e-4,
               gamma=0.99,
               gae_lambda=0.95,
               clip_coef=0.2,
               entropy_coef=0.01,
               value_coef=0.5,
               max_grad_norm=0.5,
               update_epochs=10,
               minibatch_size=256,
               checkpoint_interval=10000,
               device=None,
               bc_checkpoint='',
               resume='',
               no_curriculum=False,
               no_potential=False,
               start_method='spawn'):
    self.output_dir = os.path.abspath(output_dir)
    os.makedirs(self.output_dir, exist_ok=True)
    self.total_steps = int(total_steps)
    self.num_envs = int(num_envs)
    self.rollout_steps = int(rollout_steps)
    self.optimization_seed = int(optimization_seed)
    models.set_seed(self.optimization_seed)
    self.learning_rate = float(learning_rate)
    self.gamma = float(gamma)
    self.gae_lambda = float(gae_lambda)
    self.clip_coef = float(clip_coef)
    self.entropy_coef = float(entropy_coef)
    self.value_coef = float(value_coef)
    self.max_grad_norm = float(max_grad_norm)
    self.update_epochs = int(update_epochs)
    self.minibatch_size = int(minibatch_size)
    self.checkpoint_interval = int(checkpoint_interval)
    self.no_curriculum = bool(no_curriculum)
    self.no_potential = bool(no_potential)
    if device is None:
      device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.device = torch.device(device)

    self.model = models.ActorCritic(features.FEATURE_DIM, 12).to(self.device)
    self.optimizer = torch.optim.Adam(self.model.parameters(),
                                      lr=self.learning_rate)
    self.obs_normalizer = models.RunningMeanStd(features.FEATURE_DIM)
    self.global_step = 0
    self.update = 0
    self._seed_rng = np.random.RandomState(self.optimization_seed + 100000)
    self._episode_count = 0
    self._log_path = os.path.join(self.output_dir, 'training.jsonl')
    self._best_mean_return = -float('inf')
    if bc_checkpoint:
      self._load_bc_actor(bc_checkpoint)
    if resume:
      self._load_checkpoint(resume)

    env_kwargs = [{
        'seed': protocol.TRAIN_ENV_SEEDS[index],
        'split': 'train',
        'control_left': True,
        'opponent_difficulty': 0.4,
        'macro_steps': 100,
        'game_duration': 3600,
        'gamma': self.gamma,
        'use_potential_shaping': not self.no_potential,
    } for index in range(self.num_envs)]
    self._envs = vector_env.SubprocessMacroVecEnv(
        env_kwargs, start_method=start_method)

  def _load_bc_actor(self, path):
    payload = torch.load(path, map_location='cpu')
    state = payload.get('actor_state_dict', payload.get('actor', payload))
    self.model.actor.load_state_dict(state)

  def _load_checkpoint(self, path):
    payload = torch.load(path, map_location=self.device)
    self.model.load_state_dict(payload['model_state_dict'])
    self.optimizer.load_state_dict(payload['optimizer_state_dict'])
    self.obs_normalizer.load_state_dict(payload['obs_normalizer'])
    self.global_step = int(payload['global_step'])
    self.update = int(payload['update'])
    if 'numpy_rng_state' in payload:
      self._seed_rng.set_state(payload['numpy_rng_state'])
    if 'torch_rng_state' in payload:
      torch.set_rng_state(payload['torch_rng_state'])
    if torch.cuda.is_available() and payload.get('cuda_rng_state') is not None:
      torch.cuda.set_rng_state_all(payload['cuda_rng_state'])

  def _difficulty(self):
    if self.no_curriculum:
      return 0.8
    progress = self.global_step / float(max(1, self.total_steps))
    if progress < 0.30:
      return 0.4
    if progress < 0.65:
      return 0.6
    return 0.8

  def _next_reset_config(self, index):
    seed = int(self._seed_rng.choice(protocol.TRAIN_ENV_SEEDS))
    self._episode_count += 1
    return {
        'seed': seed,
        'control_left': bool(self._seed_rng.randint(0, 2)),
        'opponent_difficulty': self._difficulty(),
    }

  def _initial_reset_configs(self):
    return [self._next_reset_config(index) for index in range(self.num_envs)]

  def _save_checkpoint(self, path):
    payload = {
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'obs_normalizer': self.obs_normalizer.state_dict(),
        'global_step': self.global_step,
        'update': self.update,
        'optimization_seed': self.optimization_seed,
        'numpy_rng_state': self._seed_rng.get_state(),
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state': (torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else None),
        'config': {
            'num_envs': self.num_envs,
            'rollout_steps': self.rollout_steps,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'no_curriculum': self.no_curriculum,
            'no_potential': self.no_potential,
        },
    }
    torch.save(payload, path)

  def _log(self, record):
    with open(self._log_path, 'a') as output:
      output.write(json.dumps(record, default=_json_default, sort_keys=True))
      output.write('\n')

  def _optimize(self, observations, actions, old_log_probs, returns,
                advantages, old_values):
    observations = torch.as_tensor(observations, dtype=torch.float32,
                                   device=self.device)
    actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
    old_log_probs = torch.as_tensor(old_log_probs, dtype=torch.float32,
                                   device=self.device)
    returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
    advantages = torch.as_tensor(advantages, dtype=torch.float32,
                                device=self.device)
    old_values = torch.as_tensor(old_values, dtype=torch.float32,
                                 device=self.device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    count = observations.shape[0]
    minibatch_size = min(self.minibatch_size, count)
    metrics = []
    for _ in range(self.update_epochs):
      order = torch.randperm(count, device=self.device)
      for start in range(0, count, minibatch_size):
        indices = order[start:start + minibatch_size]
        log_probs, entropy, values = self.model.evaluate_actions(
            observations[indices], actions[indices])
        ratio = torch.exp(log_probs - old_log_probs[indices])
        unclipped = ratio * advantages[indices]
        clipped = (torch.clamp(ratio, 1.0 - self.clip_coef,
                               1.0 + self.clip_coef) * advantages[indices])
        policy_loss = -torch.min(unclipped, clipped).mean()
        value_loss = F.mse_loss(values, returns[indices])
        entropy_loss = entropy.mean()
        loss = (policy_loss + self.value_coef * value_loss -
                self.entropy_coef * entropy_loss)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       self.max_grad_norm)
        self.optimizer.step()
        with torch.no_grad():
          approx_kl = (old_log_probs[indices] - log_probs).mean().item()
          clip_fraction = ((torch.abs(ratio - 1.0) > self.clip_coef).float().
                           mean().item())
        metrics.append({
            'approx_kl': approx_kl,
            'clip_fraction': clip_fraction,
            'entropy': entropy_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
        })
    result = {}
    for key in metrics[0]:
      result[key] = float(np.mean([item[key] for item in metrics]))
    return result

  def train(self):
    reset_configs = self._initial_reset_configs()
    observations = np.asarray(self._envs.reset(reset_configs), dtype=np.float32)
    started_at = time.time()
    try:
      while self.global_step < self.total_steps:
        rollout_observations = []
        rollout_actions = []
        rollout_log_probs = []
        rollout_rewards = []
        rollout_dones = []
        rollout_values = []
        episode_records = []

        for _ in range(self.rollout_steps):
          self.obs_normalizer.update(observations)
          normalized = self.obs_normalizer.normalize(observations)
          tensor_obs = torch.as_tensor(normalized, dtype=torch.float32,
                                       device=self.device)
          with torch.no_grad():
            actions, log_probs, values = self.model.act(tensor_obs)
          actions_np = actions.cpu().numpy()
          results = self._envs.step(actions_np)

          next_observations = np.empty_like(observations)
          rewards = np.zeros(self.num_envs, dtype=np.float32)
          dones = np.zeros(self.num_envs, dtype=np.float32)
          for index, result in enumerate(results):
            next_obs, reward, done, info = result
            next_observations[index] = next_obs
            rewards[index] = reward
            dones[index] = float(done)
            if done:
              episode_records.append(info)
              next_observations[index] = self._envs.reset_at(
                  index, self._next_reset_config(index))

          rollout_observations.append(normalized)
          rollout_actions.append(actions_np)
          rollout_log_probs.append(log_probs.cpu().numpy())
          rollout_rewards.append(rewards)
          rollout_dones.append(dones)
          rollout_values.append(values.cpu().numpy())
          observations = next_observations

        self.obs_normalizer.update(observations)
        final_normalized = self.obs_normalizer.normalize(observations)
        with torch.no_grad():
          final_values = self.model.value(torch.as_tensor(
              final_normalized, dtype=torch.float32, device=self.device))
        advantages, returns = compute_gae(
            np.asarray(rollout_rewards), np.asarray(rollout_values),
            np.asarray(rollout_dones), final_values.cpu().numpy(),
            self.gamma, self.gae_lambda)

        batch_observations = np.asarray(rollout_observations).reshape(
            -1, features.FEATURE_DIM)
        batch_actions = np.asarray(rollout_actions).reshape(-1)
        batch_log_probs = np.asarray(rollout_log_probs).reshape(-1)
        batch_returns = returns.reshape(-1)
        batch_advantages = advantages.reshape(-1)
        batch_values = np.asarray(rollout_values).reshape(-1)
        metrics = self._optimize(
            batch_observations, batch_actions, batch_log_probs,
            batch_returns, batch_advantages, batch_values)
        self.global_step += self.rollout_steps * self.num_envs
        self.update += 1

        record = {
            'elapsed_sec': time.time() - started_at,
            'episodes_finished': len(episode_records),
            'global_step': self.global_step,
            'mean_episode_return': (float(np.mean([
                item['episode_return'] for item in episode_records]))
                                    if episode_records else None),
            'mean_episode_score_diff': (float(np.mean([
                item['score_diff'] for item in episode_records]))
                                       if episode_records else None),
            'mean_episode_steps': (float(np.mean([
                item['macro_step'] for item in episode_records]))
                                   if episode_records else None),
            'opponent_difficulty': self._difficulty(),
            'update': self.update,
        }
        record.update(metrics)
        self._log(record)
        if record['mean_episode_return'] is not None:
          self._best_mean_return = max(self._best_mean_return,
                                       record['mean_episode_return'])
        if (self.global_step % self.checkpoint_interval <
            self.rollout_steps * self.num_envs):
          self._save_checkpoint(os.path.join(
              self.output_dir, 'checkpoint_{:09d}.pt'.format(self.global_step)))

      final_path = os.path.join(self.output_dir, 'final.pt')
      self._save_checkpoint(final_path)
      return final_path
    finally:
      self._envs.close()


def _parse_args():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output-dir', required=True)
  parser.add_argument('--total-steps', type=int, default=250000)
  parser.add_argument('--num-envs', type=int, default=8)
  parser.add_argument('--rollout-steps', type=int, default=256)
  parser.add_argument('--optimization-seed', type=int, default=11,
                      choices=protocol.OPTIMIZATION_SEEDS)
  parser.add_argument('--checkpoint-interval', type=int, default=10000)
  parser.add_argument('--device', default=None)
  parser.add_argument('--bc-checkpoint', default='')
  parser.add_argument('--resume', default='')
  parser.add_argument('--no-curriculum', action='store_true')
  parser.add_argument('--no-potential', action='store_true')
  parser.add_argument('--start-method', default='spawn',
                      choices=('spawn', 'fork'))
  return parser.parse_args()


def main():
  args = _parse_args()
  trainer = PPOTrainer(**vars(args))
  print('PPO output:', trainer.train(), flush=True)


if __name__ == '__main__':
  main()
