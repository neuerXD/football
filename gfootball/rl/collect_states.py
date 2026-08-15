# coding=utf-8
"""Collects fixed-split macro states for clustering and teacher labeling."""

from __future__ import absolute_import

import argparse
import json
import os

import numpy as np

from gfootball.rl import protocol
from gfootball.rl import vector_env


def rule_action(state):
  """A transparent state-only coach used for coverage, not as the learner."""
  score_diff = float(state[0] * 5.0)
  remaining = float(state[1])
  own_possession = state[4] > 0.5
  opp_possession = state[5] > 0.5
  ball_x = float(state[7])
  if remaining < 0.12:
    if score_diff < -0.2:
      return 11
    if score_diff > 0.2:
      return 9
    return 10
  if score_diff < -0.4:
    return 10
  if score_diff > 0.4 and remaining < 0.35:
    return 9
  if opp_possession:
    return 2 if ball_x > -0.25 else 3
  if own_possession:
    return 1 if ball_x > 0.0 else 4
  return 0


def choose_action(state, rng, mode):
  if mode == 'random':
    return int(rng.randint(0, 12))
  if mode == 'fixed':
    return 0
  if mode == 'rule':
    return rule_action(state)
  if mode == 'mixture':
    draw = rng.rand()
    if draw < 0.35:
      return int(rng.randint(0, 12))
    if draw < 0.55:
      return 0
    return rule_action(state)
  raise ValueError('Unknown collection policy: {}'.format(mode))


def collect(output_dir, num_states=50000, num_envs=8, seed=17,
            policy='mixture', macro_steps=100, start_method='spawn'):
  output_dir = os.path.abspath(output_dir)
  os.makedirs(output_dir, exist_ok=True)
  rng = np.random.RandomState(seed)
  env_kwargs = [{
      'seed': protocol.TRAIN_ENV_SEEDS[index],
      'split': 'train',
      'control_left': True,
      'opponent_difficulty': 0.4,
      'macro_steps': macro_steps,
      'game_duration': 3600,
      'use_potential_shaping': True,
  } for index in range(num_envs)]
  envs = vector_env.SubprocessMacroVecEnv(env_kwargs, start_method)
  states = []
  actions = []
  seeds = []
  sides = []
  difficulties = []
  rewards = []
  dones = []
  try:
    reset_configs = []
    for _ in range(num_envs):
      reset_configs.append({
          'seed': int(rng.choice(protocol.TRAIN_ENV_SEEDS)),
          'control_left': bool(rng.randint(0, 2)),
          'opponent_difficulty': float(rng.choice((0.4, 0.6, 0.8))),
      })
    observations = np.asarray(envs.reset(reset_configs), dtype=np.float32)
    while len(states) < num_states:
      action_batch = [choose_action(obs, rng, policy) for obs in observations]
      results = envs.step(action_batch)
      next_observations = np.empty_like(observations)
      for index, result in enumerate(results):
        next_obs, reward, done, info = result
        states.append(observations[index].copy())
        actions.append(action_batch[index])
        seeds.append(info['engine_seed'])
        sides.append(int(info['control_left']))
        difficulties.append(info['opponent_difficulty'])
        rewards.append(reward)
        dones.append(done)
        next_observations[index] = next_obs
        if done:
          next_observations[index] = envs.reset_at(index, {
              'seed': int(rng.choice(protocol.TRAIN_ENV_SEEDS)),
              'control_left': bool(rng.randint(0, 2)),
              'opponent_difficulty': float(rng.choice((0.4, 0.6, 0.8))),
          })
      observations = next_observations
  finally:
    envs.close()

  arrays = {
      'states': np.asarray(states[:num_states], dtype=np.float32),
      'actions': np.asarray(actions[:num_states], dtype=np.int64),
      'seeds': np.asarray(seeds[:num_states], dtype=np.int64),
      'control_left': np.asarray(sides[:num_states], dtype=np.int8),
      'opponent_difficulty': np.asarray(difficulties[:num_states],
                                        dtype=np.float32),
      'rewards': np.asarray(rewards[:num_states], dtype=np.float32),
      'dones': np.asarray(dones[:num_states], dtype=np.int8),
  }
  np.savez_compressed(os.path.join(output_dir, 'states.npz'), **arrays)
  manifest = {
      'num_states': num_states,
      'policy': policy,
      'collection_seed': seed,
      'train_seed_range': [protocol.TRAIN_ENV_SEEDS[0],
                           protocol.TRAIN_ENV_SEEDS[-1]],
      'feature_dim': int(arrays['states'].shape[1]),
      'action_dim': 12,
  }
  with open(os.path.join(output_dir, 'collection_manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
  return manifest


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output-dir', required=True)
  parser.add_argument('--num-states', type=int, default=50000)
  parser.add_argument('--num-envs', type=int, default=8)
  parser.add_argument('--seed', type=int, default=17)
  parser.add_argument('--policy', choices=('mixture', 'random', 'fixed', 'rule'),
                      default='mixture')
  parser.add_argument('--macro-steps', type=int, default=100)
  parser.add_argument('--start-method', choices=('spawn', 'fork'),
                      default='spawn')
  args = parser.parse_args()
  print(collect(**vars(args)), flush=True)


if __name__ == '__main__':
  main()
