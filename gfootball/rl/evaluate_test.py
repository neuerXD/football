# coding=utf-8
"""Tests for macro-policy evaluation helpers."""

import numpy as np
import torch

from gfootball.rl import evaluate
from gfootball.rl import features
from gfootball.rl import models


def _contexts(count=3):
  return [{
      'seed': 20000 + index,
      'control_left': index % 2 == 0,
      'opponent_difficulty': 0.6,
      'macro_step': index,
  } for index in range(count)]


def test_make_scenarios_is_paired():
  scenarios = evaluate.make_scenarios((20000, 20001), (0.6, 0.8))
  assert len(scenarios) == 8
  assert len({item['scenario_key'] for item in scenarios}) == 8
  assert {item['control_left'] for item in scenarios} == {True, False}


def test_baseline_policies_return_valid_actions():
  observations = np.zeros((3, features.FEATURE_DIM), dtype=np.float32)
  contexts = _contexts()
  for policy in (evaluate.FixedPolicy(), evaluate.RandomPolicy(),
                 evaluate.RulePolicy()):
    actions, latencies = policy.select(observations, contexts)
    assert len(actions) == len(latencies) == 3
    assert all(0 <= action < 12 for action in actions)
    assert all(latency >= 0.0 for latency in latencies)


def test_checkpoint_policy_loads_ppo(tmp_path):
  model = models.ActorCritic(features.FEATURE_DIM, 12)
  normalizer = models.RunningMeanStd(features.FEATURE_DIM)
  checkpoint = tmp_path / 'ppo.pt'
  torch.save({
      'model_state_dict': model.state_dict(),
      'obs_normalizer': normalizer.state_dict(),
      'global_step': 123,
      'optimization_seed': 11,
  }, checkpoint)
  policy = evaluate.CheckpointPolicy(str(checkpoint), device='cpu')
  actions, _ = policy.select(
      np.zeros((2, features.FEATURE_DIM), dtype=np.float32), _contexts(2))
  assert len(actions) == 2
  assert policy.global_step == 123
  assert policy.optimization_seed == 11
