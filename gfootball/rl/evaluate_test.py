# coding=utf-8
"""Tests for macro-policy evaluation helpers."""

import json

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


def test_evaluate_freezes_provenance_at_start(tmp_path, monkeypatch):
  calls = []

  class FakeVecEnv(object):

    def __init__(self, unused_kwargs, unused_start_method):
      pass

    def reset(self, unused_configs):
      return [np.zeros(features.FEATURE_DIM, dtype=np.float32)]

    def step(self, unused_actions):
      info = {
          'score_diff': 1,
          'score': [1, 0],
          'low_level_steps': 100,
          'engine_step_latency_ms': 1.0,
      }
      observation = np.zeros(features.FEATURE_DIM, dtype=np.float32)
      return [(observation, 1.0, True, info)]

    def close(self):
      pass

  def metadata():
    calls.append(True)
    return {'git_commit': 'start-commit'}

  monkeypatch.setattr(
      evaluate.vector_env, 'SubprocessMacroVecEnv', FakeVecEnv)
  monkeypatch.setattr(evaluate.provenance, 'experiment_metadata', metadata)
  scenario = evaluate.make_scenarios((20000,), (0.6,))[0]
  result = evaluate.evaluate(
      evaluate.FixedPolicy(), 'fixed', str(tmp_path), [scenario], num_envs=1)
  with open(tmp_path / 'evaluation_manifest.json') as source:
    written = json.load(source)
  assert len(calls) == 1
  assert result['provenance'] == {'git_commit': 'start-commit'}
  assert written['provenance'] == {'git_commit': 'start-commit'}
