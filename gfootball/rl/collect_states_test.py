# coding=utf-8
"""Tests for reproducible macro-state collection."""

import json

import numpy as np

from gfootball.rl import collect_states
from gfootball.rl import features


def test_collect_records_start_provenance_without_env_argument(tmp_path,
                                                               monkeypatch):
  captured_kwargs = []

  class FakeVecEnv(object):

    def __init__(self, env_kwargs, unused_start_method):
      captured_kwargs.extend(env_kwargs)

    def reset(self, unused_configs):
      return [np.zeros(features.FEATURE_DIM, dtype=np.float32)]

    def step(self, unused_actions):
      info = {
          'engine_seed': 1000,
          'control_left': True,
          'opponent_difficulty': 0.4,
      }
      observation = np.zeros(features.FEATURE_DIM, dtype=np.float32)
      return [(observation, 0.0, False, info)]

    def close(self):
      pass

  monkeypatch.setattr(
      collect_states.vector_env, 'SubprocessMacroVecEnv', FakeVecEnv)
  monkeypatch.setattr(
      collect_states.provenance, 'experiment_metadata',
      lambda: {'git_commit': 'start-commit'})
  manifest = collect_states.collect(
      str(tmp_path), num_states=1, num_envs=1, progress_interval=0)
  with open(tmp_path / 'collection_manifest.json') as source:
    written = json.load(source)
  assert captured_kwargs
  assert 'provenance' not in captured_kwargs[0]
  assert manifest['provenance'] == {'git_commit': 'start-commit'}
  assert written['provenance'] == {'git_commit': 'start-commit'}
