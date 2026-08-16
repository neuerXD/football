# coding=utf-8
"""Tests for PPO initialization behavior."""

import numpy as np
import torch

from gfootball.rl import features
from gfootball.rl import models
from gfootball.rl import ppo


def test_bc_initialization_restores_observation_normalizer(tmp_path):
  actor = models.ActorCritic(features.FEATURE_DIM, 12).actor
  mean = np.linspace(-1.0, 1.0, features.FEATURE_DIM)
  std = np.linspace(0.5, 1.5, features.FEATURE_DIM)
  checkpoint = tmp_path / 'bc.pt'
  torch.save({
      'actor_state_dict': actor.state_dict(),
      'normalizer_mean': mean,
      'normalizer_std': std,
      'normalizer_count': 50000,
  }, checkpoint)
  trainer = object.__new__(ppo.PPOTrainer)
  trainer.model = models.ActorCritic(features.FEATURE_DIM, 12)
  trainer.obs_normalizer = models.RunningMeanStd(features.FEATURE_DIM)
  trainer._load_bc_actor(str(checkpoint))
  np.testing.assert_allclose(trainer.obs_normalizer.mean, mean)
  np.testing.assert_allclose(trainer.obs_normalizer.var, np.square(std))
  assert trainer.obs_normalizer.count == 50000
