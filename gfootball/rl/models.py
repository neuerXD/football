# coding=utf-8
"""Small PyTorch models shared by behavior cloning and PPO."""

from __future__ import absolute_import

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class RunningMeanStd(object):
  """Numerically stable running observation statistics."""

  def __init__(self, shape):
    self.mean = np.zeros(shape, dtype=np.float64)
    self.var = np.ones(shape, dtype=np.float64)
    self.count = 1e-4

  def update(self, values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == self.mean.ndim:
      values = values[None, ...]
    if len(values) == 0:
      return
    batch_mean = np.mean(values, axis=0)
    batch_var = np.var(values, axis=0)
    batch_count = values.shape[0]
    delta = batch_mean - self.mean
    total = self.count + batch_count
    new_mean = self.mean + delta * batch_count / total
    m_a = self.var * self.count
    m_b = batch_var * batch_count
    correction = np.square(delta) * self.count * batch_count / total
    self.var = (m_a + m_b + correction) / total
    self.mean = new_mean
    self.count = total

  def normalize(self, values, clip=10.0):
    normalized = (np.asarray(values, dtype=np.float32) - self.mean) / np.sqrt(
        self.var + 1e-8)
    return np.clip(normalized, -clip, clip).astype(np.float32)

  def state_dict(self):
    return {
        'mean': self.mean,
        'var': self.var,
        'count': self.count,
    }

  def load_state_dict(self, state):
    self.mean = np.asarray(state['mean'], dtype=np.float64)
    self.var = np.asarray(state['var'], dtype=np.float64)
    self.count = float(state['count'])


def _mlp(input_dim, output_dim):
  return nn.Sequential(
      nn.Linear(input_dim, 128),
      nn.Tanh(),
      nn.Linear(128, 128),
      nn.Tanh(),
      nn.Linear(128, output_dim),
  )


class ActorCritic(nn.Module):
  """Separate 128x128 actor and critic MLPs."""

  def __init__(self, observation_dim, action_dim):
    super(ActorCritic, self).__init__()
    self.actor = _mlp(observation_dim, action_dim)
    self.critic = _mlp(observation_dim, 1)

  def forward(self, observations):
    return self.actor(observations), self.critic(observations).squeeze(-1)

  def value(self, observations):
    return self.critic(observations).squeeze(-1)

  def act(self, observations, deterministic=False):
    logits, values = self(observations)
    distribution = Categorical(logits=logits)
    if deterministic:
      actions = torch.argmax(logits, dim=-1)
    else:
      actions = distribution.sample()
    return actions, distribution.log_prob(actions), values

  def evaluate_actions(self, observations, actions):
    logits, values = self(observations)
    distribution = Categorical(logits=logits)
    log_prob = distribution.log_prob(actions)
    entropy = distribution.entropy()
    return log_prob, entropy, values


def set_seed(seed):
  """Sets Python-independent NumPy and Torch seeds for a learner process."""
  seed = int(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
