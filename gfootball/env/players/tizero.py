# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TiZero pretrained policy player.

The player wraps the TiZero JiDi submission files published at
https://huggingface.co/OpenRL/tizero.  The model is trained from the left-team
perspective.  For right-side control, leave `can_play_right` false so
FootballEnv rotates observations and actions through observation_rotation.

Example:
  --players "api_llm:left_players=11;tizero:right_players=11,model_dir=.deps/tizero"
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import sys

from absl import logging
from gfootball.env import football_action_set
from gfootball.env import player_base
import numpy as np


class Player(player_base.PlayerBase):
  """Runs the TiZero PyTorch actor for controlled players."""

  def __init__(self, player_config, env_config):
    player_base.PlayerBase.__init__(self, player_config)
    self._action_set = football_action_set.get_action_set(env_config)
    self._model_dir = os.path.abspath(player_config.get(
        'model_dir', os.environ.get('TIZERO_MODEL_DIR', '.deps/tizero')))
    self._deterministic = _as_bool(player_config.get('deterministic', '1'))
    self._torch_threads = int(player_config.get(
        'torch_threads', os.environ.get('TIZERO_TORCH_THREADS', '1')))
    self._validate_model_dir()
    self._load_modules()
    self._load_model()
    self.reset()

  def _validate_model_dir(self):
    required = [
        'actor.pt',
        'openrl_policy.py',
        'openrl_utils.py',
        'goal_keeper.py',
    ]
    missing = [
        name for name in required
        if not os.path.exists(os.path.join(self._model_dir, name))
    ]
    if missing:
      raise RuntimeError(
          'TiZero model_dir "{}" is missing: {}. Download files from '
          'https://huggingface.co/OpenRL/tizero'.format(
              self._model_dir, ', '.join(missing)))

  def _load_modules(self):
    if self._model_dir not in sys.path:
      sys.path.insert(0, self._model_dir)
    try:
      import torch
      from goal_keeper import agent_get_action
      from openrl_policy import PolicyNetwork
      from openrl_utils import openrl_obs_deal
      from openrl_utils import _t2n
    except ImportError as e:
      raise RuntimeError(
          'TiZero player requires torch plus the TiZero submission files in '
          'model_dir "{}": {}'.format(self._model_dir, e))

    self._torch = torch
    self._torch.set_num_threads(max(1, self._torch_threads))
    self._agent_get_action = agent_get_action
    self._policy_network = PolicyNetwork
    self._openrl_obs_deal = openrl_obs_deal
    self._t2n = _t2n

  def _load_model(self):
    self._model = self._policy_network()
    actor_path = os.path.join(self._model_dir, 'actor.pt')
    state_dict = self._torch.load(
        actor_path, map_location=self._torch.device('cpu'))
    self._model.load_state_dict(state_dict)
    self._model.eval()
    logging.info('Loaded TiZero actor from %s', actor_path)

  def reset(self):
    self._rnn_hidden_state = [
        np.zeros([1, 1, 512], dtype=np.float32) for _ in range(11)
    ]

  def take_action(self, observations):
    actions = [football_action_set.action_idle] * len(observations)
    actor_items = []
    for index, observation in enumerate(observations):
      active = int(observation.get('active', -1))
      if active < 0 or active >= 11:
        continue
      if active == 0:
        actions[index] = self._action_set[self._goalkeeper_action(observation)]
      else:
        actor_items.append((index, observation, active))

    for index, action in self._actor_actions(actor_items):
      actions[index] = self._action_set[action]
    return actions

  def _goalkeeper_action(self, observation):
    goalkeeper_obs = copy.deepcopy(observation)
    action = self._agent_get_action(goalkeeper_obs)[0]
    return int(action)

  def _actor_actions(self, actor_items):
    if not actor_items:
      return []

    encoded = [self._openrl_obs_deal(item[1]) for item in actor_items]
    obs = np.stack([item['obs'] for item in encoded]).astype(np.float32)
    rnn_hidden_state = np.concatenate([
        self._rnn_hidden_state[item[2]] for item in actor_items
    ], axis=0)
    available_actions = np.zeros((len(actor_items), 20), dtype=np.float32)
    for index, item in enumerate(encoded):
      available_actions[index, :19] = item['available_action']

    with self._torch.no_grad():
      actions, rnn_hidden_state = self._model(
          obs,
          rnn_hidden_state,
          available_actions=available_actions,
          deterministic=self._deterministic)

    actions = self._t2n(actions).reshape(-1)
    rnn_hidden_state = self._t2n(rnn_hidden_state)
    result = []
    for batch_index, (observation_index, observation,
                      active) in enumerate(actor_items):
      action = int(actions[batch_index])
      if action == 17 and observation['sticky_actions'][8] == 1:
        action = 15
      self._rnn_hidden_state[active] = rnn_hidden_state[batch_index:batch_index
                                                       + 1]
      result.append((observation_index, action))
    return result


def _as_bool(value):
  return str(value).lower() in ('1', 'true', 'yes', 'on')
