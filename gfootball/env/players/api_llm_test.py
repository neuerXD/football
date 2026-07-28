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


"""API LLM player tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import tempfile
import time

from absl.testing import absltest
from gfootball.env import football_action_set
from gfootball.env.players import api_llm
import numpy as np


def _observation(score=(0, 0), game_mode=0, ball_owned_team=0):
  return {
      'ball': np.array([0.1, 0.0, 0.0]),
      'ball_direction': np.array([0.0, 0.0, 0.0]),
      'ball_owned_team': ball_owned_team,
      'ball_owned_player': 5,
      'score': list(score),
      'steps_left': 100,
      'game_mode': game_mode,
      'active': 5,
      'designated': 5,
      'sticky_actions': np.array([]),
      'left_team': np.zeros((11, 2)),
      'left_team_direction': np.zeros((11, 2)),
      'left_team_tired_factor': np.zeros(11),
      'left_team_yellow_card': np.zeros(11),
      'left_team_active': np.ones(11),
      'left_team_roles': np.array([0, 1, 1, 2, 3, 4, 5, 5, 6, 7, 9]),
      'right_team': np.ones((11, 2)) * 0.1,
      'right_team_direction': np.zeros((11, 2)),
      'right_team_tired_factor': np.zeros(11),
      'right_team_yellow_card': np.zeros(11),
      'right_team_active': np.ones(11),
      'right_team_roles': np.array([0, 1, 1, 2, 3, 4, 5, 5, 6, 7, 9]),
  }


class ApiLlmPlayerTest(absltest.TestCase):

  def setUp(self):
    super(ApiLlmPlayerTest, self).setUp()
    self._old_env = dict(os.environ)
    os.environ['LLM_MOCK'] = '1'

  def tearDown(self):
    os.environ.clear()
    os.environ.update(self._old_env)
    super(ApiLlmPlayerTest, self).tearDown()

  def _new_player(self, log_path=None):
    if log_path:
      os.environ['LLM_LOG_PATH'] = log_path
    return api_llm.Player({
        'left_players': '11',
        'right_players': '0',
        'team': 'left',
        'interval_steps': '1',
    }, {'action_set': 'full'})

  def _wait_until_idle(self, player):
    deadline = time.time() + 2.0
    while time.time() < deadline:
      with player._lock:
        running = player._request_running
      if not running:
        return
      time.sleep(0.01)
    self.fail('api_llm background request did not finish')

  def _set_plan_without_request(self, player, **plan):
    with player._lock:
      player._current_plan.update(plan)
      player._last_request_step = player._step

  def test_returns_builtin_ai_actions(self):
    os.environ['LLM_EXECUTE_PLAN'] = '0'
    player = self._new_player()
    actions = player.take_action([_observation()] * 11)
    self.assertLen(actions, 11)
    self.assertEqual(actions,
                     [football_action_set.action_builtin_ai] * 11)

  def test_high_pressing_overrides_designated_player(self):
    player = self._new_player()
    self._set_plan_without_request(player, pressing=0.9)

    observation = _observation(ball_owned_team=1)
    actions = player.take_action([observation])

    self.assertEqual(actions[0], football_action_set.action_team_pressure)

  def test_medium_pressing_keeps_builtin_ai_defense(self):
    player = self._new_player()
    self._set_plan_without_request(player, pressing=0.5)

    observation = _observation(ball_owned_team=1)
    actions = player.take_action([observation])

    self.assertEqual(actions[0], football_action_set.action_builtin_ai)

  def test_low_risk_on_ball_pressure_short_passes(self):
    player = self._new_player()
    self._set_plan_without_request(player, pass_risk=0.1)

    observation = _observation(ball_owned_team=0)
    observation['left_team'][5] = [0.0, 0.0]
    observation['right_team'][0] = [0.05, 0.0]
    actions = player.take_action([observation])

    self.assertEqual(actions[0], football_action_set.action_short_pass)

  def test_attack_focus_overrides_direction(self):
    player = self._new_player()
    self._set_plan_without_request(
        player, attack_focus='right', tempo=0.8, width=1.0, pass_risk=0.35)

    observation = _observation(ball_owned_team=0)
    observation['left_team'][5] = [0.0, 0.0]
    observation['right_team'] = np.ones((11, 2)) * -0.5
    actions = player.take_action([observation])

    self.assertEqual(actions[0], football_action_set.action_bottom_right)

  def test_plan_maps_to_engine_tactics(self):
    tactics = api_llm.plan_to_engine_tactics({
        'formation': '4-3-3',
        'defensive_line': 0.8,
        'pressing': 0.9,
        'width': 0.7,
        'attack_focus': 'center',
        'pass_risk': 0.6,
        'tempo': 0.4,
    })

    self.assertEqual(set(tactics), set(api_llm.ENGINE_TACTIC_KEYS))
    self.assertAlmostEqual(tactics['position_defense_midfieldfocus'], 0.8)
    self.assertAlmostEqual(
        tactics['position_defense_microfocus_strength'], 0.9)
    self.assertAlmostEqual(tactics['position_offense_width_factor'], 0.7)
    self.assertGreater(tactics['dribble_centermagnet'], 0.8)
    self.assertLess(tactics['position_offense_sidefocus_strength'], 0.3)

  def test_player_exposes_engine_tactics(self):
    player = self._new_player()
    self._set_plan_without_request(
        player, defensive_line=0.7, pressing=0.8, attack_focus='left')

    engine_tactics = player.engine_tactics()

    self.assertLen(engine_tactics, 1)
    self.assertTrue(engine_tactics[0][0])
    self.assertAlmostEqual(
        engine_tactics[0][1]['position_defense_midfieldfocus'], 0.7)

  def test_ten_zero_zero_maps_to_deep_defensive_formation(self):
    entries = api_llm.formation_to_engine_entries('10-0-0')

    self.assertLen(entries, 11)
    self.assertEqual(entries[0].role,
                     api_llm.libgame.e_PlayerRole.e_PlayerRole_GK)
    self.assertEqual(entries[1].role,
                     api_llm.libgame.e_PlayerRole.e_PlayerRole_LB)
    self.assertEqual(entries[10].role,
                     api_llm.libgame.e_PlayerRole.e_PlayerRole_RB)
    for index in range(1, 11):
      self.assertAlmostEqual(entries[index].position[0], -0.72)

  def test_player_exposes_initial_engine_formation_once(self):
    player = api_llm.Player({
        'left_players': '11',
        'right_players': '0',
        'team': 'left',
        'initial_formation': '10-0-0',
    }, {'action_set': 'full'})

    engine_formation = player.engine_formation()

    self.assertLen(engine_formation, 1)
    self.assertTrue(engine_formation[0][0])
    self.assertLen(engine_formation[0][1], 11)
    self.assertEqual(player.engine_formation(), [])

  def test_lock_formation_keeps_initial_formation_after_plan_update(self):
    player = api_llm.Player({
        'left_players': '11',
        'right_players': '0',
        'team': 'left',
        'initial_formation': '10-0-0',
        'lock_formation': '1',
        'interval_steps': '1',
    }, {'action_set': 'full'})

    player.take_action([_observation(score=(0, 1))] * 11)
    self._wait_until_idle(player)

    self.assertEqual(player.current_plan()['formation'], '10-0-0')

  def test_execute_plan_disabled_suppresses_engine_tactics(self):
    os.environ['LLM_EXECUTE_PLAN'] = '0'
    player = self._new_player()

    self.assertEqual(player.engine_tactics(), [])
    self.assertEqual(player.engine_formation(), [])

  def test_mock_plan_updates_and_logs(self):
    fd, path = tempfile.mkstemp()
    os.close(fd)
    player = self._new_player(path)
    player.take_action([_observation(score=(0, 1))] * 11)
    self._wait_until_idle(player)

    self.assertEqual(player.current_plan()['formation'], '4-3-3')
    with open(path) as f:
      lines = f.readlines()
    self.assertGreaterEqual(len(lines), 1)
    event = json.loads(lines[-1])
    self.assertEqual(event['team'], 'left')
    self.assertIsNone(event['error'])
    self.assertEqual(event['parsed_plan']['formation'], '4-3-3')

  def test_bad_json_keeps_actions_safe(self):
    player = self._new_player()
    player._call_llm = lambda prompt, state, current_plan: 'not json'
    actions = player.take_action([_observation()] * 11)
    self.assertEqual(actions,
                     [football_action_set.action_builtin_ai] * 11)
    self._wait_until_idle(player)
    self.assertEqual(player.current_plan()['formation'], '4-4-2')


if __name__ == '__main__':
  absltest.main()
