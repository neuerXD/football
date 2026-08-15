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


"""API LLM coach player.

This player is a coaching bridge: it asks an OpenAI-compatible chat completion
API for low-frequency tactical plans, pushes those plans into the built-in game
AI's runtime tactics, and optionally overrides a few designated-player actions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import os
import threading
import time
try:
  from urllib import request as urllib_request
  from urllib import error as urllib_error
except ImportError:
  import urllib2 as urllib_request
  import urllib2 as urllib_error

import gfootball_engine as libgame
from gfootball.env import football_action_set
from gfootball.env import player_base
from gfootball.env import tactical_plan


_LOG_LOCK = threading.Lock()

_DEFAULT_PLAN = tactical_plan.DEFAULT_PLAN
_ATTACK_FOCUS = tactical_plan.ATTACK_FOCUS
_ROLE_NAMES = {
    0: 'GK',
    1: 'CB',
    2: 'LB',
    3: 'RB',
    4: 'DM',
    5: 'CM',
    6: 'LM',
    7: 'RM',
    8: 'AM',
    9: 'CF',
}
ENGINE_TACTIC_KEYS = tactical_plan.ENGINE_TACTIC_KEYS
_INITIAL_FORMATION_ENV = {
    'left': 'LLM_INITIAL_FORMATION_LEFT',
    'right': 'LLM_INITIAL_FORMATION_RIGHT',
}
_LOCK_FORMATION_ENV = {
    'left': 'LLM_LOCK_FORMATION_LEFT',
    'right': 'LLM_LOCK_FORMATION_RIGHT',
}


def _truthy(value):
  return str(value).lower() in ['1', 'true', 'yes', 'y', 'on']


def _float(value, default):
  try:
    return float(value)
  except (TypeError, ValueError):
    return default


def _int(value, default):
  try:
    return int(value)
  except (TypeError, ValueError):
    return default


def _clamp(value, low=0.0, high=1.0):
  return tactical_plan.clamp(value, low, high)


def _jsonable(value):
  if hasattr(value, 'tolist'):
    return value.tolist()
  return value


def _short_points(points, limit=11):
  result = []
  for idx, point in enumerate(points[:limit]):
    result.append([
        idx,
        round(float(point[0]), 3),
        round(float(point[1]), 3),
    ])
  return result


def _roles(roles, limit=11):
  result = []
  for idx, role in enumerate(roles[:limit]):
    result.append([idx, _ROLE_NAMES.get(int(role), str(int(role)))])
  return result


def _extract_json_object(text):
  try:
    return json.loads(text)
  except ValueError:
    pass

  start = text.find('{')
  end = text.rfind('}')
  if start == -1 or end == -1 or end <= start:
    raise ValueError('No JSON object found in response')
  return json.loads(text[start:end + 1])


def _coerce_plan(candidate, fallback):
  return tactical_plan.coerce_plan(candidate, fallback)


def plan_to_engine_tactics(plan):
  """Translates the public coach plan into gfootball engine tactic modifiers."""
  return tactical_plan.plan_to_engine_tactics(plan)


def _formation_counts(formation):
  return tactical_plan.formation_counts(formation)


def _formation_name(formation):
  return tactical_plan.formation_name(formation)


def _line_x_positions(line_count):
  if line_count <= 1:
    return [-0.72]
  if line_count == 2:
    return [-0.62, -0.12]
  start = -0.56 if line_count == 3 else -0.62
  end = 0.16 if line_count == 3 else 0.22
  return [
      start + (end - start) * idx / float(line_count - 1)
      for idx in range(line_count)
  ]


def _line_y_positions(count):
  if count <= 1:
    return [0.0]
  spread = min(0.36, 0.06 + 0.04 * count)
  return [
      -spread + 2.0 * spread * idx / float(count - 1)
      for idx in range(count)
  ]


def _formation_role(line_index, line_count, slot_index, slot_count):
  role = libgame.e_PlayerRole
  if line_index == 0:
    if slot_count >= 3 and slot_index == 0:
      return role.e_PlayerRole_LB
    if slot_count >= 3 and slot_index == slot_count - 1:
      return role.e_PlayerRole_RB
    return role.e_PlayerRole_CB
  if line_index == line_count - 1:
    if slot_count >= 3 and slot_index == 0:
      return role.e_PlayerRole_LM
    if slot_count >= 3 and slot_index == slot_count - 1:
      return role.e_PlayerRole_RM
    if slot_count >= 3 and slot_index == slot_count // 2:
      return role.e_PlayerRole_CF
    return role.e_PlayerRole_CF
  if slot_count >= 3 and slot_index == 0:
    return role.e_PlayerRole_LM
  if slot_count >= 3 and slot_index == slot_count - 1:
    return role.e_PlayerRole_RM
  return role.e_PlayerRole_CM


def formation_to_engine_entries(formation):
  """Builds a runtime FormationEntryVec from an outfield formation string."""
  return tactical_plan.formation_to_engine_entries(formation)


class Player(player_base.PlayerBase):
  """Low-frequency LLM coach that delegates player actions to built-in AI."""

  def __init__(self, player_config, env_config):
    player_base.PlayerBase.__init__(self, player_config)
    assert (self.num_controlled_left_players() == 0 or
            self.num_controlled_right_players() == 0), (
                'api_llm controls one team per player definition')
    assert env_config['action_set'] in ['v2', 'full'], (
        'api_llm requires action_set=v2 or action_set=full')

    self._action_set = football_action_set.get_action_set(env_config)
    self._can_play_right = True
    self._team = player_config.get(
        'team', 'left' if self.num_controlled_left_players() else 'right')
    self._team = self._team.lower()
    assert self._team in ['left', 'right']
    if self.num_controlled_left_players():
      assert self._team == 'left', 'left_players requires team=left'
    if self.num_controlled_right_players():
      assert self._team == 'right', 'right_players requires team=right'

    side_suffix = self._team.upper()
    self._base_url = os.environ.get('LLM_BASE_URL', '').rstrip('/')
    self._api_key = os.environ.get('LLM_API_KEY', '')
    self._model = player_config.get(
        'model', os.environ.get('LLM_MODEL_{}'.format(side_suffix),
                               os.environ.get('LLM_MODEL', '')))
    self._temperature = _float(
        player_config.get('temperature', os.environ.get('LLM_TEMPERATURE')),
        0.2)
    self._max_tokens = _int(
        player_config.get('max_tokens', os.environ.get('LLM_MAX_TOKENS')), 500)
    self._timeout_sec = _float(
        player_config.get('timeout_sec', os.environ.get('LLM_TIMEOUT_SEC')), 8.0)
    self._interval_steps = _int(
        player_config.get('interval_steps',
                          os.environ.get('LLM_INTERVAL_STEPS')), 100)
    self._dead_ball_min_steps = _int(
        player_config.get('dead_ball_min_steps',
                          os.environ.get('LLM_DEAD_BALL_MIN_STEPS')), 50)
    self._log_path = player_config.get(
        'log_path',
        os.environ.get('LLM_LOG_PATH', '/tmp/gfootball_llm_coaches.jsonl'))
    self._mock = _truthy(player_config.get('mock', os.environ.get('LLM_MOCK')))
    self._execute_plan = _truthy(
        player_config.get('execute_plan',
                          os.environ.get('LLM_EXECUTE_PLAN', '1')))
    self._initial_formation = player_config.get(
        'initial_formation',
        os.environ.get(_INITIAL_FORMATION_ENV[self._team],
                       os.environ.get('LLM_INITIAL_FORMATION', '')))
    self._lock_formation = _truthy(
        player_config.get(
            'lock_formation',
            os.environ.get(_LOCK_FORMATION_ENV[self._team],
                           os.environ.get('LLM_LOCK_FORMATION', '0'))))
    self._initial_plan = copy.deepcopy(_DEFAULT_PLAN)
    if self._initial_formation:
      self._initial_plan['formation'] = self._initial_formation

    self._lock = threading.Lock()
    self._current_plan = copy.deepcopy(self._initial_plan)
    self._last_engine_formation = None
    self._request_running = False
    self._last_request_step = None
    self._step = 0

  def reset(self):
    self._step = 0
    self._last_request_step = None
    self._last_engine_formation = None
    with self._lock:
      self._current_plan = copy.deepcopy(self._initial_plan)

  def take_action(self, observations):
    if observations:
      self._maybe_request_plan(observations)
    with self._lock:
      current_plan = copy.deepcopy(self._current_plan)
    self._step += 1
    if not self._execute_plan:
      return [football_action_set.action_builtin_ai] * len(observations)
    return [
        self._action_from_plan(observation, current_plan)
        for observation in observations
    ]

  def current_plan(self):
    with self._lock:
      return copy.deepcopy(self._current_plan)

  def engine_tactics(self):
    if not self._execute_plan:
      return []
    with self._lock:
      current_plan = copy.deepcopy(self._current_plan)
    return [(self._team == 'left', plan_to_engine_tactics(current_plan))]

  def engine_formation(self):
    if not self._execute_plan:
      return []
    with self._lock:
      formation = _formation_name(self._current_plan.get('formation', ''))
      if not formation or formation == self._last_engine_formation:
        return []
      self._last_engine_formation = formation
    entries = formation_to_engine_entries(formation)
    if entries is None:
      return []
    return [(self._team == 'left', entries)]

  def _maybe_request_plan(self, observations):
    observation = observations[0]
    if not self._should_request(observation):
      return
    state = self._build_state_summary(observation)
    with self._lock:
      if self._request_running:
        return
      self._request_running = True
      self._last_request_step = self._step
      current_plan = copy.deepcopy(self._current_plan)

    prompt = self._build_prompt(state, current_plan)
    thread = threading.Thread(
        target=self._request_plan_thread,
        args=(state, current_plan, prompt))
    thread.daemon = True
    thread.start()

  def _should_request(self, observation):
    if self._last_request_step is None:
      return True
    since_last = self._step - self._last_request_step
    if since_last >= self._interval_steps:
      return True
    return observation.get('game_mode', 0) != 0 and since_last >= (
        self._dead_ball_min_steps)

  def _request_plan_thread(self, state, current_plan, prompt):
    raw_response = ''
    parsed_plan = None
    error = None
    started_at = time.time()
    try:
      raw_response = self._call_llm(prompt, state, current_plan)
      parsed = _extract_json_object(raw_response)
      parsed_plan = _coerce_plan(parsed, current_plan)
      if self._lock_formation:
        parsed_plan['formation'] = current_plan['formation']
      with self._lock:
        self._current_plan = copy.deepcopy(parsed_plan)
    except Exception as exc:  # pylint: disable=broad-except
      error = str(exc)
    finally:
      event = {
          'timestamp': started_at,
          'duration_sec': round(time.time() - started_at, 3),
          'team': self._team,
          'model': self._model,
          'step': self._step,
          'state': state,
          'raw_response': raw_response,
          'parsed_plan': parsed_plan,
          'error': error,
      }
      try:
        self._write_log(event)
      finally:
        with self._lock:
          self._request_running = False

  def _call_llm(self, prompt, state, current_plan):
    if self._mock:
      return json.dumps(self._mock_plan(state, current_plan))
    if not self._base_url:
      raise ValueError('LLM_BASE_URL is required unless LLM_MOCK=1')
    if not self._model:
      raise ValueError(
          'LLM_MODEL_{} or LLM_MODEL is required'.format(self._team.upper()))

    payload = {
        'model': self._model,
        'messages': [
            {
                'role': 'system',
                'content': self._system_prompt()
            },
            {
                'role': 'user',
                'content': prompt
            },
        ],
        'temperature': self._temperature,
        'max_tokens': self._max_tokens,
    }
    data = json.dumps(payload).encode('utf-8')
    headers = {'Content-Type': 'application/json'}
    if self._api_key:
      headers['Authorization'] = 'Bearer {}'.format(self._api_key)

    req = urllib_request.Request(
        self._base_url + '/chat/completions', data=data, headers=headers)
    try:
      response = urllib_request.urlopen(req, timeout=self._timeout_sec)
      response_body = response.read().decode('utf-8')
    except urllib_error.HTTPError as exc:
      body = exc.read().decode('utf-8', errors='replace')
      raise RuntimeError('LLM HTTP {}: {}'.format(exc.code, body[:500]))

    decoded = json.loads(response_body)
    return decoded['choices'][0]['message']['content']

  def _mock_plan(self, state, current_plan):
    plan = copy.deepcopy(current_plan)
    own_score = state['score']['own']
    opponent_score = state['score']['opponent']
    ball_x = state['ball']['position'][0]
    if own_score < opponent_score:
      plan.update({
          'formation': '4-3-3',
          'defensive_line': 0.72,
          'pressing': 0.82,
          'pass_risk': 0.62,
          'tempo': 0.75,
          'notes': 'Mock coach: chase the game with higher pressure.'
      })
    elif ball_x > 0.25:
      plan.update({
          'formation': '4-2-3-1',
          'attack_focus': 'center',
          'tempo': 0.65,
          'notes': 'Mock coach: consolidate possession in advanced areas.'
      })
    else:
      plan.update({
          'formation': '4-4-2',
          'attack_focus': 'balanced',
          'notes': 'Mock coach: balanced default plan.'
      })
    return plan

  def _action_from_plan(self, observation, plan):
    if int(observation.get('active', -1)) < 0:
      return football_action_set.action_builtin_ai
    if int(observation.get('game_mode', 0)) != 0:
      return football_action_set.action_builtin_ai

    active = int(observation.get('active', -1))
    designated = int(observation.get('designated', -1))
    is_designated = active == designated
    ball_owned_team = int(observation.get('ball_owned_team', -1))

    if ball_owned_team == self._team_id():
      return self._attacking_action(observation, plan, is_designated)
    if ball_owned_team == self._opponent_team_id():
      return self._defending_action(plan, is_designated)
    return self._loose_ball_action(plan, is_designated)

  def _attacking_action(self, observation, plan, is_designated):
    if not is_designated:
      return football_action_set.action_builtin_ai

    active_position = observation['{}_team'.format(self._team)][
        int(observation['active'])]
    distance_to_goal = (
        (float(active_position[0]) - self._goal_x()) ** 2 +
        float(active_position[1]) ** 2) ** 0.5
    if (distance_to_goal < 0.22 and plan['pass_risk'] >= 0.45 and
        self._has_action(football_action_set.action_shot)):
      return football_action_set.action_shot

    closest_opp = self._closest_opponent_distance(observation, active_position)
    if (closest_opp < 0.13 and plan['pass_risk'] <= 0.3 and
        self._has_action(football_action_set.action_short_pass)):
      return football_action_set.action_short_pass

    signed_x = float(active_position[0]) * self._attack_sign()
    if (signed_x > -0.15 and plan['pass_risk'] >= 0.75 and
        self._has_action(football_action_set.action_high_pass)):
      return football_action_set.action_high_pass

    if (plan['tempo'] >= 0.85 and
        self._has_action(football_action_set.action_sprint)):
      return football_action_set.action_sprint

    focus_action = self._attack_focus_action(observation, plan,
                                             active_position)
    if focus_action is not None:
      return focus_action

    return football_action_set.action_builtin_ai

  def _defending_action(self, plan, is_designated):
    if not is_designated:
      return football_action_set.action_builtin_ai
    if (plan['pressing'] >= 0.78 and
        self._has_action(football_action_set.action_team_pressure)):
      return football_action_set.action_team_pressure
    if (plan['pressing'] >= 0.55 and
        self._has_action(football_action_set.action_pressure)):
      return football_action_set.action_pressure
    return football_action_set.action_builtin_ai

  def _loose_ball_action(self, plan, is_designated):
    if not is_designated:
      return football_action_set.action_builtin_ai
    urgency = (plan['tempo'] + plan['pressing']) * 0.5
    if urgency >= 0.65 and self._has_action(football_action_set.action_sprint):
      return football_action_set.action_sprint
    return football_action_set.action_builtin_ai

  def _attack_focus_action(self, observation, plan, active_position):
    attack_focus = plan.get('attack_focus', 'balanced')
    if attack_focus == 'balanced' and plan['tempo'] < 0.7:
      return None

    lane_width = 0.08 + 0.24 * plan['width']
    if attack_focus == 'left':
      target_y = -lane_width * self._attack_sign()
    elif attack_focus == 'right':
      target_y = lane_width * self._attack_sign()
    else:
      target_y = 0.0

    advance = 0.18 + 0.32 * plan['tempo']
    raw_target_x = float(active_position[0]) + self._attack_sign() * advance
    if self._attack_sign() > 0:
      target_x = min(raw_target_x, self._goal_x() * 0.86)
    else:
      target_x = max(raw_target_x, self._goal_x() * 0.86)
    target = [target_x, target_y]
    delta = [
        target[0] - float(active_position[0]),
        target[1] - float(active_position[1]),
    ]
    if abs(delta[0]) + abs(delta[1]) < 0.02:
      return None
    action = self._direction_action(delta)
    return action if self._has_action(action) else None

  def _direction_action(self, delta):
    directions = [
        (football_action_set.action_top, [0, -1]),
        (football_action_set.action_top_left, [-1, -1]),
        (football_action_set.action_left, [-1, 0]),
        (football_action_set.action_bottom_left, [-1, 1]),
        (football_action_set.action_bottom, [0, 1]),
        (football_action_set.action_bottom_right, [1, 1]),
        (football_action_set.action_right, [1, 0]),
        (football_action_set.action_top_right, [1, -1]),
    ]
    best_action = football_action_set.action_idle
    best_score = None
    for action, direction in directions:
      length = (direction[0] ** 2 + direction[1] ** 2) ** 0.5
      score = (delta[0] * direction[0] + delta[1] * direction[1]) / length
      if best_score is None or score > best_score:
        best_score = score
        best_action = action
    return best_action

  def _closest_opponent_distance(self, observation, active_position):
    opponent_prefix = 'right' if self._team == 'left' else 'left'
    closest = None
    for opponent in observation['{}_team'.format(opponent_prefix)]:
      distance = ((float(opponent[0]) - float(active_position[0])) ** 2 +
                  (float(opponent[1]) - float(active_position[1])) ** 2) ** 0.5
      if closest is None or distance < closest:
        closest = distance
    return closest if closest is not None else 2.0

  def _has_action(self, action):
    return action in self._action_set

  def _team_id(self):
    return 0 if self._team == 'left' else 1

  def _opponent_team_id(self):
    return 1 - self._team_id()

  def _goal_x(self):
    return 1.0 if self._team == 'left' else -1.0

  def _attack_sign(self):
    return 1.0 if self._team == 'left' else -1.0

  def _build_state_summary(self, observation):
    own_prefix = self._team
    opponent_prefix = 'right' if self._team == 'left' else 'left'
    own_team_id = 0 if self._team == 'left' else 1
    score = observation['score']
    own_score = score[0] if self._team == 'left' else score[1]
    opponent_score = score[1] if self._team == 'left' else score[0]
    ball_owned_team = int(observation['ball_owned_team'])
    possession = 'none'
    if ball_owned_team == own_team_id:
      possession = 'own'
    elif ball_owned_team != -1:
      possession = 'opponent'

    return {
        'team': self._team,
        'step': self._step,
        'score': {
            'own': int(own_score),
            'opponent': int(opponent_score),
            'left': int(score[0]),
            'right': int(score[1]),
        },
        'steps_left': int(observation['steps_left']),
        'game_mode': int(observation['game_mode']),
        'ball': {
            'position': [
                round(float(observation['ball'][0]), 3),
                round(float(observation['ball'][1]), 3),
                round(float(observation['ball'][2]), 3),
            ],
            'direction': [
                round(float(observation['ball_direction'][0]), 3),
                round(float(observation['ball_direction'][1]), 3),
                round(float(observation['ball_direction'][2]), 3),
            ],
            'possession': possession,
            'owned_player': int(observation['ball_owned_player']),
        },
        'own_team': {
            'roles': _roles(observation['{}_team_roles'.format(own_prefix)]),
            'positions': _short_points(
                observation['{}_team'.format(own_prefix)]),
            'active_player': int(observation['active']),
            'designated_player': int(observation['designated']),
        },
        'opponent_team': {
            'roles': _roles(
                observation['{}_team_roles'.format(opponent_prefix)]),
            'positions': _short_points(
                observation['{}_team'.format(opponent_prefix)]),
        },
    }

  def _build_prompt(self, state, current_plan):
    prompt = {
        'task': 'Choose the next tactical plan for your team.',
        'team': self._team,
        'state': state,
        'current_plan': current_plan,
        'required_json_schema': {
            'formation': 'string, for example 4-4-2 or 4-2-3-1',
            'defensive_line': 'float 0..1',
            'pressing': 'float 0..1',
            'width': 'float 0..1',
            'attack_focus': 'left|center|right|balanced',
            'pass_risk': 'float 0..1',
            'tempo': 'float 0..1',
            'notes': 'short string',
        },
    }
    return json.dumps(prompt, sort_keys=True)

  def _system_prompt(self):
    return (
        'You are a football coach controlling high-level tactics only. '
        'Return exactly one JSON object and no prose. All numeric values must '
        'be between 0 and 1. Use only attack_focus values: left, center, right, '
        'balanced.')

  def _write_log(self, event):
    if not self._log_path:
      return
    directory = os.path.dirname(self._log_path)
    if directory and not os.path.exists(directory):
      os.makedirs(directory)
    safe_event = json.loads(json.dumps(event, default=_jsonable))
    with _LOG_LOCK:
      with open(self._log_path, 'a') as f:
        f.write(json.dumps(safe_event, sort_keys=True) + '\n')
