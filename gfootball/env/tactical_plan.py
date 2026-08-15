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

"""Pure tactical-plan validation, translation, and synchronous execution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import gfootball_engine as libgame


DEFAULT_PLAN = {
    'formation': '4-4-2',
    'defensive_line': 0.5,
    'pressing': 0.5,
    'width': 0.5,
    'attack_focus': 'balanced',
    'pass_risk': 0.35,
    'tempo': 0.5,
    'notes': 'Initial balanced plan.',
}

ATTACK_FOCUS = frozenset(['left', 'center', 'right', 'balanced'])

ENGINE_TACTIC_KEYS = (
    'position_defense_depth_factor',
    'position_defense_microfocus_strength',
    'position_defense_midfieldfocus',
    'position_defense_midfieldfocus_strength',
    'position_defense_sidefocus_strength',
    'position_defense_width_factor',
    'position_offense_depth_factor',
    'position_offense_microfocus_strength',
    'position_offense_midfieldfocus',
    'position_offense_midfieldfocus_strength',
    'position_offense_sidefocus_strength',
    'position_offense_width_factor',
    'dribble_centermagnet',
    'dribble_offensiveness',
)


def clamp(value, low=0.0, high=1.0):
  return max(low, min(high, value))


def _as_float(value, default):
  try:
    return float(value)
  except (TypeError, ValueError):
    return default


def coerce_plan(candidate, fallback=None):
  """Validates a partial plan and fills missing values from a fallback."""
  if not isinstance(candidate, dict):
    raise ValueError('Tactical plan is not a dictionary')

  plan = copy.deepcopy(fallback or DEFAULT_PLAN)
  plan['formation'] = str(candidate.get('formation', plan['formation']))[:32]
  for key in ('defensive_line', 'pressing', 'width', 'pass_risk', 'tempo'):
    plan[key] = clamp(_as_float(candidate.get(key, plan[key]), plan[key]))

  attack_focus = str(candidate.get(
      'attack_focus', plan['attack_focus'])).lower()
  if attack_focus in ATTACK_FOCUS:
    plan['attack_focus'] = attack_focus
  plan['notes'] = str(candidate.get('notes', plan.get('notes', '')))[:500]
  return plan


def plan_to_engine_tactics(plan):
  """Translates a public tactical plan into engine tactic modifiers."""
  plan = coerce_plan(plan if isinstance(plan, dict) else {}, DEFAULT_PLAN)
  directness = clamp(plan['tempo'] * 0.55 + plan['pass_risk'] * 0.45)
  support_focus = clamp((1.0 - plan['pass_risk']) * 0.65 +
                        plan['tempo'] * 0.35)
  defense_depth = clamp(1.0 - plan['pressing'] * 0.35)

  if plan['attack_focus'] in ('left', 'right'):
    side_focus = 0.85
    center_magnet = 0.25
  elif plan['attack_focus'] == 'center':
    side_focus = 0.2
    center_magnet = 0.9
  else:
    side_focus = 0.45
    center_magnet = 0.65

  return {
      'position_defense_depth_factor': defense_depth,
      'position_defense_microfocus_strength': plan['pressing'],
      'position_defense_midfieldfocus': plan['defensive_line'],
      'position_defense_midfieldfocus_strength': plan['pressing'],
      'position_defense_sidefocus_strength': clamp(
          0.25 + 0.55 * plan['pressing']),
      'position_defense_width_factor': plan['width'],
      'position_offense_depth_factor': directness,
      'position_offense_microfocus_strength': support_focus,
      'position_offense_midfieldfocus': plan['tempo'],
      'position_offense_midfieldfocus_strength': directness,
      'position_offense_sidefocus_strength': side_focus,
      'position_offense_width_factor': plan['width'],
      'dribble_centermagnet': center_magnet,
      'dribble_offensiveness': directness,
  }


def formation_counts(formation):
  text = str(formation).strip().lower().replace(' ', '')
  if not text:
    return None
  pieces = text.split('-')
  counts = []
  for piece in pieces:
    if not piece.isdigit():
      return None
    counts.append(int(piece))
  if not counts or sum(counts) != 10:
    return None
  return counts


def formation_name(formation):
  counts = formation_counts(formation)
  if counts is None:
    return None
  return '-'.join(str(count) for count in counts)


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
    return role.e_PlayerRole_CF
  if slot_count >= 3 and slot_index == 0:
    return role.e_PlayerRole_LM
  if slot_count >= 3 and slot_index == slot_count - 1:
    return role.e_PlayerRole_RM
  return role.e_PlayerRole_CM


def formation_to_engine_entries(formation, attack_focus='balanced'):
  """Builds engine formation entries, including an optional flank bias."""
  counts = formation_counts(formation)
  if counts is None:
    return None
  populated_counts = [count for count in counts if count > 0]
  line_count = len(populated_counts)
  line_x = _line_x_positions(line_count)
  if attack_focus == 'left':
    lateral_bias = 0.08
  elif attack_focus == 'right':
    lateral_bias = -0.08
  else:
    lateral_bias = 0.0

  entries = libgame.FormationEntryVec()
  entries.append(
      libgame.FormationEntry(-1.0, 0.0,
                             libgame.e_PlayerRole.e_PlayerRole_GK, False,
                             True))
  for line_index, count in enumerate(populated_counts):
    for slot_index, y in enumerate(_line_y_positions(count)):
      if line_index > 0:
        y = max(-0.4, min(0.4, y + lateral_bias))
      entries.append(
          libgame.FormationEntry(
              line_x[line_index], y,
              _formation_role(line_index, line_count, slot_index, count),
              False, True))
  return entries


class TacticalPlanExecutor(object):
  """Applies plans synchronously to a FootballEnvCore instance."""

  def __init__(self, env_core):
    self._env_core = env_core
    self._last_formation = {}

  def reset(self):
    self._last_formation = {}

  def set_team_plan(self, left_team, plan):
    normalized = coerce_plan(plan, DEFAULT_PLAN)
    team = bool(left_team)
    formation_key = (normalized['formation'], normalized['attack_focus'])
    if self._last_formation.get(team) != formation_key:
      entries = formation_to_engine_entries(*formation_key)
      if entries is None:
        raise ValueError('Invalid formation: {}'.format(
            normalized['formation']))
      self._env_core.set_formation(team, entries)
      self._last_formation[team] = formation_key
    self._env_core.set_tactics(team, plan_to_engine_tactics(normalized))
    return normalized
