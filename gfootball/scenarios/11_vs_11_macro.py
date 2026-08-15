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

"""Configurable 11v11 scenario for high-level tactical learning."""

from . import *


def _add_team(builder):
  builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK)
  builder.AddPlayer(0.000000, 0.020000, e_PlayerRole_RM)
  builder.AddPlayer(0.000000, -0.020000, e_PlayerRole_CF)
  builder.AddPlayer(-0.422000, -0.195760, e_PlayerRole_LB)
  builder.AddPlayer(-0.500000, -0.063560, e_PlayerRole_CB)
  builder.AddPlayer(-0.500000, 0.063559, e_PlayerRole_CB)
  builder.AddPlayer(-0.422000, 0.195760, e_PlayerRole_RB)
  builder.AddPlayer(-0.184212, -0.105680, e_PlayerRole_CM)
  builder.AddPlayer(-0.267574, 0.000000, e_PlayerRole_CM)
  builder.AddPlayer(-0.184212, 0.105680, e_PlayerRole_CM)
  builder.AddPlayer(-0.010000, -0.216100, e_PlayerRole_LM)


def build_scenario(builder):
  control_left = bool(builder.GetConfigValue('macro_control_left', True))
  opponent_difficulty = float(
      builder.GetConfigValue('macro_opponent_difficulty', 0.6))
  controlled_difficulty = float(
      builder.GetConfigValue('macro_controlled_difficulty', 1.0))
  cfg = builder.config()
  cfg.game_duration = int(builder.GetConfigValue('macro_game_duration', 3600))
  cfg.deterministic = False
  cfg.left_team_difficulty = (
      controlled_difficulty if control_left else opponent_difficulty)
  cfg.right_team_difficulty = (
      opponent_difficulty if control_left else controlled_difficulty)

  if builder.EpisodeNumber() % 2 == 0:
    first_team, second_team = Team.e_Left, Team.e_Right
  else:
    first_team, second_team = Team.e_Right, Team.e_Left
  builder.SetTeam(first_team)
  _add_team(builder)
  builder.SetTeam(second_team)
  _add_team(builder)
