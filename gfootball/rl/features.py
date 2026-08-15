# coding=utf-8
"""Side-invariant 50-dimensional macro state encoder."""

import numpy as np

from gfootball.rl import tactics


_CONTEXT_NAMES = (
    'score_diff', 'steps_left', 'elapsed', 'is_left',
    'possession_own', 'possession_opp', 'possession_loose',
    'ball_x', 'ball_y', 'ball_z', 'ball_dx', 'ball_dy', 'ball_dz',
)
_SHAPE_NAMES = tuple(
    '{}_{}'.format(team, value)
    for team in ('own', 'opp')
    for value in ('center_x', 'center_y', 'width', 'depth', 'compactness',
                  'mean_speed'))
_ZONE_NAMES = tuple(
    '{}_zone_{}'.format(team, zone)
    for team in ('own', 'opp')
    for zone in ('defense', 'middle', 'attack'))
_ADVANTAGE_NAMES = tuple(
    'zone_advantage_{}'.format(zone)
    for zone in ('defense', 'middle', 'attack'))
_DISTANCE_NAMES = ('ball_to_opp_goal', 'ball_to_own_center',
                   'ball_to_opp_center')
_TACTIC_NAMES = tuple('tactic_{}'.format(name) for name in tactics.TACTIC_NAMES)
FEATURE_NAMES = (_CONTEXT_NAMES + _SHAPE_NAMES + _ZONE_NAMES +
                 _ADVANTAGE_NAMES + _DISTANCE_NAMES + _TACTIC_NAMES +
                 ('tactic_age',))
FEATURE_DIM = len(FEATURE_NAMES)

assert FEATURE_DIM == 50


def _active_outfield(observation, prefix, side):
  positions = np.asarray(observation[prefix + '_team'], dtype=np.float32)
  directions = np.asarray(
      observation[prefix + '_team_direction'], dtype=np.float32)
  active = np.asarray(
      observation.get(prefix + '_team_active', np.ones(len(positions))),
      dtype=bool)
  roles = np.asarray(
      observation.get(prefix + '_team_roles', np.arange(len(positions))))
  mask = active & (roles != 0)
  if not np.any(mask):
    mask = active
  if not np.any(mask):
    return np.zeros((0, 2), dtype=np.float32), np.zeros(
        (0, 2), dtype=np.float32)
  return positions[mask] * side, directions[mask] * side


def _shape(positions, directions):
  if len(positions) == 0:
    return [0.0] * 6
  center = np.mean(positions, axis=0)
  width = float(np.ptp(positions[:, 1]) / 0.84)
  depth = float(np.ptp(positions[:, 0]) / 2.0)
  compactness = float(
      np.mean(np.linalg.norm(positions - center, axis=1)) / 1.1)
  mean_speed = float(np.mean(np.linalg.norm(directions, axis=1)) / 0.05)
  return [
      float(center[0]), float(center[1]), np.clip(width, 0.0, 1.0),
      np.clip(depth, 0.0, 1.0), np.clip(compactness, 0.0, 1.0),
      np.clip(mean_speed, 0.0, 1.0),
  ]


def _zone_counts(positions):
  if len(positions) == 0:
    return [0.0, 0.0, 0.0]
  values = [
      np.sum(positions[:, 0] < -0.33),
      np.sum((positions[:, 0] >= -0.33) & (positions[:, 0] <= 0.33)),
      np.sum(positions[:, 0] > 0.33),
  ]
  return [float(value) / 10.0 for value in values]


def orient_observation(observation, control_left):
  """Returns the raw components in the controlled team's attacking frame."""
  side = 1.0 if control_left else -1.0
  own_prefix = 'left' if control_left else 'right'
  opp_prefix = 'right' if control_left else 'left'
  ball = np.asarray(observation['ball'], dtype=np.float32).copy()
  ball_direction = np.asarray(
      observation['ball_direction'], dtype=np.float32).copy()
  ball[:2] *= side
  ball_direction[:2] *= side
  own_positions, own_directions = _active_outfield(
      observation, own_prefix, side)
  opp_positions, opp_directions = _active_outfield(
      observation, opp_prefix, side)
  own_team_id = 0 if control_left else 1
  score = observation['score']
  score_diff = score[0] - score[1] if control_left else score[1] - score[0]
  return {
      'ball': ball,
      'ball_direction': ball_direction,
      'own_positions': own_positions,
      'own_directions': own_directions,
      'opp_positions': opp_positions,
      'opp_directions': opp_directions,
      'possession': int(observation.get('ball_owned_team', -1)),
      'own_team_id': own_team_id,
      'score_diff': float(score_diff),
  }


class MacroFeatureEncoder(object):

  def __init__(self, game_duration=3600):
    self.game_duration = float(game_duration)

  def encode(self, observation, control_left, action_id, tactic_age_steps):
    oriented = orient_observation(observation, control_left)
    ball = oriented['ball']
    direction = oriented['ball_direction']
    possession = oriented['possession']
    own_id = oriented['own_team_id']
    steps_left = float(observation['steps_left'])

    own_shape = _shape(oriented['own_positions'],
                       oriented['own_directions'])
    opp_shape = _shape(oriented['opp_positions'],
                       oriented['opp_directions'])
    own_zones = _zone_counts(oriented['own_positions'])
    opp_zones = _zone_counts(oriented['opp_positions'])
    advantage = [own - opp for own, opp in zip(own_zones, opp_zones)]
    own_center = np.asarray(own_shape[:2], dtype=np.float32)
    opp_center = np.asarray(opp_shape[:2], dtype=np.float32)

    values = [
        np.clip(oriented['score_diff'] / 5.0, -1.0, 1.0),
        np.clip(steps_left / self.game_duration, 0.0, 1.0),
        np.clip(1.0 - steps_left / self.game_duration, 0.0, 1.0),
        1.0 if control_left else 0.0,
        1.0 if possession == own_id else 0.0,
        1.0 if possession not in (-1, own_id) else 0.0,
        1.0 if possession == -1 else 0.0,
        float(ball[0]), float(ball[1]), float(ball[2]),
        float(np.clip(direction[0] / 0.05, -1.0, 1.0)),
        float(np.clip(direction[1] / 0.05, -1.0, 1.0)),
        float(np.clip(direction[2] / 0.05, -1.0, 1.0)),
    ]
    values.extend(own_shape)
    values.extend(opp_shape)
    values.extend(own_zones)
    values.extend(opp_zones)
    values.extend(advantage)
    values.extend([
        float(np.clip(np.linalg.norm(ball[:2] - np.array([1.0, 0.0])) /
                      2.1, 0.0, 1.0)),
        float(np.clip(np.linalg.norm(ball[:2] - own_center) / 2.1, 0.0, 1.0)),
        float(np.clip(np.linalg.norm(ball[:2] - opp_center) / 2.1, 0.0, 1.0)),
    ])
    one_hot = [0.0] * tactics.NUM_TACTICS
    one_hot[int(action_id)] = 1.0
    values.extend(one_hot)
    values.append(float(np.clip(tactic_age_steps / 1000.0, 0.0, 1.0)))
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (FEATURE_DIM,):
      raise AssertionError('Unexpected macro feature shape: {}'.format(
          result.shape))
    return result


def potential(observation, control_left):
  """Bounded potential combining ball progress, possession, and team advance."""
  oriented = orient_observation(observation, control_left)
  own_positions = oriented['own_positions']
  own_center_x = float(np.mean(own_positions[:, 0])) if len(
      own_positions) else 0.0
  if oriented['possession'] == oriented['own_team_id']:
    possession_value = 1.0
  elif oriented['possession'] == -1:
    possession_value = 0.0
  else:
    possession_value = -1.0
  value = (0.5 * float(oriented['ball'][0]) +
           0.3 * possession_value + 0.2 * own_center_x)
  return float(np.clip(value, -1.0, 1.0))
