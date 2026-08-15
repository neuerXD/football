# coding=utf-8
"""Tests for macro tactical state features."""

from absl.testing import absltest
from gfootball.rl import features
import numpy as np


def _observation():
  left = np.array([
      [-1.0, 0.0], [-0.7, -0.3], [-0.65, 0.3], [-0.2, -0.25],
      [-0.2, 0.25], [0.0, 0.0], [0.15, -0.2], [0.15, 0.2],
      [0.45, -0.25], [0.55, 0.0], [0.45, 0.25],
  ], dtype=np.float32)
  right = -left
  roles = np.array([0, 1, 1, 2, 3, 4, 5, 5, 6, 7, 9])
  return {
      'ball': np.array([0.2, -0.1, 0.0], dtype=np.float32),
      'ball_direction': np.array([0.01, 0.002, 0.0], dtype=np.float32),
      'ball_owned_team': 0,
      'score': [2, 1],
      'steps_left': 1800,
      'left_team': left,
      'left_team_direction': np.ones((11, 2), dtype=np.float32) * 0.002,
      'left_team_active': np.ones(11),
      'left_team_roles': roles,
      'right_team': right,
      'right_team_direction': np.ones((11, 2), dtype=np.float32) * -0.002,
      'right_team_active': np.ones(11),
      'right_team_roles': roles,
  }


def _mirror(observation):
  mirrored = dict(observation)
  mirrored['ball'] = observation['ball'] * np.array([-1.0, -1.0, 1.0])
  mirrored['ball_direction'] = observation['ball_direction'] * np.array(
      [-1.0, -1.0, 1.0])
  mirrored['ball_owned_team'] = 1 - observation['ball_owned_team']
  mirrored['score'] = list(reversed(observation['score']))
  for target, source in (('left', 'right'), ('right', 'left')):
    mirrored[target + '_team'] = -observation[source + '_team']
    mirrored[target + '_team_direction'] = -observation[
        source + '_team_direction']
    mirrored[target + '_team_active'] = observation[source + '_team_active']
    mirrored[target + '_team_roles'] = observation[source + '_team_roles']
  return mirrored


class MacroFeaturesTest(absltest.TestCase):

  def test_feature_contract_is_exactly_50(self):
    encoder = features.MacroFeatureEncoder(3600)
    encoded = encoder.encode(_observation(), True, 4, 300)

    self.assertEqual(encoded.shape, (50,))
    self.assertEqual(len(features.FEATURE_NAMES), 50)
    self.assertEqual(encoded.dtype, np.float32)
    self.assertTrue(np.all(np.isfinite(encoded)))

  def test_side_rotation_preserves_team_relative_features(self):
    encoder = features.MacroFeatureEncoder(3600)
    left = encoder.encode(_observation(), True, 4, 300)
    right = encoder.encode(_mirror(_observation()), False, 4, 300)

    indices = [index for index in range(50) if index != 3]
    np.testing.assert_allclose(left[indices], right[indices], atol=1e-6)
    self.assertEqual(left[3], 1.0)
    self.assertEqual(right[3], 0.0)

  def test_potential_is_side_invariant(self):
    left = features.potential(_observation(), True)
    right = features.potential(_mirror(_observation()), False)

    self.assertAlmostEqual(left, right)


if __name__ == '__main__':
  absltest.main()
