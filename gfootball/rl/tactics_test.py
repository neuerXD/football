# coding=utf-8
"""Tests for tactical templates and experiment protocol."""

from absl.testing import absltest
from gfootball.env import tactical_plan
from gfootball.rl import protocol
from gfootball.rl import tactics


class TacticsTest(absltest.TestCase):

  def test_action_space_has_twelve_unique_templates(self):
    self.assertLen(tactics.TACTIC_TEMPLATES, 12)
    self.assertLen(set(tactics.TACTIC_NAMES), 12)
    engine_signatures = set()
    for action_id in range(12):
      plan = tactics.tactic_plan(action_id)
      values = tactical_plan.plan_to_engine_tactics(plan)
      entries = tactical_plan.formation_to_engine_entries(
          plan['formation'], plan['attack_focus'])
      signature = tuple(round(values[key], 4)
                        for key in tactical_plan.ENGINE_TACTIC_KEYS)
      signature += tuple(
          (round(entry.position[0], 4), round(entry.position[1], 4))
          for entry in entries)
      engine_signatures.add(signature)
    self.assertLen(engine_signatures, 12)

  def test_left_and_right_focus_are_mirrored(self):
    left = tactics.tactic_plan(6)
    right = tactics.tactic_plan(7)
    left_entries = tactical_plan.formation_to_engine_entries(
        left['formation'], left['attack_focus'])
    right_entries = tactical_plan.formation_to_engine_entries(
        right['formation'], right['attack_focus'])

    self.assertLen(left_entries, 11)
    self.assertLen(right_entries, 11)
    self.assertNotEqual(left_entries[6].position[1],
                        right_entries[6].position[1])

  def test_train_and_eval_seeds_are_disjoint(self):
    self.assertEmpty(
        set(protocol.TRAIN_ENV_SEEDS).intersection(protocol.EVAL_ENV_SEEDS))
    self.assertEqual(protocol.validate_seed(1000, 'train'), 1000)
    self.assertEqual(protocol.validate_seed(20000, 'eval'), 20000)
    with self.assertRaises(ValueError):
      protocol.validate_seed(20000, 'train')
    with self.assertRaises(ValueError):
      protocol.validate_seed(1000, 'eval')


if __name__ == '__main__':
  absltest.main()
