# coding=utf-8
"""Discrete and interpretable macro-tactical action space."""

import copy


TACTIC_TEMPLATES = (
    {
        'name': 'balanced', 'label': 'Balanced', 'formation': '4-4-2',
        'defensive_line': 0.50, 'pressing': 0.50, 'width': 0.55,
        'attack_focus': 'balanced', 'pass_risk': 0.35, 'tempo': 0.50,
    },
    {
        'name': 'possession', 'label': 'Possession', 'formation': '4-2-3-1',
        'defensive_line': 0.55, 'pressing': 0.55, 'width': 0.62,
        'attack_focus': 'balanced', 'pass_risk': 0.18, 'tempo': 0.38,
    },
    {
        'name': 'high_press', 'label': 'High press', 'formation': '4-3-3',
        'defensive_line': 0.72, 'pressing': 0.90, 'width': 0.68,
        'attack_focus': 'center', 'pass_risk': 0.55, 'tempo': 0.78,
    },
    {
        'name': 'low_block', 'label': 'Low block', 'formation': '5-4-1',
        'defensive_line': 0.20, 'pressing': 0.22, 'width': 0.72,
        'attack_focus': 'balanced', 'pass_risk': 0.18, 'tempo': 0.25,
    },
    {
        'name': 'counterattack', 'label': 'Fast counter',
        'formation': '4-2-3-1', 'defensive_line': 0.32, 'pressing': 0.42,
        'width': 0.55, 'attack_focus': 'center', 'pass_risk': 0.68,
        'tempo': 0.82,
    },
    {
        'name': 'direct_attack', 'label': 'Direct attack',
        'formation': '4-3-3', 'defensive_line': 0.60, 'pressing': 0.65,
        'width': 0.72, 'attack_focus': 'balanced', 'pass_risk': 0.82,
        'tempo': 0.86,
    },
    {
        'name': 'left_focus', 'label': 'Left focus', 'formation': '4-3-3',
        'defensive_line': 0.52, 'pressing': 0.58, 'width': 0.82,
        'attack_focus': 'left', 'pass_risk': 0.45, 'tempo': 0.63,
    },
    {
        'name': 'right_focus', 'label': 'Right focus', 'formation': '4-3-3',
        'defensive_line': 0.52, 'pressing': 0.58, 'width': 0.82,
        'attack_focus': 'right', 'pass_risk': 0.45, 'tempo': 0.63,
    },
    {
        'name': 'central_overload', 'label': 'Central overload',
        'formation': '4-1-2-1-2', 'defensive_line': 0.60,
        'pressing': 0.62, 'width': 0.38, 'attack_focus': 'center',
        'pass_risk': 0.50, 'tempo': 0.68,
    },
    {
        'name': 'protect_lead', 'label': 'Protect lead',
        'formation': '5-3-2', 'defensive_line': 0.28, 'pressing': 0.35,
        'width': 0.68, 'attack_focus': 'balanced', 'pass_risk': 0.22,
        'tempo': 0.34,
    },
    {
        'name': 'chase_game', 'label': 'Chase game', 'formation': '3-4-3',
        'defensive_line': 0.78, 'pressing': 0.78, 'width': 0.78,
        'attack_focus': 'center', 'pass_risk': 0.72, 'tempo': 0.82,
    },
    {
        'name': 'all_out_attack', 'label': 'All-out attack',
        'formation': '3-3-4', 'defensive_line': 0.90, 'pressing': 0.92,
        'width': 0.90, 'attack_focus': 'center', 'pass_risk': 0.90,
        'tempo': 0.95,
    },
)

NUM_TACTICS = len(TACTIC_TEMPLATES)
TACTIC_NAMES = tuple(item['name'] for item in TACTIC_TEMPLATES)


def tactic_plan(action_id):
  action_id = int(action_id)
  if action_id < 0 or action_id >= NUM_TACTICS:
    raise ValueError('Invalid tactic action: {}'.format(action_id))
  plan = copy.deepcopy(TACTIC_TEMPLATES[action_id])
  plan['notes'] = 'Discrete macro tactic: {}'.format(plan['label'])
  return plan


def tactic_id(name):
  try:
    return TACTIC_NAMES.index(str(name))
  except ValueError:
    raise ValueError('Unknown tactic: {}'.format(name))
