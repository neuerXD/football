# coding=utf-8
"""Tests for strict LLM teacher voting."""

from gfootball.rl import teacher


def _response(action, confidence=0.9):
  return {'action_id': action, 'confidence': confidence, 'reason': 'test'}


def test_majority_vote_accepts_two_of_three():
  label, confidence, _, valid = teacher.majority_vote([
      _response(3, 0.8), _response(3, 1.0), _response(4)
  ])
  assert label == 3
  assert confidence == 0.9
  assert valid == 3


def test_majority_vote_rejects_three_way_tie():
  label, confidence, reason, valid = teacher.majority_vote([
      _response(0), _response(1), _response(2)
  ])
  assert label == -1
  assert confidence == 0.0
  assert reason == ''
  assert valid == 3


def test_majority_vote_counts_invalid_samples_against_majority():
  label, _, _, valid = teacher.majority_vote([
      _response(5), None, None
  ])
  assert label == -1
  assert valid == 1
