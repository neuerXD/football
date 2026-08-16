# coding=utf-8
"""Tests for paired evaluation statistics."""

from gfootball.rl import report


def _record(key, win, score_diff):
  return {
      'scenario_key': key,
      'win': win,
      'draw': int(score_diff == 0),
      'loss': int(score_diff < 0),
      'score_diff': score_diff,
      'mean_policy_latency_ms': 1.0,
      'p95_policy_latency_ms': 2.0,
      'macro_decisions': 36,
      'action_counts': [1] * 12,
  }


def test_bootstrap_constant_values():
  assert report.bootstrap_ci([0.25] * 8, samples=100) == [0.25, 0.25]


def test_paired_comparison_uses_common_scenarios():
  candidate = [_record('a', 1, 2), _record('b', 0, 0)]
  reference = [_record('a', 0, 0), _record('b', 0, -1),
               _record('c', 1, 3)]
  result = report.paired_comparison(candidate, reference,
                                    bootstrap_samples=100)
  assert result['paired_scenarios'] == 2
  assert result['win_rate_delta'] == 0.5
  assert result['score_diff_delta'] == 1.5


def test_summarize_averages_repeated_optimization_seeds():
  records = [_record('a', 1, 1), _record('a', 0, -1),
             _record('b', 1, 2), _record('b', 1, 0)]
  result = report.summarize(records, bootstrap_samples=100)
  assert result['games'] == 4
  assert result['paired_scenarios'] == 2
  assert result['win_rate'] == 0.75
  assert result['mean_score_diff'] == 0.5
