# coding=utf-8
"""Aggregates tactical evaluations, paired bootstrap CIs, and plots."""

from __future__ import absolute_import

import argparse
import collections
import json
import os

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

from gfootball.rl import tactics
from gfootball.rl import provenance


def bootstrap_ci(values, samples=10000, seed=71):
  values = np.asarray(values, dtype=np.float64)
  if len(values) == 0:
    return [None, None]
  if len(values) == 1:
    return [float(values[0]), float(values[0])]
  rng = np.random.RandomState(seed)
  means = np.empty(int(samples), dtype=np.float64)
  for start in range(0, int(samples), 1000):
    count = min(1000, int(samples) - start)
    indices = rng.randint(0, len(values), size=(count, len(values)))
    means[start:start + count] = values[indices].mean(axis=1)
  return [float(np.percentile(means, 2.5)),
          float(np.percentile(means, 97.5))]


def _scenario_values(records, field):
  grouped = collections.defaultdict(list)
  for record in records:
    grouped[record['scenario_key']].append(float(record[field]))
  return {key: float(np.mean(values)) for key, values in grouped.items()}


def summarize(records, bootstrap_samples=10000, seed=71):
  wins = _scenario_values(records, 'win')
  score_diffs = _scenario_values(records, 'score_diff')
  action_counts = np.sum(
      [record['action_counts'] for record in records], axis=0).astype(float)
  action_total = max(1.0, float(action_counts.sum()))
  return {
      'games': len(records),
      'paired_scenarios': len(wins),
      'wins': int(sum(record['win'] for record in records)),
      'draws': int(sum(record['draw'] for record in records)),
      'losses': int(sum(record['loss'] for record in records)),
      'win_rate': float(np.mean(list(wins.values()))),
      'win_rate_ci95': bootstrap_ci(
          list(wins.values()), bootstrap_samples, seed),
      'mean_score_diff': float(np.mean(list(score_diffs.values()))),
      'mean_score_diff_ci95': bootstrap_ci(
          list(score_diffs.values()), bootstrap_samples, seed + 1),
      'mean_policy_latency_ms': float(np.mean(
          [record['mean_policy_latency_ms'] for record in records])),
      'p95_policy_latency_ms': float(np.mean(
          [record['p95_policy_latency_ms'] for record in records])),
      'mean_macro_decisions': float(np.mean(
          [record['macro_decisions'] for record in records])),
      'action_distribution': (action_counts / action_total).tolist(),
  }


def paired_comparison(candidate, reference, bootstrap_samples=10000, seed=73):
  result = {}
  for output_name, field in (('win_rate_delta', 'win'),
                             ('score_diff_delta', 'score_diff')):
    candidate_values = _scenario_values(candidate, field)
    reference_values = _scenario_values(reference, field)
    keys = sorted(set(candidate_values).intersection(reference_values))
    deltas = [candidate_values[key] - reference_values[key] for key in keys]
    result[output_name] = float(np.mean(deltas)) if deltas else None
    result[output_name + '_ci95'] = bootstrap_ci(
        deltas, bootstrap_samples, seed)
    result['paired_scenarios'] = len(keys)
    seed += 1
  return result


def _load_eval_dir(path):
  with open(os.path.join(path, 'evaluation_manifest.json')) as f:
    manifest = json.load(f)
  records = []
  with open(os.path.join(path, 'episodes.jsonl')) as f:
    for line in f:
      if line.strip():
        records.append(json.loads(line))
  return manifest, records


def _group_runs(eval_dirs):
  grouped = collections.defaultdict(lambda: collections.defaultdict(list))
  manifests = []
  for path in eval_dirs:
    manifest, records = _load_eval_dir(path)
    manifests.append(manifest)
    policy = manifest['policy']
    step = int(manifest['policy_metadata'].get('global_step', 0))
    grouped[policy][step].extend(records)
  return grouped, manifests


def _final_records(grouped):
  return {
      policy: by_step[max(by_step)]
      for policy, by_step in grouped.items()
  }


def _sample_efficiency(curves, target_win_rate):
  result = {}
  for policy, points in curves.items():
    reached = [point['global_step'] for point in points
               if point['win_rate'] >= target_win_rate]
    result[policy] = min(reached) if reached else None
  return result


def _plot_win_rate(summaries, output_dir):
  names = sorted(summaries)
  values = [summaries[name]['win_rate'] for name in names]
  errors = []
  for name, value in zip(names, values):
    low, high = summaries[name]['win_rate_ci95']
    errors.append([value - low, high - value])
  fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.1), 4.5))
  ax.bar(names, values, color='#2f6f4e')
  ax.errorbar(names, values, yerr=np.asarray(errors).T, fmt='none',
              ecolor='#222222', capsize=4)
  ax.set_ylabel('Win rate')
  ax.set_ylim(0.0, 1.0)
  ax.tick_params(axis='x', rotation=25)
  fig.tight_layout()
  fig.savefig(os.path.join(output_dir, 'win_rate.png'), dpi=180)
  plt.close(fig)


def _plot_tactics(summaries, output_dir):
  names = sorted(summaries)
  values = np.asarray([
      summaries[name]['action_distribution'] for name in names
  ])
  fig, ax = plt.subplots(figsize=(12, max(3.5, len(names) * 0.65)))
  image = ax.imshow(values, aspect='auto', vmin=0.0,
                    vmax=max(0.2, float(values.max())), cmap='YlGn')
  ax.set_yticks(np.arange(len(names)))
  ax.set_yticklabels(names)
  ax.set_xticks(np.arange(tactics.NUM_TACTICS))
  ax.set_xticklabels(tactics.TACTIC_NAMES, rotation=35, ha='right')
  fig.colorbar(image, ax=ax, label='Action frequency')
  fig.tight_layout()
  fig.savefig(os.path.join(output_dir, 'tactic_distribution.png'), dpi=180)
  plt.close(fig)


def _plot_latency(summaries, output_dir):
  names = sorted(summaries)
  values = [summaries[name]['mean_policy_latency_ms'] for name in names]
  fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.1), 4.5))
  ax.bar(names, values, color='#b65f42')
  ax.set_ylabel('Mean policy latency (ms/decision)')
  ax.tick_params(axis='x', rotation=25)
  fig.tight_layout()
  fig.savefig(os.path.join(output_dir, 'policy_latency.png'), dpi=180)
  plt.close(fig)


def _plot_sample_efficiency(curves, target, output_dir):
  if not any(len(points) > 1 for points in curves.values()):
    return
  fig, ax = plt.subplots(figsize=(7.5, 4.5))
  for name in sorted(curves):
    points = curves[name]
    ax.plot([item['global_step'] for item in points],
            [item['win_rate'] for item in points], marker='o', label=name)
  ax.axhline(target, color='#444444', linestyle='--', label='target')
  ax.set_xlabel('Macro environment transitions')
  ax.set_ylabel('Win rate')
  ax.set_ylim(0.0, 1.0)
  ax.legend()
  fig.tight_layout()
  fig.savefig(os.path.join(output_dir, 'sample_efficiency.png'), dpi=180)
  plt.close(fig)


def _format_delta(item, name):
  value = item[name]
  low, high = item[name + '_ci95']
  if value is None:
    return 'n/a'
  return '{:.3f} [{:.3f}, {:.3f}]'.format(value, low, high)


def _write_markdown(path, summaries, comparisons, target, efficiency,
                    reference):
  lines = [
      '# Macro Tactical RL Evaluation',
      '',
      'Reference policy for paired deltas: `{}`.'.format(reference),
      '',
      '| Policy | W/D/L | Win rate (95% CI) | Mean goal diff (95% CI) | '
      'Latency ms |',
      '|---|---:|---:|---:|---:|',
  ]
  for name in sorted(summaries):
    item = summaries[name]
    lines.append(
        '| {} | {}/{}/{} | {:.3f} [{:.3f}, {:.3f}] | {:.3f} '
        '[{:.3f}, {:.3f}] | {:.3f} |'.format(
            name, item['wins'], item['draws'], item['losses'],
            item['win_rate'], item['win_rate_ci95'][0],
            item['win_rate_ci95'][1], item['mean_score_diff'],
            item['mean_score_diff_ci95'][0],
            item['mean_score_diff_ci95'][1],
            item['mean_policy_latency_ms']))
  lines.extend([
      '',
      '## Paired Comparisons',
      '',
      '| Policy | Win-rate delta (95% CI) | Goal-diff delta (95% CI) | Pairs |',
      '|---|---:|---:|---:|',
  ])
  for name in sorted(comparisons):
    item = comparisons[name]
    lines.append('| {} | {} | {} | {} |'.format(
        name, _format_delta(item, 'win_rate_delta'),
        _format_delta(item, 'score_diff_delta'), item['paired_scenarios']))
  lines.extend([
      '',
      '## Sample Efficiency',
      '',
      'Target win rate: `{:.3f}`.'.format(target),
      '',
  ])
  for name in sorted(efficiency):
    value = efficiency[name]
    lines.append('- `{}`: {}'.format(
        name, value if value is not None else 'not reached'))
  with open(path, 'w') as output:
    output.write('\n'.join(lines) + '\n')


def generate_report(eval_dirs, output_dir, reference='', target_win_rate=None,
                    bootstrap_samples=10000, seed=71):
  output_dir = os.path.abspath(output_dir)
  os.makedirs(output_dir, exist_ok=True)
  grouped, manifests = _group_runs(eval_dirs)
  final_records = _final_records(grouped)
  summaries = {
      name: summarize(records, bootstrap_samples, seed + index * 10)
      for index, (name, records) in enumerate(sorted(final_records.items()))
  }
  if not reference:
    reference = 'scratch_ppo' if 'scratch_ppo' in summaries else sorted(
        summaries)[0]
  if reference not in summaries:
    raise ValueError('Reference policy not present: {}'.format(reference))
  comparisons = {
      name: paired_comparison(records, final_records[reference],
                              bootstrap_samples, seed + index * 10)
      for index, (name, records) in enumerate(sorted(final_records.items()))
      if name != reference
  }
  curves = {}
  for name, by_step in grouped.items():
    curves[name] = []
    for step, records in sorted(by_step.items()):
      item = summarize(records, bootstrap_samples, seed)
      curves[name].append({
          'global_step': step,
          'win_rate': item['win_rate'],
          'win_rate_ci95': item['win_rate_ci95'],
      })
  if target_win_rate is None:
    learned = [summaries[name]['win_rate'] for name in
               ('scratch_ppo', 'bc_ppo') if name in summaries]
    target_win_rate = min(learned) if learned else max(
        item['win_rate'] for item in summaries.values())
  efficiency = _sample_efficiency(curves, float(target_win_rate))
  result = {
      'reference': reference,
      'bootstrap_samples': int(bootstrap_samples),
      'target_win_rate': float(target_win_rate),
      'summaries': summaries,
      'paired_comparisons': comparisons,
      'sample_efficiency_curves': curves,
      'steps_to_target_win_rate': efficiency,
      'source_manifests': manifests,
      'provenance': provenance.experiment_metadata(),
  }
  with open(os.path.join(output_dir, 'report.json'), 'w') as f:
    json.dump(result, f, indent=2, sort_keys=True)
  _write_markdown(os.path.join(output_dir, 'report.md'), summaries,
                  comparisons, float(target_win_rate), efficiency, reference)
  _plot_win_rate(summaries, output_dir)
  _plot_tactics(summaries, output_dir)
  _plot_latency(summaries, output_dir)
  _plot_sample_efficiency(curves, float(target_win_rate), output_dir)
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--eval-dirs', nargs='+', required=True)
  parser.add_argument('--output-dir', required=True)
  parser.add_argument('--reference', default='')
  parser.add_argument('--target-win-rate', type=float, default=None)
  parser.add_argument('--bootstrap-samples', type=int, default=10000)
  parser.add_argument('--seed', type=int, default=71)
  args = parser.parse_args()
  result = generate_report(**vars(args))
  print(json.dumps({
      'output_dir': os.path.abspath(args.output_dir),
      'policies': sorted(result['summaries']),
      'reference': result['reference'],
  }, sort_keys=True), flush=True)


if __name__ == '__main__':
  main()
