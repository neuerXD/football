# coding=utf-8
"""Tests for merging teacher-label shards."""

import json
import os

import numpy as np

from gfootball.rl import merge_teacher


def _write_shard(path, total, index, indices):
  os.makedirs(path)
  labels = np.asarray(indices, dtype=np.int64) % 12
  np.savez_compressed(
      os.path.join(path, 'teacher_labels.npz'),
      labels=labels,
      confidence=np.full(len(indices), 0.9, dtype=np.float32),
      valid_samples=np.full(len(indices), 3, dtype=np.int8),
      cluster_indices=np.asarray(indices, dtype=np.int64),
  )
  with open(os.path.join(path, 'teacher_labels.jsonl'), 'w') as output:
    for cluster_index, label in zip(indices, labels):
      output.write(json.dumps({
          'cluster_index': cluster_index,
          'action_id': int(label),
      }) + '\n')
  with open(os.path.join(path, 'teacher_manifest.json'), 'w') as output:
    json.dump({
        'clusters': total,
        'num_samples': 3,
        'shard_index': index,
        'num_shards': 2,
    }, output)


def test_merge_shards_restores_cluster_order(tmp_path):
  shard0 = tmp_path / 'shard0'
  shard1 = tmp_path / 'shard1'
  output = tmp_path / 'merged'
  _write_shard(str(shard0), 6, 0, [0, 2, 4])
  _write_shard(str(shard1), 6, 1, [1, 3, 5])
  result = merge_teacher.merge_shards(
      [str(shard0), str(shard1)], str(output))
  data = np.load(output / 'teacher_labels.npz')
  assert result['accepted'] == 6
  assert result['audit']['class_counts'] == [1, 1, 1, 1, 1, 1] + [0] * 6
  assert result['audit']['format_failures'] == 0
  assert result['audit']['low_confidence'] == 0
  np.testing.assert_array_equal(data['labels'], np.arange(6) % 12)


def test_merge_shards_audits_rejections(tmp_path):
  shard0 = tmp_path / 'shard0'
  shard1 = tmp_path / 'shard1'
  output = tmp_path / 'merged'
  _write_shard(str(shard0), 6, 0, [0, 2, 4])
  _write_shard(str(shard1), 6, 1, [1, 3, 5])
  data = np.load(shard0 / 'teacher_labels.npz')
  labels = data['labels'].copy()
  confidence = data['confidence'].copy()
  valid_samples = data['valid_samples'].copy()
  labels[:2] = -1
  confidence[0] = 0.0
  confidence[1] = 0.4
  valid_samples[0] = 2
  valid_samples[1] = 1
  np.savez_compressed(
      shard0 / 'teacher_labels.npz', labels=labels, confidence=confidence,
      valid_samples=valid_samples, cluster_indices=data['cluster_indices'])
  result = merge_teacher.merge_shards(
      [str(shard0), str(shard1)], str(output))
  assert result['accepted'] == 4
  assert result['audit']['no_majority'] == 1
  assert result['audit']['low_confidence'] == 1
  assert result['audit']['format_failures'] == 3
  assert result['audit']['format_failure_rate'] == 3.0 / 18.0


def test_merge_shards_rejects_mismatched_jsonl(tmp_path):
  shard0 = tmp_path / 'shard0'
  shard1 = tmp_path / 'shard1'
  output = tmp_path / 'merged'
  _write_shard(str(shard0), 4, 0, [0, 2])
  _write_shard(str(shard1), 4, 1, [1, 3])
  with open(shard0 / 'teacher_labels.jsonl', 'w') as records:
    records.write(json.dumps({'cluster_index': 0, 'action_id': 0}) + '\n')
  with np.testing.assert_raises_regex(ValueError, 'JSONL'):
    merge_teacher.merge_shards(
        [str(shard0), str(shard1)], str(output))
