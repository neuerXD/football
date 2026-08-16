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
  np.testing.assert_array_equal(data['labels'], np.arange(6) % 12)
