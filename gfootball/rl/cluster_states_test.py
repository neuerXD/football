# coding=utf-8
"""Tests for representative-state clustering."""

import numpy as np

from gfootball.rl import cluster_states


def test_cluster_representatives_are_unique_with_empty_clusters(tmp_path,
                                                               monkeypatch):
  input_dir = tmp_path / 'input'
  output_dir = tmp_path / 'output'
  input_dir.mkdir()
  states = np.arange(30, dtype=np.float32).reshape(10, 3)
  np.savez_compressed(input_dir / 'states.npz', states=states)

  class FakeKMeans(object):

    def __init__(self, n_clusters, **unused):
      self.cluster_centers_ = np.asarray([
          [0.0, 0.0, 0.0],
          [1.0, 1.0, 1.0],
          [-1.0, -1.0, -1.0],
      ])[:n_clusters]

    def fit_predict(self, unused_states):
      return np.zeros(len(unused_states), dtype=np.int32)

  monkeypatch.setattr(cluster_states, 'MiniBatchKMeans', FakeKMeans)
  result = cluster_states.cluster(
      str(input_dir), str(output_dir), num_clusters=3)
  data = np.load(output_dir / 'clusters.npz')
  assert result['empty_clusters'] == 2
  assert len(np.unique(data['representative_indices'])) == 3
