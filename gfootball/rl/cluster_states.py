# coding=utf-8
"""Standardizes states and chooses representative K-Means states."""

from __future__ import absolute_import

import argparse
import json
import os

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from gfootball.rl import provenance


def cluster(input_dir, output_dir, num_clusters=3000, seed=23):
  input_dir = os.path.abspath(input_dir)
  output_dir = os.path.abspath(output_dir)
  os.makedirs(output_dir, exist_ok=True)
  dataset = np.load(os.path.join(input_dir, 'states.npz'))
  states = np.asarray(dataset['states'], dtype=np.float32)
  if len(states) == 0:
    raise ValueError('State dataset is empty')
  num_clusters = min(int(num_clusters), len(states))
  mean = states.mean(axis=0)
  std = states.std(axis=0)
  std[std < 1e-6] = 1.0
  standardized = (states - mean) / std
  model = MiniBatchKMeans(
      n_clusters=num_clusters,
      random_state=seed,
      batch_size=min(2048, len(states)),
      n_init=3,
      max_iter=100,
      reassignment_ratio=0.01,
  )
  labels = model.fit_predict(standardized)
  representatives = []
  selected = set()
  for cluster_id in range(num_clusters):
    indices = np.flatnonzero(labels == cluster_id)
    if len(indices) == 0:
      indices = np.arange(len(states))
    distances = np.sum(np.square(
        standardized[indices] - model.cluster_centers_[cluster_id]), axis=1)
    for offset in np.argsort(distances):
      candidate = int(indices[offset])
      if candidate not in selected:
        representatives.append(candidate)
        selected.add(candidate)
        break
    else:
      raise RuntimeError('Could not select a unique cluster representative')
  representatives = np.asarray(representatives, dtype=np.int64)
  if len(np.unique(representatives)) != num_clusters:
    raise AssertionError('Cluster representatives must be unique')
  np.savez_compressed(
      os.path.join(output_dir, 'clusters.npz'),
      centers=model.cluster_centers_.astype(np.float32),
      labels=labels.astype(np.int32),
      representative_indices=representatives,
      representative_states=states[representatives],
      normalizer_mean=mean.astype(np.float32),
      normalizer_std=std.astype(np.float32),
  )
  manifest = {
      'input_dir': input_dir,
      'num_states': int(len(states)),
      'num_clusters': int(num_clusters),
      'cluster_seed': seed,
      'feature_dim': int(states.shape[1]),
      'empty_clusters': int(np.sum(
          np.bincount(labels, minlength=num_clusters) == 0)),
      'provenance': provenance.experiment_metadata(),
  }
  with open(os.path.join(output_dir, 'cluster_manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
  return manifest


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--input-dir', required=True)
  parser.add_argument('--output-dir', required=True)
  parser.add_argument('--num-clusters', type=int, default=3000)
  parser.add_argument('--seed', type=int, default=23)
  args = parser.parse_args()
  print(cluster(**vars(args)), flush=True)


if __name__ == '__main__':
  main()
