# coding=utf-8
"""Merges deterministic multi-GPU teacher-label shards."""

from __future__ import absolute_import

import argparse
import json
import os

import numpy as np


def merge_shards(shard_dirs, output_dir):
  if not shard_dirs:
    raise ValueError('At least one shard directory is required')
  manifests = []
  shards = []
  for shard_dir in shard_dirs:
    with open(os.path.join(shard_dir, 'teacher_manifest.json')) as f:
      manifest = json.load(f)
    data = np.load(os.path.join(shard_dir, 'teacher_labels.npz'))
    manifests.append(manifest)
    shards.append({
        'indices': np.asarray(data['cluster_indices'], dtype=np.int64),
        'labels': np.asarray(data['labels'], dtype=np.int64),
        'confidence': np.asarray(data['confidence'], dtype=np.float32),
        'valid_samples': np.asarray(data['valid_samples'], dtype=np.int8),
        'dir': shard_dir,
    })
  total = int(manifests[0]['clusters'])
  for manifest in manifests:
    if int(manifest['clusters']) != total:
      raise ValueError('Teacher shards use different cluster counts')
  labels = np.full(total, -1, dtype=np.int64)
  confidence = np.zeros(total, dtype=np.float32)
  valid_samples = np.zeros(total, dtype=np.int8)
  covered = np.zeros(total, dtype=bool)
  records = []
  for shard in shards:
    indices = shard['indices']
    if (len(indices) != len(shard['labels']) or
        np.any(indices < 0) or np.any(indices >= total)):
      raise ValueError('Invalid teacher shard indices')
    if np.any(covered[indices]):
      raise ValueError('Teacher shards overlap')
    covered[indices] = True
    labels[indices] = shard['labels']
    confidence[indices] = shard['confidence']
    valid_samples[indices] = shard['valid_samples']
    with open(os.path.join(shard['dir'], 'teacher_labels.jsonl')) as f:
      records.extend(json.loads(line) for line in f if line.strip())
  if not np.all(covered):
    missing = np.flatnonzero(~covered)
    raise ValueError('Teacher shards miss {} clusters'.format(len(missing)))
  records.sort(key=lambda item: item['cluster_index'])
  output_dir = os.path.abspath(output_dir)
  os.makedirs(output_dir, exist_ok=True)
  np.savez_compressed(
      os.path.join(output_dir, 'teacher_labels.npz'),
      labels=labels,
      confidence=confidence,
      valid_samples=valid_samples,
      cluster_indices=np.arange(total, dtype=np.int64),
  )
  with open(os.path.join(output_dir, 'teacher_labels.jsonl'), 'w') as output:
    for record in records:
      output.write(json.dumps(record, sort_keys=True))
      output.write('\n')
  result = {
      'clusters': total,
      'accepted': int(np.sum(labels >= 0)),
      'acceptance_rate': float(np.mean(labels >= 0)),
      'num_shards': len(shards),
      'source_dirs': [os.path.abspath(path) for path in shard_dirs],
      'source_manifests': manifests,
  }
  with open(os.path.join(output_dir, 'teacher_manifest.json'), 'w') as f:
    json.dump(result, f, indent=2, sort_keys=True)
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--shard-dirs', nargs='+', required=True)
  parser.add_argument('--output-dir', required=True)
  args = parser.parse_args()
  print(json.dumps(merge_shards(**vars(args)), sort_keys=True), flush=True)


if __name__ == '__main__':
  main()
