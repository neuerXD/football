# coding=utf-8
"""Merges deterministic multi-GPU teacher-label shards."""

from __future__ import absolute_import

import argparse
import json
import os

import numpy as np


def _audit(labels, confidence, valid_samples, manifests):
  accepted = labels >= 0
  rejected = ~accepted
  no_valid_response = rejected & (valid_samples == 0)
  no_majority = rejected & (valid_samples > 0) & (confidence <= 0.0)
  low_confidence = rejected & (confidence > 0.0)
  sample_counts = {
      int(manifest['num_samples']) for manifest in manifests
      if 'num_samples' in manifest
  }
  num_samples = sample_counts.pop() if len(sample_counts) == 1 else None
  format_failures = None
  format_failure_rate = None
  if num_samples is not None:
    total_samples = int(len(labels) * num_samples)
    format_failures = int(total_samples - np.sum(valid_samples))
    format_failure_rate = float(format_failures / float(total_samples))
  accepted_confidence = confidence[accepted]
  return {
      'class_counts': np.bincount(
          labels[accepted], minlength=12).astype(int).tolist(),
      'mean_accepted_confidence': (
          float(np.mean(accepted_confidence))
          if len(accepted_confidence) else None),
      'format_failures': format_failures,
      'format_failure_rate': format_failure_rate,
      'no_valid_response': int(np.sum(no_valid_response)),
      'no_majority': int(np.sum(no_majority)),
      'low_confidence': int(np.sum(low_confidence)),
      'low_confidence_rate': float(np.mean(low_confidence)),
  }


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
      shard_records = [json.loads(line) for line in f if line.strip()]
    record_indices = [int(item['cluster_index']) for item in shard_records]
    if (len(record_indices) != len(indices) or
        len(set(record_indices)) != len(record_indices) or
        set(record_indices) != set(indices.tolist())):
      raise ValueError('Teacher shard JSONL does not match its NPZ indices')
    records.extend(shard_records)
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
      'audit': _audit(labels, confidence, valid_samples, manifests),
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
