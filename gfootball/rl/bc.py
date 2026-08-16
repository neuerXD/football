# coding=utf-8
"""Weighted behavior cloning from validated LLM tactical labels."""

from __future__ import absolute_import

import argparse
import json
import os

import numpy as np
import torch
from torch.nn import functional as F

from gfootball.rl import models
from gfootball.rl import provenance


def _accuracy(logits, labels):
  return float((torch.argmax(logits, dim=-1) == labels).float().mean().item())


def train_bc(cluster_dir, teacher_dir, output_dir, epochs=50, batch_size=256,
            learning_rate=3e-4, seed=41, device=None):
  output_dir = os.path.abspath(output_dir)
  os.makedirs(output_dir, exist_ok=True)
  run_provenance = provenance.experiment_metadata()
  cluster = np.load(os.path.join(cluster_dir, 'clusters.npz'))
  teacher = np.load(os.path.join(teacher_dir, 'teacher_labels.npz'))
  states = np.asarray(cluster['representative_states'], dtype=np.float32)
  labels = np.asarray(teacher['labels'], dtype=np.int64)
  confidence = np.asarray(teacher['confidence'], dtype=np.float32)
  mask = labels >= 0
  states, labels, confidence = states[mask], labels[mask], confidence[mask]
  if len(states) < 12:
    raise ValueError('Too few accepted teacher labels: {}'.format(len(states)))
  rng = np.random.RandomState(seed)
  order = rng.permutation(len(states))
  train_end = max(1, int(0.70 * len(order)))
  val_end = max(train_end + 1, int(0.85 * len(order)))
  train_indices, val_indices, test_indices = np.split(
      order, [train_end, val_end])
  mean = np.asarray(cluster['normalizer_mean'], dtype=np.float32)
  std = np.asarray(cluster['normalizer_std'], dtype=np.float32)
  std[std < 1e-6] = 1.0
  normalized = (states - mean) / std
  device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
  model = models.ActorCritic(states.shape[1], 12).to(device)
  optimizer = torch.optim.Adam(model.actor.parameters(), lr=learning_rate)
  counts = np.bincount(labels[train_indices], minlength=12).astype(np.float32)
  class_weights = np.sqrt(np.maximum(1.0, counts.sum() /
                                    (12.0 * np.maximum(counts, 1.0))))
  class_weights /= class_weights.mean()
  weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32,
                                  device=device)
  train_x = torch.as_tensor(normalized[train_indices], dtype=torch.float32,
                            device=device)
  train_y = torch.as_tensor(labels[train_indices], dtype=torch.long,
                            device=device)
  train_conf = torch.as_tensor(confidence[train_indices], dtype=torch.float32,
                               device=device)
  val_x = torch.as_tensor(normalized[val_indices], dtype=torch.float32,
                          device=device)
  val_y = torch.as_tensor(labels[val_indices], dtype=torch.long, device=device)
  test_x = torch.as_tensor(normalized[test_indices], dtype=torch.float32,
                           device=device)
  test_y = torch.as_tensor(labels[test_indices], dtype=torch.long, device=device)
  history = []
  for epoch in range(epochs):
    permutation = torch.randperm(len(train_x), device=device)
    model.actor.train()
    losses = []
    for start in range(0, len(train_x), batch_size):
      index = permutation[start:start + batch_size]
      logits = model.actor(train_x[index])
      per_sample = F.cross_entropy(
          logits, train_y[index], weight=weight_tensor, reduction='none')
      loss = (per_sample * train_conf[index]).mean()
      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 0.5)
      optimizer.step()
      losses.append(loss.item())
    model.actor.eval()
    with torch.no_grad():
      train_logits = model.actor(train_x)
      val_logits = model.actor(val_x)
      test_logits = model.actor(test_x)
    record = {
        'epoch': epoch + 1,
        'loss': float(np.mean(losses)),
        'train_accuracy': _accuracy(train_logits, train_y),
        'val_accuracy': _accuracy(val_logits, val_y),
        'test_accuracy': _accuracy(test_logits, test_y),
    }
    history.append(record)
    if epoch == 0 or (epoch + 1) % 5 == 0:
      print(record, flush=True)
  final_path = os.path.join(output_dir, 'bc_checkpoint.pt')
  torch.save({
      'actor_state_dict': model.actor.state_dict(),
      'normalizer_mean': mean,
      'normalizer_std': std,
      'normalizer_count': int(len(cluster['labels'])),
      'class_weights': class_weights,
      'history': history,
      'feature_dim': int(states.shape[1]),
      'action_dim': 12,
      'seed': seed,
  }, final_path)
  metrics = {
      'accepted_labels': int(len(states)),
      'class_counts': counts.astype(int).tolist(),
      'train_size': int(len(train_indices)),
      'val_size': int(len(val_indices)),
      'test_size': int(len(test_indices)),
      'final': history[-1],
      'provenance': run_provenance,
  }
  with open(os.path.join(output_dir, 'bc_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2, sort_keys=True)
  return final_path, metrics


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--cluster-dir', required=True)
  parser.add_argument('--teacher-dir', required=True)
  parser.add_argument('--output-dir', required=True)
  parser.add_argument('--epochs', type=int, default=50)
  parser.add_argument('--batch-size', type=int, default=256)
  parser.add_argument('--learning-rate', type=float, default=3e-4)
  parser.add_argument('--seed', type=int, default=41)
  parser.add_argument('--device', default=None)
  args = parser.parse_args()
  print(train_bc(**vars(args)), flush=True)


if __name__ == '__main__':
  main()
