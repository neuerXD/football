# coding=utf-8
"""Immutable seed and evaluation protocol for tactical RL experiments."""

TRAIN_ENV_SEEDS = tuple(range(1000, 10000))
EVAL_ENV_SEEDS = tuple(range(20000, 20050))
OPTIMIZATION_SEEDS = (11, 22, 33)
EVAL_DIFFICULTIES = (0.6, 0.8)


def validate_seed(seed, split):
  seed = int(seed)
  if split == 'train':
    valid = seed in TRAIN_ENV_SEEDS
  elif split == 'eval':
    valid = seed in EVAL_ENV_SEEDS
  else:
    raise ValueError('Unknown split: {}'.format(split))
  if not valid:
    raise ValueError('Seed {} is not in the {} split'.format(seed, split))
  return seed


assert not set(TRAIN_ENV_SEEDS).intersection(EVAL_ENV_SEEDS)
