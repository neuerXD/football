# coding=utf-8
"""Small reproducibility helpers for experiment manifests."""

from __future__ import absolute_import

import datetime
import os
import platform
import subprocess
import sys

import numpy as np


def _git(args, cwd):
  try:
    return subprocess.check_output(
        ['git'] + list(args), cwd=cwd, stderr=subprocess.DEVNULL,
        universal_newlines=True).strip()
  except (OSError, subprocess.CalledProcessError):
    return ''


def experiment_metadata(repo_dir=None):
  """Returns JSON-serializable code, runtime, and accelerator metadata."""
  repo_dir = os.path.abspath(repo_dir or os.getcwd())
  status = _git(['status', '--porcelain'], repo_dir)
  result = {
      'created_at_utc': datetime.datetime.now(
          datetime.timezone.utc).isoformat(),
      'git_commit': _git(['rev-parse', 'HEAD'], repo_dir),
      'git_dirty': bool(status),
      'hostname': platform.node(),
      'platform': platform.platform(),
      'python': sys.version.split()[0],
      'numpy': np.__version__,
  }
  try:
    import torch
    result.update({
        'torch': torch.__version__,
        'torch_cuda': torch.version.cuda,
        'cuda_available': bool(torch.cuda.is_available()),
        'cuda_device_count': int(torch.cuda.device_count()),
        'cuda_devices': [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    })
  except ImportError:
    result['torch'] = None
  return result
