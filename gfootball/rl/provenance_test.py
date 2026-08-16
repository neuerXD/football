# coding=utf-8
"""Tests for experiment provenance metadata."""

import os

from gfootball.rl import provenance


def test_experiment_metadata_has_reproducibility_fields(tmp_path):
  previous = os.getcwd()
  os.chdir(str(tmp_path))
  try:
    metadata = provenance.experiment_metadata()
  finally:
    os.chdir(previous)
  assert metadata['created_at_utc']
  assert metadata['git_commit']
  assert isinstance(metadata['git_dirty'], bool)
  assert metadata['python']
  assert metadata['numpy']
