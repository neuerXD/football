# coding=utf-8
"""Tests for experiment provenance metadata."""

from gfootball.rl import provenance


def test_experiment_metadata_has_reproducibility_fields():
  metadata = provenance.experiment_metadata()
  assert metadata['created_at_utc']
  assert metadata['git_commit']
  assert isinstance(metadata['git_dirty'], bool)
  assert metadata['python']
  assert metadata['numpy']
