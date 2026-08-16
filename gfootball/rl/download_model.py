# coding=utf-8
"""Downloads a Hugging Face model with resumable parallel transfers."""

from __future__ import absolute_import

import argparse
import json
import os

from huggingface_hub import snapshot_download


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--repo-id', required=True)
  parser.add_argument('--output-dir', required=True)
  parser.add_argument('--max-workers', type=int, default=4)
  args = parser.parse_args()
  output_dir = os.path.abspath(args.output_dir)
  os.makedirs(output_dir, exist_ok=True)
  path = snapshot_download(
      repo_id=args.repo_id,
      local_dir=output_dir,
      max_workers=args.max_workers,
  )
  print(json.dumps({
      'repo_id': args.repo_id,
      'output_dir': path,
  }, sort_keys=True), flush=True)


if __name__ == '__main__':
  main()
