# Repository Instructions

## Remote RTX 3090 Storage

- On the remote RTX 3090 host, all project data and generated files must live
  under `/data`. Do not create project data under `/home`, `$HOME`, or `~`.
- Use `/data/zx/football` as the repository root.
- Put the Python environment in `/data/zx/football/.venv`.
- Put datasets, teacher labels, checkpoints, logs, evaluation outputs, plots,
  and reports under `/data/zx/football/artifacts`.
- Put shared caches under `/data/zx/.cache` and temporary files under
  `/data/zx/tmp`.
- Before installing dependencies, downloading models, or running experiments,
  export at least:

  ```bash
  export XDG_CACHE_HOME=/data/zx/.cache
  export PIP_CACHE_DIR=/data/zx/.cache/pip
  export HF_HOME=/data/zx/.cache/huggingface
  export HUGGINGFACE_HUB_CACHE=/data/zx/.cache/huggingface/hub
  export TRANSFORMERS_CACHE=/data/zx/.cache/huggingface/transformers
  export TORCH_HOME=/data/zx/.cache/torch
  export TMPDIR=/data/zx/tmp
  ```

- Check target paths before remote installation or training. Do not rely on a
  tool's default cache or output directory when it may resolve under `/home`.
