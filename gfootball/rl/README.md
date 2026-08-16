# LLM-Guided Macro Tactical Reinforcement Learning

This package studies whether an LLM tactical prior improves sample efficiency,
final performance, and held-out generalization for high-level football control.
The LLM is an offline teacher; deployed policies are small PyTorch actors.

## Protocol

- Observations are exactly 50 side-invariant macro features.
- Actions are 12 interpretable whole-team tactical templates.
- One macro decision runs up to 100 GRF engine steps.
- Training uses engine seeds `1000..9999`; evaluation only uses `20000..20049`.
- PPO optimization seeds are `11`, `22`, and `33`.
- Formal evaluation crosses 50 seeds, both controlled sides, and built-in AI
  difficulties `0.6` and `0.8`, producing 200 paired scenarios per run.
- TiZero is an additional out-of-distribution opponent.
- Reports aggregate optimization seeds by scenario and use paired bootstrap
  95% confidence intervals.

## Remote Layout

On the RTX 3090 host, keep every environment, cache, model, and result under
`/data`:

```bash
export PROJECT_ROOT=/data/zx/football
export PATH=$PROJECT_ROOT/.venv/bin:$PATH
export LD_LIBRARY_PATH=$PROJECT_ROOT/.venv/lib
export XDG_CACHE_HOME=/data/zx/.cache
export PIP_CACHE_DIR=/data/zx/.cache/pip
export HF_HOME=/data/zx/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/data/zx/.cache/huggingface/hub
export TORCH_HOME=/data/zx/.cache/torch
export MPLCONFIGDIR=/data/zx/.cache/matplotlib
export TMPDIR=/data/zx/tmp
export SDL_AUDIODRIVER=dummy
```

The verified remote stack is Python 3.10, PyTorch 2.5.1+cu121, NumPy 1.26,
CMake 3.31, Boost.Python 1.85, and SDL2 from conda-forge. Teacher inference
uses Transformers 4.48, Accelerate 1.3, and bitsandbytes 0.45.

## Data and Teacher

Collect 50,000 real macro transitions and select 3,000 representatives:

```bash
python -m gfootball.rl.collect_states \
  --output-dir artifacts/data/states_50k_seed17 \
  --num-states 50000 --num-envs 16 --seed 17 --policy mixture

python -m gfootball.rl.cluster_states \
  --input-dir artifacts/data/states_50k_seed17 \
  --output-dir artifacts/data/clusters_3k_seed23 \
  --num-clusters 3000 --seed 23
```

Run one deterministic shard per GPU. Each state receives three stochastic
samples; only a strict majority with mean confidence at least 0.55 is kept.
Progress is atomically checkpointed after every batch, so rerun with `--resume`
after interruption.

```bash
CUDA_VISIBLE_DEVICES=0 python -m gfootball.rl.teacher \
  --cluster-dir artifacts/data/clusters_3k_seed23 \
  --output-dir artifacts/teacher/qwen14b/shard0 \
  --model-path .deps/models/Qwen2.5-14B-Instruct \
  --num-samples 3 --batch-size 4 --seed 31 \
  --shard-index 0 --num-shards 2

CUDA_VISIBLE_DEVICES=1 python -m gfootball.rl.teacher \
  --cluster-dir artifacts/data/clusters_3k_seed23 \
  --output-dir artifacts/teacher/qwen14b/shard1 \
  --model-path .deps/models/Qwen2.5-14B-Instruct \
  --num-samples 3 --batch-size 4 --seed 31 \
  --shard-index 1 --num-shards 2

python -m gfootball.rl.merge_teacher \
  --shard-dirs artifacts/teacher/qwen14b/shard0 \
    artifacts/teacher/qwen14b/shard1 \
  --output-dir artifacts/teacher/qwen14b/merged
```

The merged manifest audits accepted class counts, parsed-response failures,
missing majorities, and low-confidence filtering in addition to verifying that
the shard NPZ and JSONL indices cover every cluster exactly once.

## BC and PPO

Train weighted behavior cloning, then compare scratch and BC-initialized PPO.
BC transfers the actor and its observation normalization; the critic starts
from scratch.

```bash
python -m gfootball.rl.bc \
  --cluster-dir artifacts/data/clusters_3k_seed23 \
  --teacher-dir artifacts/teacher/qwen14b/merged \
  --output-dir artifacts/train/bc --epochs 50 --seed 41

python -m gfootball.rl.ppo \
  --output-dir artifacts/train/scratch/seed11 \
  --total-steps 250000 --num-envs 8 --rollout-steps 256 \
  --optimization-seed 11 --device cuda

python -m gfootball.rl.ppo \
  --output-dir artifacts/train/bc_ppo/seed11 \
  --total-steps 250000 --num-envs 8 --rollout-steps 256 \
  --optimization-seed 11 --device cuda \
  --bc-checkpoint artifacts/train/bc/bc_checkpoint.pt
```

Repeat PPO with seeds `22` and `33`. Run the core ablations with
`--no-curriculum` and `--no-potential`. Every run writes a JSON manifest,
JSONL learning log, periodic checkpoints, RNG state, and final checkpoint.

On the `/data` training host, queue or resume the formal teacher merge, BC run,
three-seed scratch/BC+PPO runs, and three-seed curriculum/potential ablations
with:

```bash
PROJECT_ROOT=/data/zx/football \
  nohup bash scripts/run_formal_training.sh \
  > artifacts/orchestration/formal_training.log 2>&1 &
```

The launcher accounts for active PPO and evaluation processes and keeps at
most 56 GRF environment workers active by default. Override this only with an
explicit `MAX_ENV_WORKERS` after checking host capacity.

## Evaluation

The evaluator supports `fixed`, `random`, `rule`, `llm`, and `checkpoint`
policies under one scenario schedule. Checkpoint inference defaults to CPU so
latency is not dominated by CUDA synchronization for the small MLP.

```bash
python -m gfootball.rl.evaluate \
  --output-dir artifacts/eval/bc_ppo/seed11/final \
  --policy-name bc_ppo --policy checkpoint \
  --checkpoint artifacts/train/bc_ppo/seed11/final.pt \
  --num-seeds 50 --difficulties 0.6 0.8 --num-envs 8

python -m gfootball.rl.evaluate \
  --output-dir artifacts/eval/bc_ppo/seed11/tizero \
  --policy-name bc_ppo_tizero --policy checkpoint \
  --checkpoint artifacts/train/bc_ppo/seed11/final.pt \
  --num-seeds 50 --difficulties 0.6 --num-envs 8 \
  --opponent tizero --tizero-model-dir .deps/tizero
```

Evaluate intermediate checkpoints for sample-efficiency curves, then aggregate
all run directories:

```bash
python -m gfootball.rl.report \
  --eval-dirs artifacts/eval/*/*/* \
  --output-dir artifacts/reports/formal \
  --reference scratch_ppo --bootstrap-samples 10000
```

The report contains W/D/L, win rate, mean goal difference, paired 95% CIs,
steps to a shared target win rate, tactic distribution, and decision latency.

After `scripts/run_formal_training.sh` has been queued, the full held-out
evaluation can also wait in the background. It includes intermediate
scratch/BC+PPO checkpoints, final ablations, zero-shot Qwen, TiZero, and both
reports:

```bash
PROJECT_ROOT=/data/zx/football \
  nohup bash scripts/run_formal_evaluation.sh \
  > artifacts/orchestration/formal_evaluation.log 2>&1 &
```
