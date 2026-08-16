#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/data/zx/football}"
case "$project_root" in
  /data/*) ;;
  *)
    echo "PROJECT_ROOT must be under /data on the evaluation host" >&2
    exit 2
    ;;
esac

cd "$project_root"
export PATH="$project_root/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$project_root/.venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XDG_CACHE_HOME=/data/zx/.cache
export PIP_CACHE_DIR=/data/zx/.cache/pip
export HF_HOME=/data/zx/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/data/zx/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/data/zx/.cache/huggingface/transformers
export TORCH_HOME=/data/zx/.cache/torch
export MPLCONFIGDIR=/data/zx/.cache/matplotlib
export TMPDIR=/data/zx/tmp
export CUDA_CACHE_PATH=/data/zx/.cache/nv
export SDL_AUDIODRIVER=dummy
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR" artifacts/orchestration

pipeline_dir="$project_root/artifacts/orchestration/formal_evaluation"
mkdir -p "$pipeline_dir"
echo "$$" > "$pipeline_dir/pipeline.pid"

training_outputs=(
  artifacts/train/scratch_ppo/seed11
  artifacts/train/scratch_ppo/seed22
  artifacts/train/scratch_ppo/seed33
  artifacts/train/bc_ppo/seed11
  artifacts/train/bc_ppo/seed22
  artifacts/train/bc_ppo/seed33
  artifacts/train/ablation_no_curriculum/seed11
  artifacts/train/ablation_no_curriculum/seed22
  artifacts/train/ablation_no_curriculum/seed33
  artifacts/train/ablation_no_potential/seed11
  artifacts/train/ablation_no_potential/seed22
  artifacts/train/ablation_no_potential/seed33
)

while true; do
  missing=0
  for output_dir in "${training_outputs[@]}"; do
    if [[ ! -s "$output_dir/final.pt" ]]; then
      missing=$((missing + 1))
    fi
  done
  if ((missing == 0)); then
    break
  fi
  echo "$(date --iso-8601=seconds) waiting for $missing final checkpoints"
  sleep 60
done
test -s artifacts/train/bc/bc_checkpoint.pt

max_env_workers="${MAX_ENV_WORKERS:-56}"
envs_per_run=8

active_parent_count() {
  local module="$1"
  pgrep -fc "[p]ython -m $module" || true
}

module_pid_is_running() {
  local pid_file="$1"
  local module="$2"
  local pid
  [[ -s "$pid_file" ]] || return 1
  pid="$(<"$pid_file")"
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" | grep -Fq "python -m $module"
}

wait_for_capacity() {
  while true; do
    local ppo_count
    local eval_count
    local used_workers
    ppo_count="$(active_parent_count gfootball.rl.ppo)"
    eval_count="$(active_parent_count gfootball.rl.evaluate)"
    used_workers=$(((ppo_count + eval_count) * envs_per_run))
    if ((used_workers + envs_per_run <= max_env_workers)); then
      return
    fi
    echo "$(date --iso-8601=seconds) waiting for capacity: ${used_workers}/${max_env_workers} workers"
    sleep 30
  done
}

declare -a eval_outputs=()

queue_evaluation() {
  local output_dir="$1"
  local policy_name="$2"
  shift 2
  if [[ -s "$output_dir/evaluation_manifest.json" &&
        -s "$output_dir/episodes.jsonl" ]]; then
    echo "skipping completed evaluation $output_dir"
    return
  fi
  if module_pid_is_running \
      "$output_dir/evaluate.pid" gfootball.rl.evaluate; then
    echo "monitoring active evaluation $output_dir"
    eval_outputs+=("$output_dir")
    return
  fi
  wait_for_capacity
  mkdir -p "$output_dir"
  echo "$(date --iso-8601=seconds) starting evaluation $output_dir"
  command=(
    python -m gfootball.rl.evaluate
    --output-dir "$output_dir" --policy-name "$policy_name"
    --num-envs "$envs_per_run" --macro-steps 100 --game-duration 3600
    --start-method spawn "$@"
  )
  "${command[@]}" > "$output_dir/evaluate.log" 2>&1 &
  pid=$!
  echo "$pid" > "$output_dir/evaluate.pid"
  eval_outputs+=("$output_dir")
  sleep 2
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "evaluation failed during startup: $output_dir" >&2
    exit 1
  fi
}

queue_checkpoint_curve() {
  local experiment="$1"
  local policy_name="$2"
  for seed in 11 22 33; do
    train_dir="artifacts/train/$experiment/seed$seed"
    mapfile -t checkpoints < <(
      find "$train_dir" -maxdepth 1 -name 'checkpoint_*.pt' -type f | sort)
    if ((${#checkpoints[@]} > 1)); then
      checkpoints=("${checkpoints[@]:0:${#checkpoints[@]}-1}")
    else
      checkpoints=()
    fi
    checkpoints+=("$train_dir/final.pt")
    for checkpoint in "${checkpoints[@]}"; do
      if [[ "$checkpoint" == */final.pt ]]; then
        point=final
      else
        point="$(basename "$checkpoint" .pt)"
      fi
      queue_evaluation \
        "artifacts/eval/builtin/$policy_name/seed$seed/$point" \
        "$policy_name" --policy checkpoint --checkpoint "$checkpoint" \
        --num-seeds 50 --difficulties 0.6 0.8 --device cpu
    done
  done
}

queue_evaluation artifacts/eval/builtin/bc/formal bc \
  --policy checkpoint --checkpoint artifacts/train/bc/bc_checkpoint.pt \
  --num-seeds 50 --difficulties 0.6 0.8 --device cpu
queue_evaluation artifacts/eval/builtin/bc_ppo/initial_bc bc_ppo \
  --policy checkpoint --checkpoint artifacts/train/bc/bc_checkpoint.pt \
  --num-seeds 50 --difficulties 0.6 0.8 --device cpu

queue_checkpoint_curve scratch_ppo scratch_ppo
queue_checkpoint_curve bc_ppo bc_ppo

for experiment in ablation_no_curriculum ablation_no_potential; do
  for seed in 11 22 33; do
    queue_evaluation \
      "artifacts/eval/builtin/$experiment/seed$seed/final" \
      "$experiment" --policy checkpoint \
      --checkpoint "artifacts/train/$experiment/seed$seed/final.pt" \
      --num-seeds 50 --difficulties 0.6 0.8 --device cpu
  done
done

queue_evaluation artifacts/eval/builtin/zero_shot_llm/formal zero_shot_llm \
  --policy llm --model-path .deps/models/Qwen2.5-14B-Instruct \
  --quantization 4bit --llm-samples 1 --llm-temperature 0.2 \
  --num-seeds 50 --difficulties 0.6 0.8

for experiment in scratch_ppo bc_ppo; do
  for seed in 11 22 33; do
    queue_evaluation \
      "artifacts/eval/tizero/$experiment/seed$seed/final" \
      "${experiment}_tizero" --policy checkpoint \
      --checkpoint "artifacts/train/$experiment/seed$seed/final.pt" \
      --num-seeds 50 --difficulties 0.6 --device cpu \
      --opponent tizero --tizero-model-dir .deps/tizero
  done
done

while ((${#eval_outputs[@]})); do
  remaining=()
  for output_dir in "${eval_outputs[@]}"; do
    if [[ -s "$output_dir/evaluation_manifest.json" &&
          -s "$output_dir/episodes.jsonl" ]]; then
      echo "$(date --iso-8601=seconds) completed $output_dir"
      continue
    fi
    if ! module_pid_is_running \
        "$output_dir/evaluate.pid" gfootball.rl.evaluate; then
      echo "evaluation stopped without complete outputs: $output_dir" >&2
      exit 1
    fi
    remaining+=("$output_dir")
  done
  eval_outputs=("${remaining[@]}")
  if ((${#eval_outputs[@]})); then
    echo "$(date --iso-8601=seconds) waiting for ${#eval_outputs[@]} evaluations"
    sleep 60
  fi
done

mapfile -t builtin_dirs < <(
  find artifacts/eval/builtin -type f -name evaluation_manifest.json \
    -printf '%h\n' | sort)
report_command=(
  python -m gfootball.rl.report --eval-dirs "${builtin_dirs[@]}"
  --output-dir artifacts/reports/formal_builtin
  --reference scratch_ppo --bootstrap-samples 10000
)
"${report_command[@]}" > "$pipeline_dir/report_builtin.log" 2>&1

mapfile -t tizero_dirs < <(
  find artifacts/eval/tizero -type f -name evaluation_manifest.json \
    -printf '%h\n' | sort)
report_command=(
  python -m gfootball.rl.report --eval-dirs "${tizero_dirs[@]}"
  --output-dir artifacts/reports/formal_tizero
  --reference scratch_ppo_tizero --bootstrap-samples 10000
)
"${report_command[@]}" > "$pipeline_dir/report_tizero.log" 2>&1

date --iso-8601=seconds > "$pipeline_dir/completed_at.txt"
echo "formal evaluation and reports complete"
