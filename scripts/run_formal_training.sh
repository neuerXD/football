#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/data/zx/football}"
case "$project_root" in
  /data/*) ;;
  *)
    echo "PROJECT_ROOT must be under /data on the training host" >&2
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

pipeline_dir="$project_root/artifacts/orchestration/formal_training"
mkdir -p "$pipeline_dir"
echo "$$" > "$pipeline_dir/pipeline.pid"

wait_for_artifact() {
  local artifact="$1"
  local pid_file="$2"
  local label="$3"
  while [[ ! -s "$artifact" ]]; do
    if [[ ! -s "$pid_file" ]] || ! kill -0 "$(<"$pid_file")" 2>/dev/null; then
      echo "$label stopped before producing $artifact" >&2
      return 1
    fi
    echo "$(date --iso-8601=seconds) waiting for $label"
    sleep 30
  done
}

teacher_root="$project_root/artifacts/teacher/qwen25_14b_seed31"
for shard in shard0 shard1; do
  wait_for_artifact \
    "$teacher_root/$shard/teacher_manifest.json" \
    "$teacher_root/$shard/teacher.pid" \
    "teacher $shard"
done

merged_dir="$teacher_root/merged"
python -m gfootball.rl.merge_teacher \
  --shard-dirs "$teacher_root/shard0" "$teacher_root/shard1" \
  --output-dir "$merged_dir" \
  > "$pipeline_dir/merge_teacher.log" 2>&1

bc_dir="$project_root/artifacts/train/bc"
mkdir -p "$bc_dir"
if [[ ! -s "$bc_dir/bc_checkpoint.pt" ]]; then
  python -m gfootball.rl.bc \
    --cluster-dir "$project_root/artifacts/data/clusters_3k_seed23" \
    --teacher-dir "$merged_dir" \
    --output-dir "$bc_dir" --epochs 50 --seed 41 --device cuda \
    > "$bc_dir/train.log" 2>&1
fi
test -s "$bc_dir/bc_checkpoint.pt"
test -s "$bc_dir/bc_metrics.json"

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

declare -a job_outputs=()
jobs=(
  "bc_ppo 11"
  "bc_ppo 22"
  "bc_ppo 33"
  "ablation_no_curriculum 11"
  "ablation_no_curriculum 22"
  "ablation_no_curriculum 33"
  "ablation_no_potential 11"
  "ablation_no_potential 22"
  "ablation_no_potential 33"
)

job_index=0
for specification in "${jobs[@]}"; do
  read -r experiment seed <<< "$specification"
  output_dir="$project_root/artifacts/train/$experiment/seed$seed"
  if [[ -s "$output_dir/final.pt" ]]; then
    echo "skipping completed $experiment seed $seed"
    continue
  fi
  if module_pid_is_running "$output_dir/train.pid" gfootball.rl.ppo; then
    echo "monitoring active $experiment seed $seed"
    job_outputs+=("$output_dir")
    continue
  fi
  wait_for_capacity
  mkdir -p "$output_dir"
  extra_args=()
  case "$experiment" in
    bc_ppo) ;;
    ablation_no_curriculum) extra_args+=(--no-curriculum) ;;
    ablation_no_potential) extra_args+=(--no-potential) ;;
    *) echo "unknown experiment $experiment" >&2; exit 2 ;;
  esac
  gpu=$((job_index % 2))
  echo "$(date --iso-8601=seconds) starting $experiment seed $seed on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python -m gfootball.rl.ppo \
    --output-dir "$output_dir" \
    --total-steps 250000 --num-envs "$envs_per_run" --rollout-steps 256 \
    --optimization-seed "$seed" --checkpoint-interval 25000 \
    --device cuda --start-method spawn \
    --bc-checkpoint "$bc_dir/bc_checkpoint.pt" "${extra_args[@]}" \
    > "$output_dir/train.log" 2>&1 &
  pid=$!
  echo "$pid" > "$output_dir/train.pid"
  job_outputs+=("$output_dir")
  job_index=$((job_index + 1))
  sleep 2
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "$experiment seed $seed failed during startup" >&2
    exit 1
  fi
done

while ((${#job_outputs[@]})); do
  remaining=()
  for output_dir in "${job_outputs[@]}"; do
    if [[ -s "$output_dir/final.pt" ]]; then
      echo "$(date --iso-8601=seconds) completed $output_dir"
      continue
    fi
    if ! module_pid_is_running "$output_dir/train.pid" gfootball.rl.ppo; then
      echo "training stopped without a final checkpoint: $output_dir" >&2
      exit 1
    fi
    remaining+=("$output_dir")
  done
  job_outputs=("${remaining[@]}")
  if ((${#job_outputs[@]})); then
    echo "$(date --iso-8601=seconds) waiting for ${#job_outputs[@]} training jobs"
    sleep 60
  fi
done
date --iso-8601=seconds > "$pipeline_dir/completed_at.txt"
echo "formal BC and PPO training complete"
