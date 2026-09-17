#!/usr/bin/env bash
# Run INSIDE curry, normally via docker exec -d. SSH can safely disconnect.
set -euo pipefail

project=/workspace/rl_training
cd "$project"
exec 9> "$project/.lite3_flat_training.lock"
if ! flock -n 9; then
    printf 'A Lite3 flat training launcher already holds the lock. Not starting another.\n' >&2
    exit 1
fi
if pgrep -af '^/isaac-sim/kit/python/bin/python3 .*scripts/reinforcement_learning/rsl_rl/train.py'; then
    printf 'Another training process is running. Not starting a duplicate.\n' >&2
    exit 1
fi

num_envs=${LITE3_NUM_ENVS:-4096}
iterations=${LITE3_MAX_ITERATIONS:-30000}
resume_checkpoint=${LITE3_RESUME_CHECKPOINT:-}
run_name=${LITE3_RUN_NAME:-flat_footwork_v2}
if [[ ! "$num_envs" =~ ^[1-9][0-9]*$ || ! "$iterations" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Environment count and iteration count must be positive integers.\n' >&2
    exit 1
fi
if [[ ! "$run_name" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    printf 'Run name must contain only letters, numbers, underscores or hyphens.\n' >&2
    exit 1
fi

# RSL-RL interprets num_learning_iterations as ADDITIONAL updates after loading.
# Here LITE3_MAX_ITERATIONS means the total target displayed in the training log.
resume_iteration=0
remaining_iterations=$iterations
if [[ -n "$resume_checkpoint" ]]; then
    if [[ "$resume_checkpoint" != /* || ! -f "$resume_checkpoint" ]]; then
        printf 'Resume checkpoint must be an existing absolute path: %s\n' "$resume_checkpoint" >&2
        exit 1
    fi
    resume_checkpoint=$(readlink -f "$resume_checkpoint")
    # train.py resolves a run directory and a checkpoint filename pattern;
    # unlike play.py, it does not accept an absolute path in --checkpoint.
    if [[ "$(dirname "$(dirname "$resume_checkpoint")")" != "$project/logs/rsl_rl/deeprobotics_lite3_flat" ]]; then
        printf 'Resume checkpoint must belong to the Lite3 flat experiment directory.\n' >&2
        exit 1
    fi
    # Only pass a trusted checkpoint produced by this project's own training.
    resume_iteration=$(/isaac-sim/python.sh -c \
        'import sys, torch; checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False); print(int(checkpoint["iter"]))' \
        "$resume_checkpoint")
    if [[ ! "$resume_iteration" =~ ^[0-9]+$ ]] || (( resume_iteration >= iterations )); then
        printf 'Checkpoint iteration must be below the requested total: %s / %s\n' "$resume_iteration" "$iterations" >&2
        exit 1
    fi
    remaining_iterations=$((iterations - resume_iteration))
fi

run_dir="$project/launches/lite3_footwork_v2/$(date -u +%Y%m%d_%H%M%S)_UTC_$$"
mkdir -p "$run_dir"
# Only replace the launcher's own symlink, never an unrelated regular file/directory.
latest="$project/flat_footwork_v2_latest"
if [[ -e "$latest" && ! -L "$latest" ]]; then
    printf 'Refusing to replace non-symlink: %s\n' "$latest" >&2
    exit 1
fi
ln -sfn "$run_dir" "$latest"
printf '%s\n' "$$" > "$run_dir/launcher.pid"
date -u --iso-8601=seconds > "$run_dir/started_utc.txt"
trap 'result=$?; printf "%s\n" "$result" > "$run_dir/exit_code.txt"; date -u --iso-8601=seconds > "$run_dir/ended_utc.txt"' EXIT
printf 'total_target=%s\nresume_iteration=%s\nadditional_iterations=%s\n' \
    "$iterations" "$resume_iteration" "$remaining_iterations" > "$run_dir/iteration_budget.txt"
if [[ -n "$resume_checkpoint" ]]; then
    printf '%s\n' "$resume_checkpoint" > "$run_dir/resume_checkpoint.txt"
    sha256sum "$resume_checkpoint" > "$run_dir/resume_checkpoint.sha256"
fi

# Snapshot includes untracked new reward code, which a plain git diff omits.
tar --exclude='__pycache__' --exclude='*.pyc' -czf "$run_dir/training_source.tar.gz" \
    source/rl_training scripts/reinforcement_learning scripts/tools/start_lite3_flat_footwork_v2.sh
sha256sum \
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/flat_env_cfg.py \
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/lite3_flat_gait.py \
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/commands.py \
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py \
    > "$run_dir/source.sha256"

command=(/isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/train.py
    --task=Flat-Deeprobotics-Lite3-v0
    --num_envs="$num_envs"
    --max_iterations="$remaining_iterations"
    --run_name="$run_name"
    --seed=42
    --logger=tensorboard
    --headless)
if [[ -n "$resume_checkpoint" ]]; then
    command+=(--resume --load_run="$(basename "$(dirname "$resume_checkpoint")")" --checkpoint="$(basename "$resume_checkpoint")")
fi
printf '%q ' "${command[@]}" > "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"
export PYTHONUNBUFFERED=1
"${command[@]}" > "$run_dir/console.log" 2>&1
