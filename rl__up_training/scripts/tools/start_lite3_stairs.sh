#!/usr/bin/env bash
set -euo pipefail

cd /workspace/rl_training

exec /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Stairs-Deeprobotics-Lite3-v0 \
  --num_envs="${NUM_ENVS:-256}" \
  --max_iterations="${MAX_ITERATIONS:-15000}" \
  --headless
