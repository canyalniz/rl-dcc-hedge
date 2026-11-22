#!/usr/bin/env bash
set -euo pipefail

# PLACEHOLDER: point this at the bin/ directory of the Python environment you
# installed requirements.txt into (see README.md, "Setup").
PYTHON_ENV_BIN="/path/to/your/python/env/bin"
export PATH="$PYTHON_ENV_BIN:$PATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

DEFAULT_SWEEP_CONFIG="configs/sweep_config.yaml"
DEFAULT_MAX_RUNS=""

usage() {
  echo "Usage: $0 [SWEEP_CONFIG_PATH] [MAX_RUNS]"
  echo "Default SWEEP_CONFIG_PATH: $DEFAULT_SWEEP_CONFIG"
  echo "Default MAX_RUNS: unlimited"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 2 ]]; then
  usage >&2
  exit 1
fi

SWEEP_CONFIG=${1:-$DEFAULT_SWEEP_CONFIG}
MAX_RUNS=${2:-$DEFAULT_MAX_RUNS}

if [[ ! -f "$SWEEP_CONFIG" ]]; then
  echo "Sweep config not found: $SWEEP_CONFIG" >&2
  exit 1
fi

# Initialize the sweep and capture the agent target from wandb output.
WANDB_DIR=${WANDB_DIR:-results/wandb}
export WANDB_DIR
mkdir -p "$WANDB_DIR"

SWEEP_OUTPUT=$(wandb sweep "$SWEEP_CONFIG" 2>&1)
AGENT_TARGET=$(echo "$SWEEP_OUTPUT" | sed -nE 's/.*wandb agent ([^ ]+).*/\1/p' | tail -n 1)

if [[ -z "$AGENT_TARGET" ]]; then
  SWEEP_ID=$(echo "$SWEEP_OUTPUT" | awk -F': ' '/Created sweep with ID/ {print $2}' | tail -n 1)
  if [[ -n "$SWEEP_ID" && -n "${WANDB_ENTITY:-}" && -n "${WANDB_PROJECT:-}" ]]; then
    AGENT_TARGET="${WANDB_ENTITY}/${WANDB_PROJECT}/${SWEEP_ID}"
  fi
fi

if [[ -z "$AGENT_TARGET" ]]; then
  echo "Failed to parse sweep target from wandb output:" >&2
  echo "$SWEEP_OUTPUT" >&2
  echo "Expected a line like: 'wandb agent ENTITY/PROJECT/SWEEP_ID'" >&2
  exit 1
fi

# Run the sweep agent with optional max run count.
if [[ -n "$MAX_RUNS" ]]; then
  wandb agent --count "$MAX_RUNS" "$AGENT_TARGET"
else
  wandb agent "$AGENT_TARGET"
fi