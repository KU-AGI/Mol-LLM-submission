#!/usr/bin/env bash
set -euo pipefail
: "${RELEASE:?Set RELEASE to the downloaded data and checkpoints directory}"
RELEASE=$(cd "$RELEASE" && pwd)
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$REPO_ROOT"
export MOLLLM_GINE_CHECKPOINT="$RELEASE/checkpoints/common/gine-init.ckpt"
export MOLLLM_TOKENGT_CHECKPOINT="$RELEASE/checkpoints/common/tokengt-init.ckpt"
export MOLLLM_TRAIN="$RELEASE/data/train"
export MOLLLM_VALIDATION="$RELEASE/data/test"
export MOLLLM_TEST="$RELEASE/data/test"
export MOLLLM_DATASET_CACHE=${MOLLLM_DATASET_CACHE:-"$REPO_ROOT/.cache/datasets"}
export MOLLLM_OUTPUT_DIR=${MOLLLM_OUTPUT_DIR:-"$REPO_ROOT/outputs"}
export WANDB_MODE=${WANDB_MODE:-offline}
