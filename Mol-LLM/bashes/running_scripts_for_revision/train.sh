#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/environment.sh"
: "${MOLLLM_STAGE2_CHECKPOINT:?Set MOLLLM_STAGE2_CHECKPOINT to the Stage-2 checkpoint}"
exec "${PYTHON:-python}" Mol-LLM/stage3.py --config-name revision "$@"
