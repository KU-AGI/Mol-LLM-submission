#!/usr/bin/env bash
set -euo pipefail
: "${RELEASE:?Set RELEASE to the downloaded data and checkpoints directory}"
RELEASE=$(cd "$RELEASE" && pwd)
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export PYTHONPATH="$REPO_ROOT/Mol-LLM${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m data.builders.hard_negative \
  --source "$RELEASE/data/sources" --pairs "$RELEASE/data/selections" \
  --output "$REPO_ROOT/rebuilt-train" --workers 8 "$@"
