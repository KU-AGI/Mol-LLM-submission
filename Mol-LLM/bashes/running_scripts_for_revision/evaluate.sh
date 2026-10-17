#!/usr/bin/env bash
set -euo pipefail
model=mol-llm
seed=42
gpus=0
tasks=moleculeace-CHEMBL1862_Ki
while (($#)); do
  case "$1" in
    --model|--seed|--gpus|--tasks)
      if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
      case "$1" in
        --model) model=$2 ;;
        --seed) seed=$2 ;;
        --gpus) gpus=$2 ;;
        --tasks) tasks=$2 ;;
      esac
      shift 2 ;;
    --help|-h)
      echo "Usage: bash evaluate.sh [--model {mol-llm,mol-llm-wo-molpo,mol-llm-wo-graph}] [--seed {42,1234,4042}] [--gpus 0,1] [--tasks all|TASK,TASK]"
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
case "$model" in
  mol-llm|mol-llm-wo-molpo) representation=graph+string; tune_gnn=true ;;
  mol-llm-wo-graph) representation=string_only; tune_gnn=false ;;
  *) echo "Unknown model: $model" >&2; exit 2 ;;
esac
case "$seed" in
  42|1234|4042) ;;
  *) echo "Unsupported checkpoint seed: $seed" >&2; exit 2 ;;
esac
if [[ ! "$gpus" =~ ^[0-9]+(,[0-9]+)*$ ]]; then echo "Invalid GPU list: $gpus" >&2; exit 2; fi
if [[ "$tasks" == all ]]; then
  task_filter=null
elif [[ "$tasks" =~ ^[A-Za-z0-9_-]+(,[A-Za-z0-9_-]+)*$ ]]; then
  task_filter="[$tasks]"
else
  echo "Invalid task list: $tasks" >&2; exit 2
fi
source "$(dirname "${BASH_SOURCE[0]}")/environment.sh"
exec "${PYTHON:-python}" Mol-LLM/stage3.py --config-name revision \
  mode=test pretrained_ckpt_path=null \
  "ckpt_path=$RELEASE/checkpoints/$model-seed$seed.ckpt" seed="$seed" \
  trainer.use_ema=false "trainer.devices='$gpus'" \
  trainer.mol_representation="$representation" trainer.tune_gnn="$tune_gnn" \
  trainer.apply_sequence_packing=false trainer.molpo_share_prefix=false \
  +trainer.attn_implementation=eager \
  "+trainer.tasks=$task_filter" filename="$model-s$seed-evaluation"
