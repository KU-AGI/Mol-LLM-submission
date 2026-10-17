# Introduction

This repository accompanies the revision of "Mol-LLM: Multimodal Generalist Molecular LLM with Improved Graph Utilization". This release provides code for training and evaluating Mol-LLM and constructing preference pairs, training and evaluation datasets, and the model checkpoints required to reproduce the results reported in the paper.

- **MolPO training:** graph and output preference losses, paired-output supervision, and their integration into training ([Methods and Implementation](#methods-and-implementation)).
- **Dataset curation:** shared and dataset-specific selection rules, TRAIN-source validation, and reconstruction from released pair selections ([Dataset Curation](#dataset-curation)).
- **Training and evaluation data:** molecular inputs, task outputs, graphs and prepared preference pairs ([Dataset Curation](#dataset-curation)).
- **Model evaluation:** Mol-LLM, Mol-LLM w/o MolPO and Mol-LLM w/o Graph using the released checkpoints and evaluation data ([Quick start](#quick-start)).

# Models and Datasets

Model checkpoints and prepared training/evaluation data are available on [Google Drive](https://drive.google.com/drive/folders/1kNy1MhYoJKOr1QW4lgN-49pJydPxVRPJ?usp=sharing). The release includes Mol-LLM, Mol-LLM w/o MolPO and Mol-LLM w/o Graph for seeds 42, 1234 and 4042.

# Quick Start

## Environment

The Docker environment is defined in `Dockerfile` and `Makefile`.
Set `REPO_PATH`, `CACHE_PATH`, `DATA_PATH` and `IMAGE_NAME_TAG` in `Makefile` for the local repository, model cache, data directory and Docker image, respectively.

```bash
make build-image
make init-container
```

The container mounts the repository at `/text-mol` and `DATA_PATH` at `/data`. Set `RELEASE` to the release directory within the container.

## Execution

Preserve the downloaded `data/` and `checkpoints/` directories. Run from the repository root in the project environment.

```bash
export RELEASE=/absolute/path/to/reviewer-release-20261017
bash Mol-LLM/bashes/running_scripts_for_revision/evaluate.sh --model mol-llm --seed 42 --tasks all --gpus 0
```

| Evaluation option | Accepted values | Default |
|---|---|---|
| `--model` | `mol-llm`, `mol-llm-wo-molpo`, `mol-llm-wo-graph` | `mol-llm` |
| `--seed` | `42`, `1234`, `4042` | `42` |
| `--tasks` | `all` or comma-separated task names | `moleculeace-CHEMBL1862_Ki` |
| `--gpus` | Comma-separated GPU indices | `0` |

Results are saved under `outputs/<model>-s<seed>-evaluation/`.

| Script in `Mol-LLM/bashes/running_scripts_for_revision/` | Operation | Output |
|---|---|---|
| `evaluate.sh` | Evaluate a released checkpoint | Predictions and task metrics |
| `inspect_data.sh` | Read a MoleculeACE example and its paired graphs | Terminal output |
| `verify_files.sh` | Verify downloaded file checksums | Terminal output |
| `reconstruct_data.sh` | Reconstruct rows from fixed pair selections | `rebuilt-train/` |
| `train.sh` | Train with the MolPO configuration | Training outputs and checkpoints |

Run each script with `bash Mol-LLM/bashes/running_scripts_for_revision/<script>`. Training requires the separate initialization checkpoint specified below.

# Repository Structure

```text
Mol-LLM/
├── bashes/running_scripts_for_revision/  Execution scripts
├── configs/                             Training and evaluation settings
├── data/
│   ├── builders/hard_negative/           Shared preference-pair construction
│   │   └── recipes/                     Dataset-specific selection rules
│   └── storage.py                       Released dataset loading
├── model/                               Model architecture and MolPO losses
├── data_module.py                       Dataset loading and batching
├── data_utils.py                        Model input construction
└── stage3.py                            Training and evaluation entry point
```

# Methods and Implementation

## Code Entry Points

| Component | Entry point | Source |
|---|---|---|
| Training and combined loss | `training_step`, `get_total_molpo_loss` | [blip2_stage3.py](Mol-LLM/model/blip2_stage3.py) |
| Output preference and normalization | `model_answer_correction`, `MolPOWindowReducer` | [molpo.py](Mol-LLM/model/molpo.py) |
| Model inputs | `DataCollator` | [data_utils.py](Mol-LLM/data_utils.py) |
| Common selection and dataset dispatch | `select` | [selection.py](Mol-LLM/data/builders/hard_negative/selection.py) |
| TRAIN validation and pair construction | `TrainPool`, `materialize` | [train_pool.py](Mol-LLM/data/builders/hard_negative/train_pool.py), [materialize.py](Mol-LLM/data/builders/hard_negative/materialize.py) |

## MolPO Training

Each example contains a prompt, the original graph and answer, three paired graphs and answers, and an output-preference target. One paired graph is used for each training example.

```text
for each optimizer step:
    for each accumulated batch:
        prompt, graph, answer = batch.anchor
        reject_graph, paired_answer = select_reject_slot(batch)
        wrong_answer, answer_weight = batch.answer_preference

        S1 = NLL(answer | prompt, graph)
        S2 = graph_preference(answer, graph > reject_graph | prompt)
        S3 = NLL(paired_answer | prompt, reject_graph)
        S4 = answer_weight * answer_preference(answer > wrong_answer | prompt, graph)
        accumulate each term with its own valid-token/example denominator
    loss = S1 + 0.25*S2 + 0.25*S3 + 0.25*S4
    update trainable parameters
```

S2 compares graphs for the same answer. S3 supervises the paired answer under its graph. S4 compares answers under the original graph; its alternative answer may differ from the selected graph's paired answer. All terms retain the original prompt.

MolPO training uses [train.sh](Mol-LLM/bashes/running_scripts_for_revision/train.sh) and [revision_molpo.yaml](Mol-LLM/configs/trainer/revision_molpo.yaml). Training additionally requires `MOLLLM_STAGE2_CHECKPOINT`, which is not included in the release.

## Dataset Curation

| Directory | Contents |
|---|---|
| `data/train/`, `data/test/` | Prepared instances and graph payloads, loaded directly by training and evaluation |
| `data/sources/train/` | Canonical TRAIN instances used for pair construction |
| `data/selections/` | Final selection indices and transformation parameters |

Upsampling repetitions are removed from the shared training partition. The released checkpoints were trained with the original sampling distribution.

```text
pool = load_canonical_TRAIN(source)
for task in tasks:
    assignments = dataset_rule(pool, task, required_intermediate_inputs)
        filter eligible candidates using the task's answer/chemistry rule
        rank and allocate candidates using shared helpers where applicable
        return TRAIN indices or a supported TRAIN-derived transformation
    validate source fingerprint, indices and endpoint boundaries
    materialize graphs and answers from TRAIN into three reject slots
    store answer-preference target and its eligibility weight
```

The common selection function calls each dataset's rule. Pair construction validates TRAIN membership and endpoint consistency before retrieving graphs and answers. Forward-synthesis examples may use stereochemical transformations; description-guided generation uses output preferences only.

| Dataset-specific implementation | Scope |
|---|---|
| [basic.py](Mol-LLM/data/builders/hard_negative/recipes/basic.py) | Binary molecular selection |
| [properties.py](Mol-LLM/data/builders/hard_negative/recipes/properties.py) | BACE, BBBP, ClinTox, SIDER, HIV, QM9 and MoleculeACE |
| [esol.py](Mol-LLM/data/builders/hard_negative/recipes/esol.py), [lipo.py](Mol-LLM/data/builders/hard_negative/recipes/lipo.py), [tdc.py](Mol-LLM/data/builders/hard_negative/recipes/tdc.py) | ESOL, Lipophilicity and TDC |
| [reactions.py](Mol-LLM/data/builders/hard_negative/recipes/reactions.py) | Forward synthesis, retrosynthesis and stereochemistry transformations |
| [reagents.py](Mol-LLM/data/builders/hard_negative/recipes/reagents.py) | Reagent selection |
| [text.py](Mol-LLM/data/builders/hard_negative/recipes/text.py) | Molecule captioning and description-guided generation |

Reconstruct prepared rows from the supplied selections into a new directory:

```bash
bash Mol-LLM/bashes/running_scripts_for_revision/reconstruct_data.sh
```

Use `--output` for a new output directory or `--pairs` with a selection `.jsonl.gz` for one dataset. Reconstruction uses the released fixed selections. Rerunning candidate selection requires additional inputs, including Lipophilicity and TDC model-mining inputs not included in the release.
