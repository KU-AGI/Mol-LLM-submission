import torch
from transformers import DataCollatorForSeq2Seq
from transformers.tokenization_utils_base import BatchEncoding
from torch_geometric.data import Data
from torch_geometric.loader.dataloader import Collater as GraphCollater

import numpy as np

from collections import Counter

import selfies as sf

import rdkit.Chem as Chem
import re
import copy
import hashlib
import json

def pool_size(args, available: int) -> int:
    size = getattr(args, "molpo_reject_pool_size", None)
    if type(size) is not int or not 1 <= size <= available:
        raise ValueError(f"molpo_reject_pool_size must be in [1,{available}], got {size!r}")
    return size


def occurrence_row(row, index: int):
    index = int(index)
    if index < 0:
        raise ValueError("dataset index must be nonnegative")
    return dict(row, _pair_index=index)


def select_slots(args, batch, epoch: int, available: int):
    size = pool_size(args, available)
    result = []
    for row in batch:
        index = row.get("_pair_index")
        if type(index) is not int or index < 0:
            raise ValueError("pair selection requires a dataset index")
        key = json.dumps(
            [
                "occurrence_cycle_v1",
                index,
                *[
                    str(row.get(f, ""))
                    for f in ("input_mol_string", "prompt_text", "target_text", "molpo_dup_idx")
                ],
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        offset = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
        result.append((offset + int(epoch)) % size)
    return result


class IndexedRows:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        index = int(index)
        return occurrence_row(self.dataset[index], index)



def molpo_batch_division(args) -> int:
    layout = getattr(args, "molpo_batch_layout", None)
    if layout is not None:
        raise ValueError(f"unknown molpo_batch_layout {layout!r}")
    return 3 if getattr(args, "sft_reject_weight", None) else 2

def _diff_mask_core(read, write, n, skip=()):

    plan = []
    skip = set(skip)
    for i in range(n):
        if i in skip:
            continue
        (cv, ci), (mv, mi), (rv, _) = read(0, i), read(1, i), read(2, i)


        assert cv.numel() == mv.numel(), (
            f"instance {i}: block1 has {cv.numel()} scored tokens, block2 has {mv.numel()} -- "
            "blocks 1 and 2 must carry the same answer"
        )
        k = min(cv.numel(), rv.numel())
        if k == 0:
            continue
        eq = cv[:k] == rv[:k]
        if bool(eq.any()):
            plan.append((i, ci[:k][eq], mi[:k][eq]))
    n_masked = 0
    for i, c_hit, m_hit in plan:
        write(0, i, c_hit)
        write(1, i, m_hit)
        n_masked += int(c_hit.numel())
    return n_masked

def apply_molpo_diff_mask(molpo_labels, seq_lens, skip=()):
    row = molpo_labels[0]
    bounds, off = [], 0
    for L in seq_lens:
        bounds.append((off, off + int(L)))
        off += int(L)
    n_seg = len(bounds)
    assert n_seg % 3 == 0, f"{n_seg} packed segments is not a multiple of 3"
    n = n_seg // 3

    def read(block, i):
        a, b = bounds[block * n + i]
        idx = (row[a:b] != -100).nonzero(as_tuple=True)[0] + a
        return row[idx], idx

    def write(block, i, idx):
        row[idx] = -100

    return _diff_mask_core(read, write, n, skip)

def apply_molpo_diff_mask_padded(molpo_labels, skip=()):
    n_rows = molpo_labels.shape[0]
    assert n_rows % 3 == 0, f"{n_rows} padded rows is not a multiple of 3"
    n = n_rows // 3

    def read(block, i):
        r = molpo_labels[block * n + i]
        idx = (r != -100).nonzero(as_tuple=True)[0]
        return r[idx], idx

    def write(block, i, idx):
        molpo_labels[block * n + i, idx] = -100


    return _diff_mask_core(read, write, n, skip)
CLASSIFICATION_BENCHMARKS = [
    "smol-property_prediction-bbbp",
    "smol-property_prediction-clintox",
    "smol-property_prediction-hiv",
    "smol-property_prediction-sider",
    "bace",
    "tox21",
    "toxcast",

    "drugood-lbap-core-ic50-scaffold",
    "drugood-lbap-core-ec50-scaffold",
    "drugood-lbap-core-ki-scaffold",
    "drugood-lbap-core-potency-scaffold",


    "drugood-lbap-core-ic50-scaffold-iid",
    "drugood-lbap-core-potency-scaffold-iid",
    "drugood-lbap-core-ki-scaffold-iid",
    "drugood-lbap-core-ec50-scaffold-iid",


    "tdc-hia_hou",
    "tdc-pgp_broccatelli",
    "tdc-bioavailability_ma",
    "tdc-cyp2d6_veith",
    "tdc-cyp3a4_veith",
    "tdc-cyp2c9_veith",
    "tdc-cyp2d6_substrate_carbonmangels",
    "tdc-cyp3a4_substrate_carbonmangels",
    "tdc-cyp2c9_substrate_carbonmangels",
    "tdc-herg",
    "tdc-ames",
    "tdc-dili",
]
REGRESSION_BENCHMARKS = ["smol-property_prediction-esol", "smol-property_prediction-lipo", "qm9_homo", "qm9_lumo", "qm9_homo_lumo_gap", "qm9_dipole_moment", "qm9_isotropic_polarizability", "qm9_electronic_spatial_extent", "qm9_zero_point_vibrational_energy", "qm9_heat_capacity_298K", "qm9_internal_energy_298K", "qm9_enthalpy_298K", "qm9_free_energy_298K", "alchemy_homo", "alchemy_lumo", "alchemy_homo_lumo_gap", "aqsol-logS", "pcqm_homo_lumo_gap", 'tdc-caco2_wang', 'tdc-ppbr_az', 'tdc-vdss_lombardo', 'tdc-half_life_obach', 'tdc-clearance_microsome_az', 'tdc-clearance_hepatocyte_az', 'tdc-ld50_zhu', 'moleculeace-CHEMBL1862_Ki', 'moleculeace-CHEMBL1871_Ki', 'moleculeace-CHEMBL2034_Ki', 'moleculeace-CHEMBL2047_EC50', 'moleculeace-CHEMBL204_Ki', 'moleculeace-CHEMBL2147_Ki', 'moleculeace-CHEMBL214_Ki', 'moleculeace-CHEMBL218_EC50', 'moleculeace-CHEMBL219_Ki', 'moleculeace-CHEMBL228_Ki', 'moleculeace-CHEMBL231_Ki', 'moleculeace-CHEMBL233_Ki', 'moleculeace-CHEMBL234_Ki', 'moleculeace-CHEMBL235_EC50', 'moleculeace-CHEMBL236_Ki', 'moleculeace-CHEMBL237_EC50', 'moleculeace-CHEMBL237_Ki', 'moleculeace-CHEMBL238_Ki', 'moleculeace-CHEMBL239_EC50', 'moleculeace-CHEMBL244_Ki', 'moleculeace-CHEMBL262_Ki', 'moleculeace-CHEMBL264_Ki', 'moleculeace-CHEMBL2835_Ki', 'moleculeace-CHEMBL287_Ki', 'moleculeace-CHEMBL2971_Ki', 'moleculeace-CHEMBL3979_EC50', 'moleculeace-CHEMBL4005_Ki', 'moleculeace-CHEMBL4203_Ki', 'moleculeace-CHEMBL4616_EC50', 'moleculeace-CHEMBL4792_Ki']
REACTION_BENCHMARKS = [
    "forward_reaction_prediction",
    "smol-forward_synthesis",
    "retrosynthesis",
    "smol-retrosynthesis",
    "reagent_prediction",
    "presto-forward_reaction_prediction",
    "presto-retrosynthesis",
    "presto-reagent_prediction",
    "orderly-forward_reaction_prediction",
    "orderly-retrosynthesis",
    "orderly-reagent_prediction",
]
TEXT2MOL_BENCHMARKS = [
    "chebi-20-text2mol",
    "smol-molecule_generation",
]
MOL2TEXT_BENCHMARKS = [
    "chebi-20-mol2text",
    "smol-molecule_captioning",
]
NAME_CONVERSION_BENCHMARKS = [
    "smol-name_conversion-i2s",
    "smol-name_conversion-i2f",
    "smol-name_conversion-s2f",
    "smol-name_conversion-s2i",
]


tasks = (
    CLASSIFICATION_BENCHMARKS
    + REGRESSION_BENCHMARKS
    + REACTION_BENCHMARKS
    + TEXT2MOL_BENCHMARKS
    + MOL2TEXT_BENCHMARKS
    + NAME_CONVERSION_BENCHMARKS
)

def _load_multi_assay_subtasks():
    import importlib.util
    import os

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moleculenet_assays.py")
    if not os.path.exists(path):
        return {}
    spec = importlib.util.spec_from_file_location("_moleculenet_assays", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod.MULTI_ASSAY_SUBTASKS)
MULTI_ASSAY_SUBTASKS = _load_multi_assay_subtasks()
MULTI_ASSAY_PAIRS = [
    f"{base}/{assay}"
    for base in sorted(MULTI_ASSAY_SUBTASKS)
    for assay in MULTI_ASSAY_SUBTASKS[base]
]
tasks = tasks + MULTI_ASSAY_PAIRS
AUXILIARY_TASKS = [
    "graph2selfies",
]
tasks = tasks + AUXILIARY_TASKS
PUBCHEM3DMOIT_TRANSFER_TASKS = {
    'pubchem3dmoit-transfer-caption': 'caption',
    'pubchem3dmoit-transfer-logp': 'regression',
}
tasks = tasks + list(PUBCHEM3DMOIT_TRANSFER_TASKS)
MOL2TEXT_BENCHMARKS = MOL2TEXT_BENCHMARKS + ['pubchem3dmoit-transfer-caption']
REGRESSION_BENCHMARKS = REGRESSION_BENCHMARKS + ['pubchem3dmoit-transfer-logp']

def base_task(task):
    return task.split("/")[0]
input_mol_string_pattern = re.compile("<SELFIES>.*?</SELFIES>")
graph_sequence = re.compile("<GRAPH>[<mol>]+?</GRAPH>")
MOL_REPR_BOTH = "graph+string"
_MOL_REPR_ALIASES = {"string+graph": MOL_REPR_BOTH}
_selfies_then_graph = re.compile(r"(<SELFIES>.*?</SELFIES>)(<GRAPH>(?:<mol>)+</GRAPH>)", re.DOTALL)

def _graph_first(prompt):
    return _selfies_then_graph.sub(r"\2\1", prompt)

def canonical_mol_representation(value):
    return _MOL_REPR_ALIASES.get(value, value)

def task2id(task):
    # task name to task id
    task2id = {k: i for i, k in enumerate(tasks)}
    return task2id[task]


def id2task(task_id):
    # task id to task name
    id2task = {i: k for i, k in enumerate(tasks)}
    return id2task[task_id]
_DUMMY_NODE = [5, 0, 4, 5, 3, 0, 2, 0, 0]

def _dummy_graph_data():
    return Data(
        x=torch.tensor([_DUMMY_NODE, _DUMMY_NODE], dtype=torch.int64),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.int64),
        edge_attr=torch.zeros((2, 3), dtype=torch.int64),
    )

def _needs_second_graph(sample) -> bool:
    return str(sample.get("prompt_text", "")).count("<GRAPH>") > 1

class DataCollator(DataCollatorForSeq2Seq):
    def __init__(
        self,
        tokenizer,
        padding=True,
        max_length=512,
        pad_to_multiple_of=None,
        return_tensors=None,
        train=True,
        args=None,
        truncate=True,
    ):
        super().__init__(
            tokenizer,
            padding=padding,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        self.train = train
        self.max_length = max_length


        policy = getattr(args, 'sequence_length_policy', 'legacy')
        if policy not in ('legacy', 'full_preserve'):
            raise ValueError(f'Unknown sequence_length_policy={policy!r}')
        self.full_sequence_preserve = policy == 'full_preserve'
        if self.full_sequence_preserve:


            context = getattr(args, 'sequence_context_limit', None)
            if context is None or int(context) < 2:
                raise ValueError('full_preserve requires an explicit sequence_context_limit')
            self.truncate = False
            self._cap = int(context)
        else:
            self.truncate = truncate
            self._cap = max_length if truncate else 1 << 30
        self.tokenizer.padding_side = "left"
        self.mol_representation = canonical_mol_representation(args.mol_representation)


        self.apply_sequence_packing = self.train and getattr(
            args, "apply_sequence_packing", False
        )

        self._probe_pad_to_max = bool(getattr(args, "probe_pad_instances_to_max", False)) and self.train
        if self.full_sequence_preserve and self._probe_pad_to_max:
            raise ValueError('full_preserve forbids artificial padding to the context limit; '
                             'probe real packed batches instead')

        self.apply_molpo = args.train_molpo if self.train else args.eval_molpo

        self.projector_type = args.projector_type
        self.current_epoch = args.current_epoch
        self.args = args

        if self.mol_representation in [MOL_REPR_BOTH, "graph_only"]:
            self.graph_collator = GraphCollater([], [])

    def select_mol_representation(self, prompt_text, mol_representation=MOL_REPR_BOTH):
        mol_representation = canonical_mol_representation(mol_representation)
        if mol_representation == MOL_REPR_BOTH:


            return [_graph_first(p) for p in prompt_text]
        elif mol_representation == "string_only":
            string_only_prompt_text = [graph_sequence.sub("", p) for p in prompt_text]
            return string_only_prompt_text
        elif mol_representation == "graph_only":
            graph_only_prompt_text = [
                input_mol_string_pattern.sub("", p) for p in prompt_text
            ]
            return graph_only_prompt_text
        else:
            raise ValueError(
                f"mol_representation should be one of ['{MOL_REPR_BOTH}', 'string_only', "
                f"'graph_only'] (the historical spelling 'string+graph' is accepted and normalised)"
            )

    def enumerate_selfies(
        self,
        origin_selfies,
    ):
        origin_smiles = sf.decoder(origin_selfies)

        isomericSmiles = bool(self.args.isomericSmiles)
        canonical = bool(self.args.canonical)
        allHsExplicit = bool(self.args.allHsExplicit)

        processed_smiles = Chem.MolToSmiles(
            Chem.MolFromSmiles(origin_smiles),
            isomericSmiles=isomericSmiles,
            canonical=canonical,
            doRandom=not canonical,
            allHsExplicit=allHsExplicit,
            allBondsExplicit=False,
            kekuleSmiles=False,
        )
        processed_selfies = sf.encoder(processed_smiles)
        return processed_selfies

    def pack_features(self, full_input_ids, prompt_length, target_input_ids, tasks,
                      list_graphs=None, list_additional_graphs=None,
                      molpo_mask_ids=None):


        _share = (
            getattr(self.args, "molpo_share_prefix", False)
            and molpo_mask_ids is not None
            and molpo_batch_division(self.args) == 3
            and len(full_input_ids) % 3 == 0
        )
        _n = len(full_input_ids) // 3 if _share else 0


        _nolabel = set(getattr(self, "_nolabel_rows", ()) or ())
        _nl3 = set()
        if _nolabel:
            if molpo_mask_ids is None or molpo_batch_division(self.args) != 3 or len(full_input_ids) % 3:
                raise ValueError("label-free reject slots need the packed MolPO three-block layout")
            _nl3 = {2 * (len(full_input_ids) // 3) + i for i in _nolabel}
        _emit = []
        if _share:
            for blk in range(3):
                for i in range(_n):
                    j = blk * _n + i
                    ids = list(full_input_ids[j])
                    seg_labels = [-100] * len(ids)
                    if prompt_length[j] < self._cap:
                        tgt = target_input_ids[j]
                        n_label = min(len(tgt), self._cap - prompt_length[j])
                        if n_label > 0:
                            seg_labels[len(ids) - n_label :] = tgt[:n_label]
                    if blk == 2 and i in _nolabel:

                        seg_labels = [-100] * len(seg_labels)
                    if blk == 2:
                        P = prompt_length[_n + i]
                        if prompt_length[j] != P:
                            raise ValueError(
                                f"instance {i}: S2 prompt is {P} tokens and S3 is {prompt_length[j]}. "
                                "Sharing a prefix requires them to BE the same prefix."
                            )
                        if list(full_input_ids[_n + i][:P]) != ids[:P]:
                            raise ValueError(
                                f"instance {i}: S2 and S3 prompts differ token-for-token, so the "
                                "shared prefix would answer a different question for one of them."
                            )


                        if P < 1:
                            raise ValueError(f"instance {i}: prompt length {P} leaves no carry token")
                        _emit.append((ids[P - 1 :], P - 1, seg_labels[P - 1 :]))
                    else:
                        _emit.append((ids, 0, seg_labels))
        else:
            for j, ids in enumerate(full_input_ids):
                seg_labels = [-100] * len(ids)
                if prompt_length[j] < self._cap and j not in _nl3:
                    tgt = target_input_ids[j]
                    n_label = min(len(tgt), self._cap - prompt_length[j])
                    if n_label > 0:
                        seg_labels[len(ids) - n_label :] = tgt[:n_label]
                _emit.append((ids, 0, seg_labels))

        input_ids, position_ids, labels, seq_lens = [], [], [], []
        molpo_seg = []
        for i, (ids, pos0, seg_labels) in enumerate(_emit):


            if self._probe_pad_to_max and len(ids) < self._cap:
                pad = [self.tokenizer.pad_token_id] * (self._cap - len(ids))
                ids = list(ids) + pad
                seg_labels = list(seg_labels) + [-100] * len(pad)
            seg_len = len(ids)
            input_ids.extend(ids)
            position_ids.extend(range(pos0, pos0 + seg_len))
            seq_lens.append(seg_len)
            molpo_seg.extend([i] * seg_len)
            labels.extend(seg_labels)

        features = BatchEncoding(
            {
                "input_ids": torch.tensor([input_ids], dtype=torch.int64),
                "position_ids": torch.tensor([position_ids], dtype=torch.int64),
                "labels": torch.tensor([labels], dtype=torch.int64),


                "tasks": torch.tensor(tasks, dtype=torch.int16),
                "seq_lens": torch.tensor(seq_lens, dtype=torch.int64),
            }
        )
        if _share:


            features["molpo_seg"] = torch.tensor([molpo_seg], dtype=torch.int64)
            n_seg = len(_emit)
            kv_src = [-1] * n_seg
            kv_pre = [0] * n_seg
            for i in range(_n):
                kv_src[2 * _n + i] = _n + i


                kv_pre[2 * _n + i] = prompt_length[_n + i] - 1
            features["molpo_kv_src"] = torch.tensor(kv_src, dtype=torch.int64)
            features["molpo_kv_pre"] = torch.tensor(kv_pre, dtype=torch.int64)

        features["labels"] = features["labels"].masked_fill(
            features["labels"] == self.tokenizer.pad_token_id, -100
        )

        pos = features["position_ids"][0]
        if _share:


            _seg_t = features["molpo_seg"][0]
            _changes = int((_seg_t[1:] != _seg_t[:-1]).sum())
            assert _changes == len(_emit) - 1, (
                f"molpo_seg has {_changes} boundaries for {len(_emit)} segments (seq_lens={seq_lens})"
            )
            assert int(_seg_t.max()) + 1 == len(_emit), "molpo_seg does not cover every segment"


            assert int((pos == 0).sum()) == 2 * _n, (
                f"{int((pos == 0).sum())} position resets for {2 * _n} graph-carrying segments"
            )
        else:
            n_boundaries = int((pos[1:] - pos[:-1] < 0).sum())
            assert n_boundaries == len(_emit) - 1, (
                f"packed row has {n_boundaries} negative position steps but "
                f"{len(_emit) - 1} instance boundaries -- an instance of length 1 "
                f"makes the pack indistinguishable from a single sequence (seq_lens={seq_lens})"
            )
        assert features["input_ids"].size(1) == sum(seq_lens)


        if molpo_mask_ids is not None:
            molpo_labels = features["labels"].clone()
            for molpo_mask_id in molpo_mask_ids:
                molpo_labels = molpo_labels.masked_fill(
                    molpo_labels == molpo_mask_id, -100
                )


            _want = bool(getattr(self.args, "molpo_label_diff_mask", True))
            _dv = molpo_batch_division(self.args)
            _applied = _want and _dv == 3
            if _applied:
                if _nolabel:
                    apply_molpo_diff_mask(molpo_labels, seq_lens, skip=_nolabel)
                else:
                    apply_molpo_diff_mask(molpo_labels, seq_lens)
            if not getattr(DataCollator, "_diff_mask_announced", False):
                DataCollator._diff_mask_announced = True
                if _applied:
                    _why = ("S2 graph margin only: scored where y+ and the S3 donor y- differ; S4 has its own "
                            "answer_correction_scoring_policy line")
                elif not _want:
                    _why = "DISABLED by molpo_label_diff_mask=false"
                else:
                    _why = (f"not applicable: molpo_batch_division={_dv}, so there is no y- block "
                            "to diff against (set sft_reject_weight for the three-block layout)")
                print(f"[molpo] label diff mask {'ON' if _applied else 'OFF'} ({_why})", flush=True)


            _s3off = sorted(set(getattr(self, "_s3off_rows", ()) or ()))
            if _s3off:
                if not _share or _n * 3 != len(seq_lens):
                    raise NotImplementedError("molpo_s3_off_column needs the packed shared-prefix layout")
                _b = [0]
                for L in seq_lens:
                    _b.append(_b[-1] + int(L))
                for i in _s3off:
                    lo, hi = _b[2 * _n + i], _b[2 * _n + i + 1]
                    molpo_labels[0, lo:hi] = -100
                    features["labels"][0, lo:hi] = -100


            _s23 = sorted(set(getattr(self, "_s23off_rows", ()) or ()))
            if _s23:
                if not _share or _n * 3 != len(seq_lens):
                    raise NotImplementedError("molpo_s23_off_column needs the packed shared-prefix layout")
                _b2 = [0]
                for L in seq_lens:
                    _b2.append(_b2[-1] + int(L))
                for i in _s23:
                    for blk in (0, 1):
                        lo, hi = _b2[blk * _n + i], _b2[blk * _n + i + 1]
                        molpo_labels[0, lo:hi] = -100
                        if blk == 1:
                            features["labels"][0, lo:hi] = -100
                    lo, hi = _b2[2 * _n + i], _b2[2 * _n + i + 1]
                    molpo_labels[0, lo:hi] = -100
                    features["labels"][0, lo:hi] = -100

            features["molpo_labels"] = molpo_labels
        elif getattr(self, "_s3off_rows", None) or getattr(self, "_s23off_rows", None):
            raise NotImplementedError("molpo_s3_off_column needs MolPO labels (molpo_mask_ids)")


        import os as _os
        _dp = _os.environ.get("MOLPO_DUMP_BATCH")
        if _dp and not _os.path.exists(_dp):
            import json as _json
            _d = {
                "share": bool(_share),
                "input_ids": features["input_ids"][0].tolist(),
                "labels": features["labels"][0].tolist(),
                "position_ids": features["position_ids"][0].tolist(),
                "seq_lens": seq_lens,
                "prompt_length": list(prompt_length),
            }
            if _share:
                _d["molpo_seg"] = features["molpo_seg"][0].tolist()
                _d["kv_src"] = features["molpo_kv_src"].tolist()
                _d["kv_pre"] = features["molpo_kv_pre"].tolist()
            open(_dp, "w").write(_json.dumps(_d))

        if list_graphs is not None:
            if _share:


                list_graphs = list_graphs[: 2 * _n]
                list_additional_graphs = list_additional_graphs[: 2 * _n]
            features["graphs"] = self.graph_collator(list_graphs)
            features["additional_graphs"] = self.graph_collator(list_additional_graphs)


            features["is_mol_token"] = (
                features["input_ids"] == self.tokenizer.mol_token_id
            )
        return features

    def __call__(self, batch, return_tensors=None):


        return self._collate(batch, return_tensors)

    def _collate(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors


        self._nolabel_rows = []


        self._s3off_rows = []
        self._s23off_rows = []

        tasks = [task2id(sample["task"]) for sample in batch]  # task id
        task_names = [id2task(task) for task in tasks]


        src_row_idx = (
            [int(sample["src_row_idx"]) for sample in batch]
            if batch and "src_row_idx" in batch[0]
            else None
        )
        prompt_text = [sample["prompt_text"] for sample in batch]
        target_text = [sample["target_text"] for sample in batch]
        input_mol_strings = [sample["input_mol_string"] for sample in batch]
        list_selfies = [
            i.replace("<SELFIES> ", "").replace(" </SELFIES>", "")
            for i in input_mol_strings
        ]
        list_graphs = [
            Data(
                x=torch.tensor(sample["x"], dtype=torch.int64),
                edge_index=torch.tensor(sample["edge_index"], dtype=torch.int64),
                edge_attr=torch.tensor(sample["edge_attr"], dtype=torch.int64),
            )
            for sample in batch
        ]
        # for reagent prediction
        list_additional_graphs = [
            Data(
                x=torch.tensor(sample["additional_x"], dtype=torch.int64),
                edge_index=torch.tensor(
                    sample["additional_edge_index"], dtype=torch.int64
                ),
                edge_attr=torch.tensor(
                    sample["additional_edge_attr"], dtype=torch.int64
                ),
            )
            if _needs_second_graph(sample)
            else _dummy_graph_data()
            for sample in batch
        ]

        prompt_text = self.select_mol_representation(
            prompt_text, mol_representation=self.mol_representation
        )

        if not self.train and self.args.eval_modality_util in [
            "string",
            "graph",
        ]:
            shuffled_idx = []
            # shuffle the selfies_idx, guarantee that the selfies_idx is not in order
            for i in range(len(list_selfies)):
                idxs = np.random.choice(
                    range(len(list_selfies)), size=2, replace=False
                ).tolist()
                if i in idxs:
                    idxs.remove(i)
                shuffled_idx.append(idxs[0])

            if self.args.eval_modality_util == "string":
                processed_selfies = [list_selfies[i] for i in shuffled_idx]
                for i in range(len(prompt_text)):
                    assert (
                        list_selfies[i] in prompt_text[i]
                    ), f"{list_selfies[i]} not in {prompt_text[i]}"
                    prompt_text[i] = prompt_text[i].replace(
                        list_selfies[i], processed_selfies[i]
                    )

            if self.args.eval_modality_util == "graph":
                list_graphs = [list_graphs[i] for i in shuffled_idx]
                list_additional_graphs = [
                    list_additional_graphs[i] for i in shuffled_idx
                ]

        if self.apply_molpo:
            if self.train:


                n_rej = getattr(self, "_n_reject_cardinals", None)
                if n_rej is None:
                    n_rej = len([k for k in batch[0] if k.endswith("-th_rejected_x")]) or 1
                    self._n_reject_cardinals = n_rej
                n_use = pool_size(self.args, n_rej)
                self._row_cardinals = select_slots(self.args, batch, self.current_epoch, n_rej)
                self.reject_cardinal = None if n_use > 1 else self._row_cardinals[0]
            else:
                self.reject_cardinal = 0
                self._row_cardinals = [0] * len(batch)


            _eval_label_free = (not self.train) and bool(getattr(self.args, "eval_molpo_label_free_rejects", False))
            if getattr(self.args, "molpo_optional_reject_labels", False) or _eval_label_free:
                self._nolabel_rows = [
                    i for i, (c, sample) in enumerate(zip(self._row_cardinals, batch))
                    if f"{c}-th_rejected_target_text" in sample
                    and not str(sample[f"{c}-th_rejected_target_text"] or "").strip()
                ]
                if self._nolabel_rows:


                    if molpo_batch_division(self.args) != 3:
                        raise ValueError(
                            "label-free reject slots need the three-block layout (sft_reject_weight set): in the "
                            "two-block layout there is no S3 to skip and nothing else may be masked")
                    if not self.apply_sequence_packing and not _eval_label_free:
                        raise NotImplementedError(
                            "label-free reject slots are implemented for the packed TRAIN layout only; the "
                            "padded/eval layout would score the placeholder block-3 target as S3")


            if getattr(self.args, "molpo_s3_off_column", False):
                _miss = next((c for c, s in zip(self._row_cardinals, batch)
                              if f"{c}-th_rejected_s3_off" not in s), None)
                if _miss is not None:
                    raise KeyError(f"molpo_s3_off_column is set but this tag has no '{_miss}-th_rejected_s3_off'")
                self._s3off_rows = [
                    i for i, (c, sample) in enumerate(zip(self._row_cardinals, batch))
                    if int(sample[f"{c}-th_rejected_s3_off"] or 0)
                ]
                if self._s3off_rows:
                    if molpo_batch_division(self.args) != 3:
                        raise ValueError("molpo_s3_off_column needs the three-block layout (sft_reject_weight set)")
                    if not self.apply_sequence_packing or not getattr(self.args, "molpo_share_prefix", False):
                        raise NotImplementedError(
                            "molpo_s3_off_column is implemented for the packed shared-prefix TRAIN layout only")


            if getattr(self.args, "molpo_s23_off_column", False):
                _miss23 = next((c for c, s in zip(self._row_cardinals, batch)
                                if f"{c}-th_rejected_s23_off" not in s), None)
                if _miss23 is not None:
                    raise KeyError(f"molpo_s23_off_column is set but this tag has no '{_miss23}-th_rejected_s23_off'")
                self._s23off_rows = [
                    i for i, (c, sample) in enumerate(zip(self._row_cardinals, batch))
                    if int(sample[f"{c}-th_rejected_s23_off"] or 0)
                ]
                if self._s23off_rows:
                    if molpo_batch_division(self.args) != 3:
                        raise ValueError("molpo_s23_off_column needs the three-block layout (sft_reject_weight set)")
                    if not self.apply_sequence_packing or not getattr(self.args, "molpo_share_prefix", False):
                        raise NotImplementedError(
                            "molpo_s23_off_column is implemented for the packed shared-prefix TRAIN layout only")


            _div = molpo_batch_division(self.args)
            prompt_text = prompt_text * _div
            if _div == 3:


                _missing = next(
                    (c for c, s in zip(self._row_cardinals, batch)
                     if f"{c}-th_rejected_target_text" not in s), None
                )
                if _missing is not None:
                    raise KeyError(
                        f"sft_reject_weight is set but this tag has no "
                        f"'{_missing}-th_rejected_target_text'. Derive it from "
                        f"'{{k}}-th_rejected_src_idx' before running v6."
                    )
                if self._nolabel_rows:
                    _nl = set(self._nolabel_rows)
                    target_text = target_text * 2 + [
                        (sample["target_text"] if i in _nl else sample[f"{c}-th_rejected_target_text"])
                        for i, (c, sample) in enumerate(zip(self._row_cardinals, batch))
                    ]
                else:
                    target_text = target_text * 2 + [
                        sample[f"{c}-th_rejected_target_text"]
                        for c, sample in zip(self._row_cardinals, batch)
                    ]
            else:
                target_text = target_text * _div
            tasks = tasks * _div
            task_names = task_names * _div


            _n = len(batch)
            assert len(prompt_text) == _div * _n, f"prompt block count {len(prompt_text)} != {_div}x{_n}"
            assert len(target_text) == _div * _n, f"target block count {len(target_text)} != {_div}x{_n}"
            assert len(tasks) == _div * _n, f"task block count {len(tasks)} != {_div}x{_n}"
            assert prompt_text[:_n] == prompt_text[_n : 2 * _n], (
                "blocks 1 and 2 must share the prompt -- the margin compares two graphs for ONE question"
            )
            assert target_text[:_n] == target_text[_n : 2 * _n], (
                "blocks 1 and 2 must share the target -- the margin is defined on the chosen answer"
            )
            if _div == 3:
                assert prompt_text[_n : 2 * _n] == prompt_text[2 * _n :], (
                    "blocks 2 and 3 must share the prompt -- they differ only in the answer"
                )


                _lab = [i for i in range(_n) if i not in set(self._nolabel_rows)]
                _same = sum(1 for i in _lab if target_text[i] == target_text[2 * _n + i])
                assert (not _lab and self._nolabel_rows) or _same < len(_lab), (
                    f"all {_n} rows have y- identical to y+. The reject is selected by ANSWER "
                    "distance, so this means the derived target column is the chosen answer copied, "
                    "and block 3 would train the reject graph toward the wrong label."
                )

            if "graph" in self.mol_representation:
                list_rejected_graphs = [
                    Data(
                        x=torch.tensor(
                            sample[f"{c}-th_rejected_x"],
                            dtype=torch.int64,
                        ),
                        edge_index=torch.tensor(
                            sample[f"{c}-th_rejected_edge_index"],
                            dtype=torch.int64,
                        ),
                        edge_attr=torch.tensor(
                            sample[f"{c}-th_rejected_edge_attr"],
                            dtype=torch.int64,
                        ),
                    )
                    for c, sample in zip(self._row_cardinals, batch)
                ]


                _has_reject_add = "0-th_additional_rejected_x" in batch[0]
                _add_keys = [
                    (f"{c}-th_additional_rejected_" if _has_reject_add else "additional_")
                    for c in self._row_cardinals
                ]
                # for reagent prediction
                list_rejected_additional_graphs = [
                    Data(
                        x=torch.tensor(
                            sample[_add_key + "x"],
                            dtype=torch.int64,
                        ),
                        edge_index=torch.tensor(
                            sample[_add_key + "edge_index"],
                            dtype=torch.int64,
                        ),
                        edge_attr=torch.tensor(
                            sample[_add_key + "edge_attr"],
                            dtype=torch.int64,
                        ),
                    )
                    if _needs_second_graph(sample)
                    else _dummy_graph_data()
                    for _add_key, sample in zip(_add_keys, batch)
                ]


                list_graphs = list_graphs + list_rejected_graphs * (_div - 1)
                assert len(list_graphs) == _div * _n, (
                    f"graph block count {len(list_graphs)} != {_div}x{_n}"
                )


                if _div == 3:
                    assert all(
                        list_graphs[_n + i] is list_graphs[2 * _n + i] for i in range(_n)
                    ), "blocks 2 and 3 do not hold the same reject graph"


                list_additional_graphs = (
                    list_additional_graphs + list_rejected_additional_graphs * (_div - 1)
                )
                assert len(list_additional_graphs) == len(list_graphs), (
                    f"additional-graph block count {len(list_additional_graphs)} != "
                    f"{len(list_graphs)}; the two slots must agree or graph injection cannot cat them"
                )

        # address <mol> token in prompt_text, for the case of using graph modality
        if self.projector_type == "mlp" and "graph" in self.mol_representation:
            for i in range(len(prompt_text)):
                if task_names[i] in ["reagent_prediction"]:
                    num_nodes_in_graph = list_graphs[i].x.size(0)
                    num_nodes_mol = "<mol>" * num_nodes_in_graph


                    mol_tokens_pattern = re.compile(r"(<mol>)+(?=</GRAPH>)(?=[\s\S]*\|>>\|)")
                    assert mol_tokens_pattern.search(
                        prompt_text[i]
                    ), f"{prompt_text[i]}"
                    prompt_text[i] = mol_tokens_pattern.sub(
                        num_nodes_mol, prompt_text[i]
                    )

                    num_additional_nodes_in_graph = list_additional_graphs[i].x.size(0)
                    num_additional_nodes_mol = "<mol>" * num_additional_nodes_in_graph
                    additional_mol_tokens_pattern = re.compile(
                        r"(?<=\|>>\|<GRAPH>)(<mol>)+"
                    )
                    assert additional_mol_tokens_pattern.search(
                        prompt_text[i]
                    ), f"{prompt_text[i]}"
                    prompt_text[i] = additional_mol_tokens_pattern.sub(
                        num_additional_nodes_mol, prompt_text[i]
                    )
                elif task_names[i] in TEXT2MOL_BENCHMARKS:
                    # there is no input <mol> token
                    pass
                else:
                    num_nodes_in_graph = list_graphs[i].x.size(0)
                    num_nodes_mol = "<mol>" * num_nodes_in_graph
                    mol_tokens_pattern = re.compile("(<mol>)+")
                    assert mol_tokens_pattern.search(
                        prompt_text[i]
                    ), f"{prompt_text[i]}"
                    prompt_text[i] = mol_tokens_pattern.sub(
                        num_nodes_mol, prompt_text[i]
                    )

        self.tokenizer.padding_side = "left"
        prompt_tokenized = self.tokenizer(
            prompt_text,
            truncation=self.truncate,
            max_length=self.max_length if self.truncate else None,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )
        target_tokenized = self.tokenizer(
            target_text,
            truncation=self.truncate,
            max_length=self.max_length if self.truncate else None,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )

        full_input_ids = [
            p + t
            for p, t in zip(
                prompt_tokenized["input_ids"], target_tokenized["input_ids"]
            )
        ]
        full_attention_mask = [
            p + t
            for p, t in zip(
                prompt_tokenized["attention_mask"], target_tokenized["attention_mask"]
            )
        ]

        prompt_length = [len(p) for p in prompt_tokenized["input_ids"]]
        if self.full_sequence_preserve:
            for i, ids in enumerate(full_input_ids):
                if len(ids) > self._cap:
                    raise ValueError(f'task={task_names[i]} prompt={prompt_length[i]} '
                                     f'target={len(target_tokenized["input_ids"][i])} '
                                     f'exceeds model context {self._cap}; no tokens dropped')
        full_input_ids = [f_ids[: self._cap] for f_ids in full_input_ids]
        full_attention_mask = [f_ids[: self._cap] for f_ids in full_attention_mask]

        if self.apply_sequence_packing:
            has_graph = "graph" in self.mol_representation
            packed = self.pack_features(
                full_input_ids=full_input_ids,
                prompt_length=prompt_length,
                target_input_ids=target_tokenized["input_ids"],
                tasks=tasks,
                list_graphs=list_graphs if has_graph else None,
                list_additional_graphs=list_additional_graphs if has_graph else None,
                molpo_mask_ids=(
                    self.tokenizer.molpo_mask_ids if self.apply_molpo else None
                ),
            )


            if self.apply_molpo:


                tcols = [f"{c}-th_rejected_margin" for c in self._row_cardinals]
                if tcols[0] in batch[0]:
                    packed["molpo_target"] = torch.tensor(
                        [
                            float(sample[tc])
                            for tc, sample in zip(tcols, batch)
                        ],
                        dtype=torch.float,
                    )


                if "molpo_ok" in batch[0]:
                    packed["molpo_ok"] = torch.tensor(
                        [bool(sample["molpo_ok"]) for sample in batch], dtype=torch.bool
                    )
            from model.molpo import attach_answer_batch
            return attach_answer_batch(self, packed, batch)

        self.tokenizer.padding_side = "left"
        features = self.tokenizer.pad(
            {"input_ids": full_input_ids, "attention_mask": full_attention_mask},
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        if not self.train:
            prompt_features = self.tokenizer.pad(
                {
                    "input_ids": [p for p in prompt_tokenized["input_ids"]],
                    "attention_mask": [p for p in prompt_tokenized["attention_mask"]],
                },
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )

            features["prompt_input_ids"] = prompt_features.input_ids  # ['input_ids']
            features["prompt_attention_mask"] = (
                prompt_features.attention_mask
            )  # ['attention_mask']

            self.tokenizer.padding_side = "right"
            gen_features = self.tokenizer.pad(
                {
                    "input_ids": [t for t in target_tokenized["input_ids"]],
                },
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )
            gen_features.input_ids = gen_features.input_ids.masked_fill(
                gen_features.input_ids == self.tokenizer.pad_token_id, -100
            )
            features["gen_labels"] = gen_features.input_ids

            input_mol_strings_tokenized = self.tokenizer(
                input_mol_strings,
                truncation=False,
                max_length=self.max_length,
                padding=True,
                return_tensors=return_tensors,
                add_special_tokens=False,
            )

            features["input_mol_strings"] = input_mol_strings_tokenized.input_ids

        if src_row_idx is not None:
            features["src_row_idx"] = torch.tensor(src_row_idx, dtype=torch.int64)

        labels_ids = torch.full_like(features["input_ids"], self.tokenizer.pad_token_id)
        for i, target in enumerate(target_tokenized["input_ids"]):
            label = target
            if prompt_length[i] >= self._cap:
                continue
            else:
                len_label = min(len(label), self._cap - prompt_length[i])
                if len_label == 0:


                    continue
                labels_ids[i, -len_label:] = torch.tensor(
                    label[:len_label], dtype=torch.int64
                )

        labels_ids = labels_ids.masked_fill(
            labels_ids == self.tokenizer.pad_token_id, -100
        )
        features["labels"] = labels_ids
        if self.apply_molpo:
            molpo_labels_ids = labels_ids.clone()
            for molpo_mask_id in self.tokenizer.molpo_mask_ids:
                molpo_labels_ids = molpo_labels_ids.masked_fill(
                    molpo_labels_ids == molpo_mask_id, -100
                )


            _want = bool(getattr(self.args, "molpo_label_diff_mask", True))
            _dv = molpo_batch_division(self.args)
            _applied = _want and _dv == 3
            if _applied:
                _nl_eval = list(getattr(self, "_nolabel_rows", ()) or ())
                if _nl_eval:
                    apply_molpo_diff_mask_padded(molpo_labels_ids, skip=_nl_eval)
                else:
                    apply_molpo_diff_mask_padded(molpo_labels_ids)
            if not getattr(DataCollator, "_diff_mask_announced_eval", False):
                DataCollator._diff_mask_announced_eval = True
                _why = ("S2 graph margin only: scored where y+ and y- differ" if _applied
                        else "DISABLED by molpo_label_diff_mask=false" if not _want
                        else f"not applicable: molpo_batch_division={_dv}, no y- block to diff against")
                print(f"[molpo] label diff mask {'ON' if _applied else 'OFF'} "
                      f"(eval/padded path; {_why})", flush=True)
            features["molpo_labels"] = molpo_labels_ids

        assert (
            features.input_ids.size(1) <= self._cap
        ), f"features.input_ids.size(1)={features.input_ids.size(1)} > cap={self._cap}"
        assert (
            features.labels.size(1) <= self._cap
        ), f"features.labels.size(1)={features.labels.size(1)} > cap={self._cap}"

        features["tasks"] = torch.tensor(tasks, dtype=torch.int16)
        if "graph" in self.mol_representation:
            graphs = self.graph_collator(list_graphs)
            additional_graphs = self.graph_collator(list_additional_graphs)
            features["graphs"] = graphs
            features["additional_graphs"] = additional_graphs
            features["is_mol_token"] = (
                features["input_ids"] == self.tokenizer.mol_token_id
            )
            if not self.train:
                features["prompt_is_mol_token"] = (
                    torch.tensor(features["prompt_input_ids"])
                    == self.tokenizer.mol_token_id
                )

        from model.molpo import attach_answer_batch
        return attach_answer_batch(self, features, batch)

def random_noise_selfies(selfies, tokenizer, sl_noise_ratio=0.3):
    selfies_ids = tokenizer.encode(selfies, add_special_tokens=False)
    total_selfies_token_ids = tokenizer.selfies_token_ids
    num_ids_to_replace = int(sl_noise_ratio * len(selfies_ids))
    replacing_random_ids = np.random.choice(
        total_selfies_token_ids, num_ids_to_replace, replace=True
    )

    # replace selfies_ids with randomly selected total_selfies_token_ids as many as num_ids_to_replace
    position_to_replace = np.random.choice(
        len(selfies_ids), num_ids_to_replace, replace=False
    )
    noised_selfies_ids = copy.deepcopy(selfies_ids)
    for i, replance_idx in enumerate(position_to_replace):
        noised_selfies_ids[replance_idx] = replacing_random_ids[i]

    noised_selfies = tokenizer.decode(
        noised_selfies_ids, skip_special_tokens=True
    ).replace(" ", "")
    return noised_selfies
