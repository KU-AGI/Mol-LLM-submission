"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import torch
import torch.nn as nn
from torch.amp import autocast as autocast
from torch.nn import functional as F
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    LoraConfig,
    TaskType,
    PeftModel,
)
from ogb.utils import smiles2graph
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from torch_geometric.loader.dataloader import Collater
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph
import numpy as np
from lavis.models.blip2_models.blip2 import (
    # Blip2Base,
    disabled_train,
)
from model.blip2 import Blip2Base
from transformers import AutoTokenizer
from transformers import OPTForCausalLM
import model.added_tokens as added_tokens

from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import replace_return_docstrings

from typing import Optional, List, Tuple, Union

# get logging from transformers library
from transformers import logging

logger = logging.get_logger(__name__)


def mask_by_len(input, lens, fill_value=0):
    """
    input: shape = [N, D]
    lens: shape = [N]
    """
    mask = torch.arange(input.shape[1], device=input.device).reshape(1, -1)
    mask = mask < lens.reshape(-1, 1)
    input[mask] = fill_value
    return input


def smiles2data(smiles):
    graph = smiles2graph(smiles)
    x = torch.from_numpy(graph["node_feat"])
    edge_index = torch.from_numpy(
        graph["edge_index"],
    )
    edge_attr = torch.from_numpy(graph["edge_feat"])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


def shuffle_masked_embeddings(
    embeddings: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    embeddings: torch.Tensor of shape (B, L, D)
    mask:       torch.BoolTensor of shape (B, L)

    Returns a new tensor of same shape where, for each batch b,
    the rows embeddings[b, mask[b]] are randomly permuted among those positions.
    """
    if embeddings.dim() != 3 or mask.dim() != 2:
        raise ValueError(
            "Expected embeddings of shape (B, L, D) and mask of shape (B, L)"
        )
    B, L, D = embeddings.shape
    if mask.shape != (B, L):
        raise ValueError(f"mask shape must be {(B, L)}, got {mask.shape}")
    if mask.dtype != torch.bool:
        mask = mask.to(torch.bool)

    shuffled = embeddings.clone()

    for b in range(B):
        pos = torch.nonzero(mask[b], as_tuple=False).squeeze(1)  # shape (n,)
        n = pos.size(0)
        if n > 1:
            perm = torch.randperm(n, device=embeddings.device)
            shuffled[b, pos] = embeddings[b, pos[perm]]

    return shuffled


def split_batch_by_components(batch: Batch) -> Batch:
    num_nodes = batch.num_nodes
    edge_index = batch.edge_index
    edge_attr = batch.edge_attr if hasattr(batch, "edge_attr") else None

    row, col = edge_index.cpu().numpy()
    adj = coo_matrix(
        (np.ones(len(row), dtype=bool), (row, col)), shape=(num_nodes, num_nodes)
    )

    n_components, labels = connected_components(
        csgraph=adj, directed=False, return_labels=True
    )
    labels = torch.from_numpy(labels).to(edge_index.device)

    new_data_list = []
    for comp_id in range(n_components):
        node_mask = labels == comp_id
        nodes = node_mask.nonzero(as_tuple=False).view(-1)

        sub_ei, sub_ea = subgraph(
            nodes.cpu().tolist(),
            edge_index,
            relabel_nodes=True,
            num_nodes=num_nodes,
            edge_attr=edge_attr,
        )
        sub_x = batch.x[nodes]
        sub_batch = batch.batch[nodes]
        orig_gid = torch.unique(sub_batch)
        assert (
            orig_gid.numel() == 1
        ), "한 컴포넌트가 두 개 이상의 그래프에 걸쳐 있습니다."

        data = Data(x=sub_x, edge_index=sub_ei)
        if edge_attr is not None:
            data.edge_attr = sub_ea

        # 노드-레벨 텐서 속성 복사
        for key, attr in batch.items():
            if key in ("x", "edge_index", "edge_attr", "batch"):
                continue
            if torch.is_tensor(attr) and attr.size(0) == num_nodes:
                data[key] = attr[nodes]
            elif torch.is_tensor(attr) and attr.size(0) == batch.num_graphs:
                data[key] = attr[orig_gid]
            else:
                data[key] = attr

        new_data_list.append(data)

    return new_data_list
    # return Batch.from_data_list(new_data_list)


def count_connected_components(
    edge_index: torch.LongTensor, num_nodes: int = None
) -> int:
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1

    parent = list(range(num_nodes))

    # Find with path compression
    def find(u: int) -> int:
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    # Union
    def union(u: int, v: int):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[rv] = ru

    for u, v in edge_index.t().tolist():
        union(u, v)
        union(v, u)

    for i in range(num_nodes):
        parent[i] = find(i)

    return len(set(parent))


import re


class Blip2OPT(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    def __init__(self, args=None):
        super().__init__()
        self.args = args
        bert_name = args.bert_name
        tune_gnn = args.tune_gnn
        num_query_token = args.num_query_token
        cross_attention_freq = args.cross_attention_freq
        tune_llm = args.tune_llm
        peft_dir = args.peft_dir
        llm_model = args.llm_model
        prompt = args.prompt
        self.peft_dir = peft_dir
        gnn_hidden_dim = args.gnn_hidden_dim

        # initialize opt model
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            # llm_model, use_fast=False, padding_side="right"
            llm_model,
            revision=getattr(args, "llm_revision", None),
            use_fast=False,
            padding_side="left",
        )
        self.add_necessary_tokens()

        self.set_llm_model(llm_model)

        if getattr(args, 'sequence_length_policy', 'legacy') == 'full_preserve':
            declared = int(args.sequence_context_limit)
            actual = int(self.llm_model.config.max_position_embeddings)
            if declared > actual:
                raise ValueError(f'sequence_context_limit={declared} exceeds model limit {actual}')

        n_old_vocab = self.llm_model.get_input_embeddings().weight.shape[0]
        self.llm_model.resize_token_embeddings(
            len(self.llm_tokenizer)
        )  # this will cause bug when full fine-tuning the opt model
        self.n_added_tokens = (
            self.llm_model.get_input_embeddings().weight.shape[0] - n_old_vocab
        )


        self.tune_llm = tune_llm
        if tune_llm == "lora":
            if peft_dir:
                self.llm_model = PeftModel.from_pretrained(
                    self.llm_model, peft_dir, is_trainable=True
                )
            else:
                if self.args.peft_config:
                    peft_config = LoraConfig(
                        **LoraConfig.from_json_file(self.args.peft_config)
                    )
                else:
                    peft_config = LoraConfig(
                        target_modules=self.get_lora_target_modules(),
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        r=args.lora_r,
                        lora_alpha=args.lora_alpha,
                        lora_dropout=args.lora_dropout,
                    )
                self.peft_config = peft_config
                self.llm_model = get_peft_model(self.llm_model, peft_config)
                self.llm_model.print_trainable_parameters()
        elif tune_llm == "freeze":
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False
        elif tune_llm == "full":
            pass
        else:
            raise NotImplementedError()


        _lg = bool(getattr(args, "label_gather_loss", False))
        _mids = list(getattr(self.llm_tokenizer, "molpo_mask_ids", []) or [])
        _n_set = 0
        for _m in self.llm_model.modules():
            if type(_m).__name__.endswith("ForCausalLM_custom"):
                _m.label_gather_loss = _lg


                _m.molpo_mask_ids = _mids
                _n_set += 1
        print(f"label_gather_loss={_lg} set on {_n_set} causal-LM module(s); "
              f"molpo_mask_ids={len(_mids)}", flush=True)

        self.set_params_requires_grads(
            model=self.llm_model, keyword="embed", grad=True, IsPrint=False
        )


        self.set_params_requires_grads(
            model=self.llm_model, keyword="lm_head", grad=True, IsPrint=False
        )

        if self.args.llava_pretraining:
            self.set_params_requires_grads(
                model=self.llm_model, keyword="lora", grad=False, IsPrint=False
            )

        if "graph" in self.args.mol_representation:
            self.graph_encoder, self.ln_graph = self.init_graph_encoder(args)

            self.tune_gnn = tune_gnn
            if not tune_gnn:
                for name, param in self.graph_encoder.named_parameters():
                    param.requires_grad = False
                self.graph_encoder = self.graph_encoder.eval()
                self.graph_encoder.train = disabled_train
                print("freeze graph encoder")

            if self.args.projector_type == "qformer":

                self.num_query_token = num_query_token
                self.Qformer, self.query_tokens = self.init_Qformer(
                    bert_name,
                    num_query_token,
                    gnn_hidden_dim,
                    cross_attention_freq,
                    bert_num_hidden_layers=args.bert_num_hidden_layers,
                    revision=getattr(args, "bert_revision", None),
                )

                ## remove the unused parameters
                self.Qformer.cls = None
                self.Qformer.bert.embeddings.word_embeddings = None
                self.Qformer.bert.embeddings.position_embeddings = None
                for layer in self.Qformer.bert.encoder.layer:
                    layer.output = None
                    layer.intermediate = None

                self.opt_proj = nn.Linear(
                    self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
                )
            elif self.args.projector_type == "mlp":
                # build self.opt_proj with single layers
                self.opt_proj = nn.Linear(
                    gnn_hidden_dim, self.llm_model.config.hidden_size
                )

    def get_lora_target_modules(self):
        return ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]

    def set_llm_model(self, llm_model):
        if llm_model == "facebook/galactica-125m":
            self.llm_model = OPTForCausalLM_Custom.from_pretrained(
                llm_model, torch_dtype=torch.bfloat16
            )
        else:
            self.llm_model = OPTForCausalLM_Custom.from_pretrained(
                llm_model,
                torch_dtype=(
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                ),
            )

    def add_necessary_tokens(self):
        # pad toekn for galactica is "<pad>""
        if not self.llm_tokenizer.pad_token:
            self.llm_tokenizer.add_special_tokens({"pad_token": "<pad>"})
        if not self.llm_tokenizer.eos_token:
            self.llm_tokenizer.add_special_tokens({"eos_token": "\n"})

        if self.args.add_selfies_tokens:
            # Read txt from selfies_token_path
            with open(self.args.selfies_token_path, "r") as f:
                selfies_tokens = f.readlines()
                selfies_tokens = [token.strip() for token in selfies_tokens]
            self.llm_tokenizer.add_tokens(selfies_tokens)
            # get token id of the selfies_tokens
            self.llm_tokenizer.selfies_token_ids = [
                self.llm_tokenizer.convert_tokens_to_ids(token)
                for token in selfies_tokens
            ]
            self.llm_tokenizer.added_selfies_tokens = selfies_tokens
            # remove '.' from the marked list for selfies token
            # self.llm_tokenizer.added_selfies_tokens.remove(".")
            # self.llm_tokenizer.selfies_token_ids.remove(36)
            print(f"Added {len(selfies_tokens)} selfies tokens to the tokenizer")

        additional_tokens = [
            getattr(added_tokens, tokens)
            for tokens in dir(added_tokens)
            if not re.match("__.*__", tokens)
        ]
        additional_tokens = [
            token for sublist in additional_tokens for token in sublist
        ]

        self.llm_tokenizer.add_tokens(additional_tokens)

        molpo_mask_tokens = (
            added_tokens.BOOL
            + added_tokens.FLOAT
            + added_tokens.DESCRIPTION
            + added_tokens.SELFIES
            + added_tokens.IUPAC
            + added_tokens.MOLFORMULA
        )
        molpo_mask_tokens += [self.llm_tokenizer.eos_token]
        self.llm_tokenizer.molpo_mask_tokens = molpo_mask_tokens

        # get ids of task tokens
        self.llm_tokenizer.molpo_mask_ids = [
            self.llm_tokenizer.convert_tokens_to_ids(token)
            for token in molpo_mask_tokens
        ]  # if llm model is mistral, add
        if "mistral" in self.llm_tokenizer.name_or_path:
            self.llm_tokenizer.molpo_mask_ids += [29473]  # '_' token id
            self.llm_tokenizer.molpo_mask_tokens += [
                self.llm_tokenizer.convert_ids_to_tokens(29473)
            ]

        # self.llm_tokenizer.mol_token = added_tokens.MOL_EMBEDDING[0]
        self.llm_tokenizer.add_special_tokens(
            {"additional_special_tokens": [added_tokens.MOL_EMBEDDING[0]]}
        )
        self.llm_tokenizer.mol_token_id = self.llm_tokenizer.convert_tokens_to_ids(
            added_tokens.MOL_EMBEDDING[0]
        )

    def merge_and_initialize_lora(self):
        self.model.blip2model.llm_model.merge_and_unload(progressbar=True)

        if self.tune_llm == "lora":
            if self.peft_dir:
                self.llm_model = PeftModel.from_pretrained(
                    self.llm_model, self.peft_dir, is_trainable=True
                )
            else:
                if self.args.peft_config:
                    peft_config = LoraConfig(
                        **LoraConfig.from_json_file(self.args.peft_config)
                    )
                else:
                    peft_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        r=self.args.lora_r,
                        lora_alpha=self.args.lora_alpha,
                        lora_dropout=self.args.lora_dropout,
                    )
                self.peft_config = peft_config
                self.llm_model = get_peft_model(self.llm_model, peft_config)
                self.llm_model.print_trainable_parameters()
        elif self.tune_llm == "freeze":
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False
        elif self.tune_llm == "full":
            pass
        else:
            raise NotImplementedError()

    def forward(self, batch):
        input_ids = batch.input_ids  # ['input_ids']
        target_ids = batch.labels  # ['labels']


        attention_mask = batch.get("attention_mask", None)
        position_ids = batch.get("position_ids", None)


        targets = target_ids.masked_fill(
            target_ids == self.llm_tokenizer.pad_token_id, -100
        )

        if "graphs" in batch.keys():
            graphs = batch["graphs"]
            additional_graphs = batch["additional_graphs"]
            is_mol_token = batch["is_mol_token"]

            input_embeds = self.llm_model.get_input_embeddings()(input_ids)
            input_embeds, graph_avg_norm, moltoken_avg_norm = (
                self.inject_graph_embeds2input_embeds(
                    input_embeds=input_embeds,
                    is_mol_token=is_mol_token,
                    # graphs=graphs,
                    graphs=(graphs, additional_graphs),
                    position_ids=position_ids,
                )
            )

            outputs = self.llm_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
                labels=targets,
            )
            results = {
                "loss": outputs.loss,
                "instance_loss": outputs.instance_loss,
                "instance_loss_molpo": getattr(outputs, "instance_loss_molpo", None),
                "logits": outputs.logits,
                "graph_avg_norm": graph_avg_norm,
                "moltoken_avg_norm": moltoken_avg_norm,


                "gathered_logps": getattr(outputs, "gathered_logps", None),
                "gathered_selected": getattr(outputs, "gathered_selected", None),
            }

        else:
            outputs = self.llm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
                labels=targets,
            )
            results = {
                "loss": outputs.loss,
                "instance_loss": outputs.instance_loss,
                "instance_loss_molpo": getattr(outputs, "instance_loss_molpo", None),
                "logits": outputs.logits,
                "gathered_logps": getattr(outputs, "gathered_logps", None),
                "gathered_selected": getattr(outputs, "gathered_selected", None),
            }

        return results

    def debug_pred(self, logits, targets):
        max_logits = logits.argmax(dim=-1)
        target_masks = targets != -100
        predictions = []
        labels = []
        for i in range(max_logits.shape[0]):
            max_logit = max_logits[i]
            target = targets[i]
            target_mask = target_masks[i]

            # prediction = self.llm_tokenizer.decode(max_logit)
            # label = self.llm_tokenizer.decode(target)

            prediction = self.llm_tokenizer.decode(max_logit[target_mask])
            label = self.llm_tokenizer.decode(target[target_mask])
            predictions.append(prediction)
            labels.append(label)
        return predictions, labels

    def inject_graph_embeds2input_embeds(
        self, input_embeds, is_mol_token, graphs, position_ids=None
    ):
        mol_graphs, mol2_graphs = graphs


        if position_ids is not None and is_mol_token.size(0) == 1 and bool(
            (position_ids[0, 1:] - position_ids[0, :-1] < 0).any()
        ):
            _seg = torch.cumsum((position_ids[0] == 0).long(), dim=0) - 1
            _tok = is_mol_token[0].nonzero(as_tuple=True)[0]
            max_mol_tokens_needed = int(torch.bincount(_seg[_tok]).max()) if _tok.numel() else 0
        else:
            max_mol_tokens_needed = int(is_mol_token.sum(dim=1).max()) if is_mol_token.numel() else 0

        mol_token_sequence = []

        for graph_slot, graphs in enumerate([mol_graphs, mol2_graphs]):
            if graph_slot == 1 and mol_token_sequence and (
                max_mol_tokens_needed <= mol_token_sequence[0].size(1)
            ):
                break

            mol_x = graphs["x"]
            mol_edge_index = graphs["edge_index"]
            mol_edge_attr = graphs["edge_attr"]
            mol_batch = graphs["batch"]

            if self.args.process_disjoint:
                num_graph_list = []
                graph_list = []
                for graph in graphs.to_data_list():
                    tmp_batch = Batch.from_data_list([graph])
                    tmp_batch = split_batch_by_components(tmp_batch)
                    graph_list.extend(tmp_batch)
                    num_graph_list.append(len(tmp_batch))
                # graph_batch = split_batch_by_components(graphs)
                graph_batch = Batch.from_data_list(graph_list)
                graph_embeds, graph_masks = self.graph_encoder(
                    graph_batch.x,
                    graph_batch.edge_index,
                    graph_batch.edge_attr,
                    graph_batch.batch,
                )
                mol_embeds_list = []
                mol_mask_list = []

                graph_embeds = torch.split(graph_embeds, num_graph_list, dim=0)
                graph_masks = torch.split(graph_masks, num_graph_list, dim=0)
                for graph_embed, graph_mask in zip(graph_embeds, graph_masks):
                    mol_embeds_list.append(graph_embed[graph_mask])
                    mol_mask_list.append(graph_mask[graph_mask])
                mol_embeds = pad_sequence(mol_embeds_list, batch_first=True)
                mol_masks = pad_sequence(mol_mask_list, batch_first=True)
            else:
                mol_embeds, mol_masks = self.graph_encoder(
                    mol_x, mol_edge_index, mol_edge_attr, mol_batch
                )


            if not self.tune_gnn:
                mol_embeds = mol_embeds.detach()
            mol_embeds = self.ln_graph(mol_embeds, mol_masks)


            if graph_slot == 0:
                graph_embedding = mol_embeds[:, 0, :]
                graph_avg_norm = torch.norm(graph_embedding, p=1, dim=-1)

            if self.args.projector_type == "qformer":
                query_tokens = self.query_tokens.expand(mol_embeds.shape[0], -1, -1)
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=mol_embeds,
                    encoder_attention_mask=mol_masks,
                    return_dict=True,
                )
                mol_tokens = self.opt_proj(query_output.last_hidden_state)
            else:
                mol_tokens = self.opt_proj(mol_embeds)
            mol_token_sequence.append(mol_tokens)


        mol_tokens = torch.cat(mol_token_sequence, dim=1)
        moltoken_avg_norm = torch.norm(mol_tokens, p=1, dim=-1).mean(1)

        num_mol_tokens_per_sample = is_mol_token.sum(dim=1)  # Shape: (batch_size,)
        if (num_mol_tokens_per_sample > 0).any():
            if position_ids is not None and is_mol_token.size(0) == 1 and bool(
                (position_ids[0, 1:] - position_ids[0, :-1] < 0).any()
            ):


                segment_of_position = (
                    torch.cumsum((position_ids[0] == 0).long(), dim=0) - 1
                )
                token_indices = is_mol_token[0].nonzero(as_tuple=True)[0]
                instance_indices = segment_of_position[token_indices]


                counts = torch.bincount(
                    instance_indices, minlength=mol_tokens.size(0)
                )
                offsets = torch.cumsum(counts, dim=0) - counts
                within_instance = (
                    torch.arange(token_indices.numel(), device=token_indices.device)
                    - offsets[instance_indices]
                )
                assert int(counts.max()) <= mol_tokens.size(1), (
                    f"an instance carries {int(counts.max())} <mol> tokens but only "
                    f"{mol_tokens.size(1)} graph tokens were produced for it"
                )
                input_embeds[0, token_indices, :] = mol_tokens[
                    instance_indices, within_instance, :
                ]
            else:
                mol_token_indices_full = (
                    is_mol_token.cumsum(dim=1) - 1
                )


                batch_indices, token_indices = is_mol_token.nonzero(
                    as_tuple=True
                )


                mol_token_indices = mol_token_indices_full[
                    batch_indices, token_indices
                ]
                input_embeds[batch_indices, token_indices, :] = mol_tokens[
                    batch_indices, mol_token_indices, :
                ]

        return input_embeds, graph_avg_norm, moltoken_avg_norm

    @torch.no_grad()
    def generate(
        self,
        graphs,
        # input_tokens,
        input_ids,
        attention_mask,
        is_mol_token=None,
        do_sample=False,
        num_beams=5,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        output_attentions=False,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """

        input_embeds = self.llm_model.get_input_embeddings()(input_ids)
        if "graph" in self.args.mol_representation:
            assert (
                is_mol_token is not None
            ), "is_mol_token should be provided for graph representation"
            input_embeds, graph_avg_norm, moltoken_avg_norm = (
                self.inject_graph_embeds2input_embeds(
                    input_embeds=input_embeds,
                    is_mol_token=is_mol_token,
                    graphs=graphs,
                )
            )

        outputs = self.llm_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_length,
            # min_length=min_length,
            min_new_tokens=min_length,  # TODO: change to min_new_tokens for all layered methods
            # pad_token_id=self.pad_token_id,
            eos_token_id=self.llm_tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
            output_scores=True,
            output_logits=True,
            return_dict_in_generate=True,
            output_attentions=output_attentions,
        )

        batch_size, sequence_length = outputs.sequences.shape
        # stack logtis
        logits_stacked = torch.zeros(
            batch_size,
            0,
            self.llm_model.config.vocab_size,
            device=outputs.logits[0].device,
        )
        for i in range(sequence_length):
            logits = outputs.logits[i].unsqueeze(1)
            logits = (
                logits.view(batch_size, num_beams, -1).max(dim=1).values.unsqueeze(1)
            )
            logits_stacked = torch.cat([logits_stacked, logits], dim=1)

        outputs.logits = logits_stacked
        output_text = self.llm_tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=False
        )

        output_text = [text.strip() for text in output_text]
        outputs.predictions = output_text
        return outputs


_CONFIG_FOR_DOC = "OPTConfig"


class OPTForCausalLM_Custom(OPTForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        # conventional decoder implementation does not support sequence packing
        # new implementations for pos embeds, 4d causal attention mask is required for sequence packing
        self.model.decoder = OPTDecoder_sequence_packing(config)

    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForCausalLM

        >>> model = OPTForCausalLM_Custom.from_pretrained("facebook/opt-350m")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious. I'm just a little bit of a weirdo."
        ```"""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0]).contiguous()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)

            # custom forward to get not reduced loss
            loss_fct_not_reduced = CrossEntropyLoss(reduction="none")
            loss_not_reduced = loss_fct_not_reduced(shift_logits, shift_labels).view(
                labels.size(0), -1
            )
            # normalization exclude default ignore index -100
            instance_non_pad_tokens = torch.where(
                shift_labels != -100,
                torch.tensor(1).to(shift_labels.device),
                torch.tensor(0).to(shift_labels.device),
            ).view(labels.size(0), -1)
            instance_loss = (loss_not_reduced * instance_non_pad_tokens).sum(
                dim=-1
            ) / instance_non_pad_tokens.sum(dim=-1)

            # cross entropy aggregate not row-wise, but sum of all instances
            loss = (
                loss_not_reduced * instance_non_pad_tokens
            ).sum() / instance_non_pad_tokens.sum()
        else:
            instance_loss = None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast_Custom(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            instance_loss=instance_loss,
        )


from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass


@dataclass
class CausalLMOutputWithPast_Custom(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    instance_loss: Optional[torch.FloatTensor] = None


from transformers.models.opt.modeling_opt import OPTDecoder
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask


class OPTDecoder_sequence_packing(OPTDecoder):
    def __init__(self, config):
        super().__init__(config)
        self.embed_positions = OPTLearnedPositionalEmbedding_sequence_packing(
            config.max_position_embeddings, config.hidden_size
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )
        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values_length + seq_length

        # embed positions
        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            causal_attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
            attention_mask = (
                torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
                if attention_mask is None
                else attention_mask
            )
        else:
            # 4d mask is passed through the layers
            if attention_mask is None:
                attention_mask = torch.ones(
                    batch_size, mask_seq_length, device=inputs_embeds.device
                )
            # NOTE: implemented for passing 4d causal mask for sequence packing
            elif len(attention_mask.shape) == 4 and attention_mask.shape[1] == 1:
                attention_mask = attention_mask.to(
                    dtype=inputs_embeds.dtype, device=inputs_embeds.device
                )
            elif attention_mask.shape[1] != mask_seq_length:
                raise ValueError(
                    f"The provided attention mask has length {attention_mask.shape[1]}, but its length should be "
                    f"{mask_seq_length} (sum of the lengths of current and past inputs)"
                )
            causal_attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# NOTE: implemented for positional embedding not affected by packed sequence
class OPTLearnedPositionalEmbedding_sequence_packing(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(
        self, attention_mask: torch.LongTensor, past_key_values_length: int = 0
    ):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()
        # NOTE: implemented for positional embedding not affected by packed sequence
        if len(attention_mask.shape) == 4:
            attention_mask = attention_mask.squeeze(1)
            # TODO: mark the padding token as -1, though it does not affect the loss calculation
            positions = attention_mask.sum(dim=-1) - 1
            assert (
                positions.max() < self.num_embeddings
            ), f"Positional index is out of range: position {positions.max()} >= num_embeddings {self.num_embeddings}"
        else:
            # create positions depending on attention_mask
            positions = (
                torch.cumsum(attention_mask, dim=1).type_as(attention_mask)
                * attention_mask
            ).long() - 1

            # cut positions if `past_key_values_length` is > 0
            positions = positions[:, past_key_values_length:]

        return super().forward(positions + self.offset)
