"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import contextlib
import logging
import os

import torch
import torch.nn as nn

from lavis.common.dist_utils import download_cached_file
from lavis.common.utils import is_url
from lavis.models.base_model import BaseModel
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
from transformers import BertTokenizer
from model.gin_model import GNN, GNN_MoleculeSTM
from model.tokenGT import BERTTokenGT
from model.gine_tokengt import GINE_TokenGT
from collections import OrderedDict


class Blip2Base(BaseModel):
    @classmethod
    def init_tokenizer(cls, revision=None):
        if True:
            bert_name = "allenai/scibert_scivocab_uncased"
        else:
            bert_name = "bert_pretrained/"
        tokenizer = BertTokenizer.from_pretrained(bert_name, revision=revision)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.amp.autocast(dtype=dtype, device_type="cuda")
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_Qformer(
        cls,
        model_name,
        num_query_token,
        graph_width,
        cross_attention_freq=2,
        bert_num_hidden_layers=-1,
        revision=None,
    ):
        assert model_name == "scibert"
        print("bert load scibert")
        if True:
            bert_name = "allenai/scibert_scivocab_uncased"
        else:
            bert_name = "bert_pretrained/"

        encoder_config = BertConfig.from_pretrained(bert_name, revision=revision)
        encoder_config.encoder_width = graph_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        if bert_num_hidden_layers > 0:
            encoder_config.num_hidden_layers = bert_num_hidden_layers

        Qformer = BertLMHeadModel.from_pretrained(bert_name, config=encoder_config, revision=revision)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    @classmethod
    def init_graph_encoder(cls, args):

        if args.gnn_type == "gine":
            graph_encoder = GNN_MoleculeSTM(
                num_layer=args.gnn_num_layers,
                emb_dim=args.gnn_hidden_dim,
                gnn_type="gin",
                drop_ratio=args.drop_ratio,
                JK=args.gnn_jk,
                args=args,
            )
        elif args.gnn_type == "tokengt":
            graph_encoder = BERTTokenGT(
                input_feat_dim=args.input_feat_dim,
                hidden_dim=args.gnn_hidden_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                method=args.method,
                d_p=args.d_p,
                d_e=args.d_e,
                use_graph_token=args.use_graph_token,
                max_position_embeddings=args.max_position_embeddings,
            )
        elif args.gnn_type == "gine_tokengt":
            graph_encoder = GINE_TokenGT(args)
            ln_graph = LayerNorm(args.gine.gnn_hidden_dim)

            return graph_encoder, ln_graph

        if "MoleculeSTM" in args.graph_encoder_ckpt:
            if args.graph_encoder_ckpt is not None:
                ckpt = torch.load(
                    args.graph_encoder_ckpt, map_location=torch.device("cpu")
                )
                renamed_state_dict = {}
                for k, v in ckpt.items():
                    if k.startswith("molecule_node_model."):
                        renamed_state_dict[k.replace("molecule_node_model.", "")] = v
                ckpt = renamed_state_dict
                print(f"load graph encoder from {args.graph_encoder_ckpt}")
                missing_keys, unexpected_keys = graph_encoder.load_state_dict(
                    ckpt, strict=False
                )
                if len(missing_keys) or len(unexpected_keys):
                    print(missing_keys)
                    print(unexpected_keys)
        elif "Custom_gnn_models" in args.graph_encoder_ckpt:
            ckpt = torch.load(args.graph_encoder_ckpt, map_location=torch.device("cpu"))
            renamed_state_dict = {}
            for param, value in ckpt["state_dict"].items():
                if param.startswith("gnn."):
                    renamed_state_dict[param.replace("gnn.", "")] = value
            graph_encoder.load_state_dict(renamed_state_dict, strict=True)
            print(f"load graph encoder from {args.graph_encoder_ckpt}")
        elif "scratch" in args.graph_encoder_ckpt:
            pass
        else:
            raise NotImplementedError(
                f"Please provide a valid graph encoder checkpoint. {args.graph_encoder_ckpt} is not supported."
            )

        ln_graph = LayerNorm(args.gnn_hidden_dim)

        # qm9_pretrained = torch.load("gnn_ablation/all_except_lumo_homo_gap_scaled/GM_GM_-_-/lightning_logs/version_0/checkpoints/best-model.ckpt")['state_dict']
        # qm9_renamed_gnn_state_dict = {}
        # qm9_renamed_lngraph_state_dict = {}

        # for name, param in qm9_pretrained.items():
        #     if name.startswith("mlp."):
        #         continue
        #     if name.startswith('ln_graph.'):
        #         qm9_renamed_lngraph_state_dict[name.replace("ln_graph.", "")] = param
        #         continue
        #     renamed = name.replace("gin.", "")
        #     qm9_renamed_gnn_state_dict[renamed] = param

        # graph_encoder.load_state_dict(qm9_renamed_gnn_state_dict, strict=True)
        # ln_graph.load_state_dict(qm9_renamed_lngraph_state_dict, strict=True)

        return graph_encoder, ln_graph

    @classmethod
    def init_graph_decoder(cls, num_layers, hidden_dim, drop_ratio, args):
        from train_graph_decoder import GraphReconstruction

        if args.graph_decoder_ckpt is not None:
            graph_enc_dec = GraphReconstruction.load_from_checkpoint(
                args.graph_decoder_ckpt, device=args.devices, args=args
            )
        else:
            graph_enc_dec = GraphReconstruction(args)
        graph_decoder = graph_enc_dec.decoder
        logging.info(f"load graph decoder from {args.graph_decoder_ckpt}")
        graph_enc_dec = None
        return graph_decoder

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg

    def set_params_requires_grads(cls, model, keyword, grad=True, IsPrint=True):
        names = []
        for name, param in model.named_parameters():
            if keyword in name:
                param.requires_grad = grad
                names.append(name)
        if IsPrint:
            for n in names:
                print(f"{n} set to requires_grad: {grad}")

    def get_params_by_keywords(state_dict, keywords):
        """
        Filters a state_dict to include only parameters whose names contain any of the specified keywords.

        Args:
            state_dict (dict or OrderedDict): The model's state_dict.
            keywords (str or list of str): A keyword or a list of keywords to search for in parameter names.

        Returns:
            OrderedDict: A new dictionary containing only the matching parameters.
                        Using OrderedDict to preserve original parameter order.
        """
        if isinstance(keywords, str):
            keywords = [keywords]  # Convert single keyword to list for uniformity

        # Using a dictionary comprehension for conciseness
        filtered_params = OrderedDict({
            param_name: param_tensor
            for param_name, param_tensor in state_dict.items()
            if any([keyword in param_name for keyword in keywords])  # Include if any keyword matches
        })
        return filtered_params

    def get_params_without_keywords(state_dict, keywords_to_exclude):
        """
        Filters a state_dict to include only parameters whose names
        do NOT contain any of the specified keywords.

        Args:
            state_dict (dict or OrderedDict): The model's state_dict.
            keywords_to_exclude (list of str): A list of keywords to exclude from parameter names.

        Returns:
            OrderedDict: A new dictionary containing only the parameters whose
                        names do not contain any of the specified keywords.
        """
        # Using a dictionary comprehension for conciseness
        filtered_params = OrderedDict({
            param_name: param_tensor
            for param_name, param_tensor in state_dict.items()
            if not any([keyword in param_name for keyword in keywords_to_exclude])  # Exclude if any keyword matches
        })
        return filtered_params

    def check_grads(cls, model, keyword):
        names = []
        requires_grad = []
        for name, param in model.named_parameters():
            if keyword in name:
                names.append(name)
                requires_grad.append(param.requires_grad)
                print(name, param.requires_grad)
        if len(requires_grad) == 0:
            print("No param with keyword found")
        else:
            print("=====================================")
            print("all params with keyword requires grad:", all(requires_grad))
            print("=====================================")


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor, mask=None):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
