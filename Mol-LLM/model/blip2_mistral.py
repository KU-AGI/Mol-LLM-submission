"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.amp import autocast as autocast

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import replace_return_docstrings
from transformers import AutoModelForCausalLM
from transformers.models.mistral.modeling_mistral import MistralModel


from model.blip2_opt import Blip2OPT
from typing import Optional, List, Tuple, Union
import logging


_CONFIG_FOR_DOC = "MistralConfig"


# @registry.register_model("blip2")
# @registry.register_model("blip2_feature_extractor")
class Blip2Mistral(Blip2OPT):
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

    def __init__(self, args):
        from instructions_smol import validate_runtime_revision
        validate_runtime_revision(getattr(args, "prompt_revision", None), getattr(args, "eval_molpo", False),
                                  getattr(args, "eval_molpo_endpoint_matched_donors", False))
        super().__init__(args=args)

        self.system_prompt_mode = getattr(args, "system_prompt_mode", "legacy")

    def set_llm_model(self, llm_model):


        kwargs = {"torch_dtype": torch.bfloat16}
        if getattr(self.args, "llm_revision", None):
            kwargs["revision"] = self.args.llm_revision
        attn_impl = getattr(self.args, "attn_implementation", None)
        if getattr(self.args, "apply_sequence_packing", False) and attn_impl is None:
            attn_impl = "flash_attention_2"
        if attn_impl is not None:
            kwargs["attn_implementation"] = attn_impl

        self.llm_model = MistralForCausalLM_custom.from_pretrained(llm_model, **kwargs)
        self.llm_model.label_gather_loss = getattr(
            self.args, "label_gather_loss", False
        )

        if attn_impl is not None:


            actual = self.llm_model.model.config._attn_implementation
            assert actual == attn_impl, (
                f"requested attn_implementation={attn_impl} but the model was built "
                f"with {actual}"
            )
        return

    def fit_llm_input_convention(self, llm_prompt):
        if self.args.llm_model == "mistralai/Mistral-7B-Instruct-v0.3":
            from instructions_smol import chat_messages
            message = chat_messages(llm_prompt, self.system_prompt_mode)
            formatted_ids = self.llm_tokenizer.apply_chat_template(conversation=message)
            formatted_text = self.llm_tokenizer.decode(
                formatted_ids,
            )
            return formatted_text
        else:
            raise ValueError("llm_model template formatting is not implemented")

    def get_lora_target_modules(self):
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]


from transformers.utils.doc import add_start_docstrings_to_model_forward
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import is_torchdynamo_compiling
from transformers.models.mistral.modeling_mistral import (
    MistralForCausalLM,
    MISTRAL_INPUTS_DOCSTRING,
)

logger = logging.get_logger(__name__)

def _is_packed(position_ids):
    return (
        position_ids is not None
        and position_ids.size(0) == 1
        and bool((position_ids[0, 1:] - position_ids[0, :-1] < 0).any())
    )

def _instance_loss(token_loss, selected, position_ids):
    if _is_packed(position_ids):

        segment_of_position = torch.cumsum((position_ids[0] == 0).long(), dim=0) - 1
        index = segment_of_position[1:][selected[0]]
        n_instances = int(segment_of_position[-1]) + 1
    else:
        rows = torch.arange(selected.size(0), device=selected.device)
        index = rows.unsqueeze(1).expand_as(selected)[selected]
        n_instances = selected.size(0)

    totals = torch.zeros(n_instances, dtype=token_loss.dtype, device=token_loss.device)
    counts = torch.zeros_like(totals)
    totals.index_add_(0, index, token_loss)
    counts.index_add_(0, index, torch.ones_like(token_loss))

    return totals / counts

class MistralForCausalLM_custom(MistralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = MistralModel(config)

    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MistralForCausalLM

        >>> model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
        >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
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
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if labels is None and not is_torchdynamo_compiling():
            logger.warning_once(
                "Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)"
            )
        _lg_flag = getattr(self, 'label_gather_loss', False)
        if labels is not None and _lg_flag and torch.is_grad_enabled():
            shift_hidden = hidden_states[:, :-1, :]
            shift_labels = labels[:, 1:]
            selected = shift_labels != -100
            loss_not_reduced = CrossEntropyLoss(reduction='none')(
                self.lm_head(shift_hidden[selected]).float(), shift_labels[selected]
            )
            loss_sum = loss_not_reduced.sum()
            label_token_count = selected.sum()
            loss = loss_sum / label_token_count.clamp(min=1)
            instance_loss = _instance_loss(loss_not_reduced, selected, position_ids)
            if not return_dict:
                return (loss, None) + outputs[1:]
            return CausalLMOutputWithPast_Custom(
                loss=loss,
                loss_sum=loss_sum,
                label_token_count=label_token_count,
                logits=None,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                instance_loss=instance_loss,
                gathered_logps=-loss_not_reduced,
                gathered_selected=selected,
            )
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()

        loss = None
        loss_sum = label_token_count = None
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
            if _is_packed(position_ids):
                selected = instance_non_pad_tokens.bool()
                instance_loss = _instance_loss(loss_not_reduced[selected], selected, position_ids)
            instance_loss_molpo = None
            _mids = getattr(self, 'molpo_mask_ids', None)
            if _mids:
                _sl = shift_labels.view(labels.size(0), -1)
                molpo_keep = instance_non_pad_tokens.clone()
                for _mid in _mids:
                    molpo_keep = molpo_keep * (_sl != _mid).to(molpo_keep.dtype)
                if _is_packed(position_ids):
                    _sel_m = molpo_keep.bool()
                    instance_loss_molpo = _instance_loss(loss_not_reduced[_sel_m], _sel_m, position_ids)
                else:
                    _den = molpo_keep.sum(dim=-1)
                    instance_loss_molpo = (loss_not_reduced * molpo_keep).sum(dim=-1) / _den.clamp(min=1)
                    instance_loss_molpo = instance_loss_molpo.masked_fill(_den == 0, float('nan'))
            loss_sum = (loss_not_reduced * instance_non_pad_tokens).sum()
            label_token_count = instance_non_pad_tokens.sum()
            loss = loss_sum / label_token_count.clamp(min=1)
        else:
            instance_loss = None
            instance_loss_molpo = None
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast_Custom(
            loss=loss,
            loss_sum=loss_sum,
            label_token_count=label_token_count,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            instance_loss=instance_loss,
            instance_loss_molpo=instance_loss_molpo,
        )


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


    instance_loss_molpo: Optional[torch.FloatTensor] = None


    gathered_logps: Optional[torch.FloatTensor] = None
    gathered_selected: Optional[torch.BoolTensor] = None

    loss_sum: Optional[torch.FloatTensor] = None
    label_token_count: Optional[torch.LongTensor] = None
