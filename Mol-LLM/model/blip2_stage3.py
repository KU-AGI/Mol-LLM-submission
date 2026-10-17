import math
from model.molpo import (
    MODE as MOLPO_WINDOW_MODE,
    MolPOWindowReducer,
    WindowTaskEMA,
    count_molpo_terms,
)
from model.scheduler import LinearWarmupConstantLRScheduler
from data_utils import molpo_batch_division
import os
from typing import Any, Dict
import torch
from model.blip2_opt import Blip2OPT
from model.blip2_llama import Blip2Llama
from model.blip2_mistral import Blip2Mistral
from model.blip2_t5 import Blip2T5
import pytorch_lightning as pl
from torch import optim
from model.scheduler import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
import json
from model.help_funcs import (
    per_device_evaluate,
    total_device_evaluate,
    AttrDict,
    convert_logit2binary_prob,
)
from transformers import Adafactor
import json
from data_utils import CLASSIFICATION_BENCHMARKS, id2task
from transformers.utils import logging

from torch.nn import CrossEntropyLoss

logger = logging.get_logger(__name__)
logging.set_verbosity_info()


def load_ignore_unexpected(model, state_dict):
    keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in keys}

    # try to print keys that are not included
    model.load_state_dict(state_dict, strict=True)


def get_module_state_dict(state_dict, module_name):
    module_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(module_name):
            key = key[len(module_name) + 1 :]
            if key == "":
                return value
            module_state_dict[key] = value
    return module_state_dict


class Blip2Stage3(pl.LightningModule):
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        to_be_removed = []
        for key, value in checkpoint["state_dict"].items():
            # graph encoder parameters
            if (
                "running_mean" in key
                or "running_var" in key
                or "num_batches_tracked" in key
                or "lora" in key
            ):
                continue
            try:
                if not self.get_parameter(key).requires_grad:
                    to_be_removed.append(key)
            except AttributeError:
                to_be_removed.append(key)
        for key in to_be_removed:
            checkpoint["state_dict"].pop(key)

        if hasattr(self, "task_specific_chosen_reward"):
            checkpoint[f"task_specific_chosen_reward"] = (
                self.task_specific_chosen_reward
            )

        self.log_model_parameters()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if "task_specific_chosen_reward" in checkpoint:
            self.task_specific_chosen_reward = checkpoint["task_specific_chosen_reward"]

    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)

        self.args = args
        if getattr(args, 'sft_loss_reduction', 'microbatch_token_mean') != 'microbatch_token_mean':
            raise ValueError('Only microbatch_token_mean is supported for standalone SFT')
        self._molpo_window_enabled = getattr(args, 'molpo_loss_reduction', 'microbatch_means') == MOLPO_WINDOW_MODE
        self._molpo_window_reducer = None
        if getattr(args, 'molpo_loss_reduction', 'microbatch_means') not in ('microbatch_means', MOLPO_WINDOW_MODE):
            raise ValueError('unknown MolPO reduction policy')
        if self._molpo_window_enabled:
            if not getattr(args, 'train_molpo', False):
                raise ValueError('global_molpo_terms_v1 requires MolPO')
            self._molpo_window_kappa = WindowTaskEMA(
                decay=float(getattr(args, 'molpo_kappa_window_decay', 0.99)),
                placeholders=getattr(args, 'molpo_placeholder_tasks', ()) or (),
            )
        _mv = getattr(args, 'molpo_variant', None)
        if _mv not in (None, 'v7', 'off'):
            raise ValueError('Only the main MolPO objective is supported')
        if _mv == 'v7':
            args.loss_type = 'v7'
        if _mv == 'off':
            args.train_molpo = False
        if getattr(args, 'molpo_share_prefix', False):
            from model.molpo_shared_prefix import install_hook
            install_hook()
        self.num_beams = args.num_beams
        self.gen_max_len = args.gen_max_len
        self.min_len = args.min_len
        self.tune_llm = args.tune_llm
        self.on_second_stage = False
        # set strict_loading to False to load model in a lightweight way
        self.strict_loading = False
        if "galactica" in args.llm_model:
            blip2model = Blip2OPT
        elif "llama" in args.llm_model:
            blip2model = Blip2Llama
        elif "mistral" in args.llm_model:
            blip2model = Blip2Mistral
        elif "t5" in args.llm_model:
            blip2model = Blip2T5
        else:
            raise NotImplementedError()

        self.blip2model = blip2model(args)
        self.tokenizer = self.blip2model.init_tokenizer(revision=getattr(args, "bert_revision", None))
        self.save_hyperparameters(args)
        if getattr(args, 'train_molpo', False) and (not self._molpo_window_enabled):
            raise ValueError('MolPO training requires global_molpo_terms_v1')

    def load_from_stage1_checkpoint(self, path):
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt["state_dict"]
        graph_encoder_dict = get_module_state_dict(
            state_dict, "blip2qformer.graph_encoder"
        )
        qformer_dict = get_module_state_dict(state_dict, "blip2qformer.Qformer")
        ln_graph_dict = get_module_state_dict(state_dict, "blip2qformer.ln_graph")
        qs_weight = get_module_state_dict(state_dict, "blip2qformer.query_tokens")
        load_ignore_unexpected(self.blip2model.Qformer, qformer_dict)
        self.blip2model.graph_encoder.load_state_dict(graph_encoder_dict)
        self.blip2model.ln_graph.load_state_dict(ln_graph_dict)
        self.blip2model.query_tokens.data.copy_(qs_weight)
        return self

    def configure_optimizers(self):
        if self.args.optimizer == "adafactor":
            print("Using adafactor optimizer")
            optimizer = Adafactor(
                self.parameters(),
                lr=1e-3,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
            )
            self.scheduler = None
        else:
            self.trainer.fit_loop.setup_data()
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.args.init_lr,
                weight_decay=self.args.weight_decay,
            )
            self.steps_per_epoch = (
                len(self.trainer.train_dataloader) / self.args.accumulate_grad_batches
            )
            if getattr(self, '_molpo_window_enabled', False):
                self.steps_per_epoch = math.ceil(self.trainer.num_training_batches / self.args.accumulate_grad_batches)

            max_step = int(self.args.max_epochs * self.steps_per_epoch)

            warmup_steps = self.args.warmup_steps

            if self.args.scheduler == "linear_warmup_cosine_lr":
                self.scheduler = LinearWarmupCosineLRScheduler(
                    optimizer=optimizer,
                    max_step=max_step,
                    min_lr=self.args.min_lr,
                    init_lr=self.args.init_lr,
                    warmup_steps=warmup_steps,
                    warmup_start_lr=self.args.warmup_lr,
                )
            elif self.args.scheduler == "linear_warmup_constant_lr":

                self.scheduler = LinearWarmupConstantLRScheduler(
                    optimizer=optimizer,
                    max_step=max_step,
                    min_lr=self.args.min_lr,
                    init_lr=self.args.init_lr,
                    warmup_steps=warmup_steps,
                    warmup_start_lr=self.args.warmup_lr,
                )
            elif self.args.scheduler == "linear_warmup_step_lr":
                self.scheduler = LinearWarmupStepLRScheduler(
                    optimizer,
                    self.args.max_epochs,
                    self.args.min_lr,
                    self.args.init_lr,
                    self.args.lr_decay_rate,
                    self.args.warmup_lr,
                    self.args.warmup_steps,
                )
            elif self.args.scheduler == "None":
                self.scheduler = None
            else:
                raise NotImplementedError()
        return optimizer

    def save_predictions(
        self,
        predictions,
        targets,
        tasks,
        prompts,
        input_mol_strings,
        probs=None,
        filename="predictions.json",
    ):
        assert len(predictions) == len(targets)
        assert len(predictions) == len(tasks)
        assert len(predictions) == len(prompts)
        assert len(predictions) == len(input_mol_strings)
        if probs is not None:
            assert len(predictions) == len(probs)
        instances = []
        for i in range(len(predictions)):
            instance = {
                "task": tasks[i],
                "prediction": predictions[i],
                "target": targets[i],
                "prompt": prompts[i],
                "input_mol_strings": input_mol_strings[i],
            }
            if tasks[i] in CLASSIFICATION_BENCHMARKS and probs is not None:
                instance["prob"] = probs[i]
            instances.append(instance)
        os.makedirs(self.logger.log_dir, exist_ok=True)

        with open(os.path.join(self.logger.log_dir, filename), "w") as f:
            json.dump(instances, f, ensure_ascii=False, indent=4)

    def on_test_epoch_start(self) -> None:
        self.on_evaluation_epoch_start()

    @torch.no_grad()
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.evaluation_step(batch, batch_idx, dataloader_idx, mode="test")

    def on_test_epoch_end(self):
        self.on_evaluation_epoch_end(mode="test")

    def on_validation_epoch_start(self) -> None:
        self.on_evaluation_epoch_start()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.evaluation_step(batch, batch_idx, dataloader_idx, mode="val")

    def on_validation_epoch_end(self) -> None:
        self.on_evaluation_epoch_end(mode="val")

    def get_total_molpo_loss(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        molpo_labels: torch.LongTensor,
        instance_loss: torch.FloatTensor,
        tasks,
        is_train=True,
        position_ids: torch.LongTensor = None,
        molpo_ok: torch.BoolTensor = None,
        gathered_logps=None,
        gathered_selected=None,
    ):
        if not self._molpo_window_enabled:
            raise ValueError('MolPO training requires global_molpo_terms_v1')
        _div = molpo_batch_division(self.args)
        assert (_div == 3) == bool(
            getattr(self.args, 'sft_reject_weight', None)
        ), f'sft_reject_weight={getattr(self.args, 'sft_reject_weight', None)!r} but the molpo batch division resolved to {_div}'
        out = concatenated_forward(
            all_logits=logits,
            all_labels=molpo_labels,
            label_pad_token_id=-100,
            instance_loss=instance_loss,
            position_ids=position_ids,
            gathered_logps=gathered_logps,
            gathered_selected=gathered_selected,
            division=_div,
        )
        sft_instance_loss = out["sft_instance_loss"]

        policy_chosen_logps = out["chosen_logps"]
        chosen_loss_counts = out['chosen_loss_counts']
        policy_rejected_logps = out["rejected_logps"]
        rejected_loss_counts = out['rejected_loss_counts']
        chosen_rewards = self.args.beta * policy_chosen_logps
        rejected_rewards = self.args.beta * policy_rejected_logps
        sft_all_counts = instance_label_counts(labels, position_ids)
        assert (
            sft_all_counts.shape[0] % _div == 0
        ), f'instance count should be divisible by the molpo batch division ({_div})'
        _n_inst = sft_all_counts.shape[0] // _div
        sft_loss_counts = sft_all_counts[:_n_inst]
        sft_loss = (sft_instance_loss * sft_loss_counts)[
            sft_loss_counts > 0
        ].sum() / sft_loss_counts.sum().clamp(min=1)
        if getattr(self, '_molpo_window_enabled', False):
            if gathered_logps is not None:
                sums, _ = reduce_logps_gathered(
                    gathered_logps,
                    gathered_selected,
                    position_ids,
                    average_log_prob=False,
                    sub_mask=labels[:, 1:] != -100,
                )
            else:
                lp, keep = shifted_per_token_logps(logits, labels)
                sums, _ = reduce_logps(lp, keep, position_ids, average_log_prob=False)
            sft_numerator = -sums[:_n_inst].sum()
            sft_loss = sft_numerator / sft_loss_counts.sum().clamp_min(1)
        reject_sft_loss = None
        if _div == 3:
            _rl = out['reject_own_logps']
            _rc = out['reject_own_counts']
            reject_sft_loss = (-_rl * _rc)[_rc > 0].sum() / _rc.sum().clamp(min=1)
            instance_reject_loss = (-_rl).clone().detach().cpu()
        if not hasattr(self, '_placeholder_tasks'):
            _cfg_ph = getattr(self.args, 'molpo_placeholder_tasks', None)
            if _cfg_ph is not None:
                self._placeholder_tasks = set(_cfg_ph)
                print(
                    f'[molpo] placeholder tasks from config, excluded from molpo logging: {sorted(self._placeholder_tasks)}',
                    flush=True,
                )
        if getattr(self, '_molpo_window_enabled', False):
            if is_train:
                self._molpo_window_kappa.add(tasks, policy_chosen_logps, chosen_loss_counts)
        per_task_scaling = self._molpo_window_kappa.scales(
            tasks, device=chosen_rewards.device, dtype=chosen_rewards.dtype
        )
        _beta = getattr(self.args, 'beta', None)
        if _beta is not None and float(_beta) <= 0:
            raise ValueError(f'beta must be positive, got {_beta!r}')
        metrics_target_pertoken = (
            per_task_scaling.detach().cpu()
            if torch.is_tensor(per_task_scaling)
            else torch.full_like(chosen_rewards, float(per_task_scaling)).detach().cpu()
        )
        if getattr(self, '_molpo_window_enabled', False):
            _seeded = self._molpo_window_kappa.seeded(tasks, device=chosen_rewards.device)
            molpo_ok = _seeded if molpo_ok is None else molpo_ok.to(_seeded.device).reshape(-1).bool() & _seeded
        loss_molpo, losses_molpo, molpo_loss_mask = molpo_loss(
            chosen_rewards=chosen_rewards,
            chosen_loss_counts=chosen_loss_counts,
            rejected_rewards=rejected_rewards,
            rejected_loss_counts=rejected_loss_counts,
            loss_type=self.args.loss_type,
            beta=self.args.beta,
            gamma_beta_ratio=getattr(self.args, 'gamma_beta_ratio', 0.0),
            per_task_scaling=per_task_scaling,
            instance_ok=molpo_ok,
        )
        if self.args.molpo_weight > 0.0:
            eff_weight = self.args.molpo_weight
            loss = self.args.sft_weight * sft_loss + eff_weight * loss_molpo
            if reject_sft_loss is not None:
                loss = loss + float(self.args.sft_reject_weight) * reject_sft_loss
        else:
            eff_weight = 0.0
            loss = self.args.sft_weight * sft_loss
            if reject_sft_loss is not None:
                loss = loss + float(self.args.sft_reject_weight) * reject_sft_loss
        if getattr(self, '_molpo_window_enabled', False) and is_train:
            reject_numerator = (-_rl * _rc)[_rc > 0].sum() if _div == 3 else sft_numerator * 0.0
            self._molpo_window_term_sums = (sft_numerator, losses_molpo[molpo_loss_mask].sum(), reject_numerator)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        return (loss, {})

    def training_step(self, batch, batch_idx):
        if getattr(self, '_molpo_window_enabled', False):
            meta = batch.get('molpo_window')
            if meta is None:
                raise ValueError('global MolPO requires the window-aware training loader')
            if int(meta['offset']) == 0:
                if self.trainer.accumulate_grad_batches != self._molpo_window_reducer.accumulation:
                    raise ValueError('global MolPO requires fixed accumulation across all windows')
                self._molpo_window_reducer.start(
                    meta['local_denominators'],
                    window_id=int(meta['window_start']),
                    microbatches=int(meta['microbatches']),
                )
                if int(self._molpo_window_reducer.denominators[0]) == 0:
                    raise RuntimeError('global chosen SFT token count is zero; refusing step')
                self._molpo_window_kappa.begin(batch.labels.device)
        if self.scheduler:
            self.scheduler.step(cur_step=self.trainer.global_step)

        tasks = [id2task(task_id.item()) for task_id in batch.tasks]

        outputs = self.blip2model(batch)
        logits = outputs.pop("logits")
        loss = outputs.pop("loss")
        _gl = outputs.pop('gathered_logps', None)
        _gs = outputs.pop('gathered_selected', None)
        if hasattr(self.args, "train_molpo") and self.args.train_molpo:
            compute_loss_context_manager = torch.amp.autocast
            n_instances = batch['seq_lens'].numel() if 'seq_lens' in batch else batch.labels.shape[0]
            n_instances = int(batch.get('answer_correction_base_segments', n_instances))
            len_tuple = n_instances // molpo_batch_division(self.args)
            tasks = tasks[:len_tuple]
            from model.molpo import original_molpo_inputs
            with original_molpo_inputs(
                batch, logits, _gl, _gs
            ) as main_inputs, compute_loss_context_manager(
                device_type=batch.labels.device.type
            ):
                loss, metrics = self.get_total_molpo_loss(
                    logits=main_inputs['logits'],
                    labels=main_inputs['labels'],
                    molpo_labels=main_inputs['molpo_labels'],
                    instance_loss=outputs["instance_loss"],
                    tasks=tasks,
                    is_train=True,
                    position_ids=main_inputs['position_ids'],
                    molpo_ok=batch.get('molpo_ok', None),
                    gathered_logps=main_inputs['gathered_logps'],
                    gathered_selected=main_inputs['gathered_selected'],
                )
            outputs.update(metrics)

            if "graph_avg_norm" in outputs:
                graph_keys = ["graph_avg_norm", "moltoken_avg_norm"]
                for k in graph_keys:
                    avg_norm = outputs.pop(k)
                    chosen_avg_norm = avg_norm[:len_tuple]
                    reject_avg_norm = avg_norm[len_tuple:2 * len_tuple]
                    outputs[f"{k}/chosen"] = chosen_avg_norm
                    outputs[f"{k}/reject"] = reject_avg_norm
        from model.molpo import model_answer_correction
        answer_loss, answer_metrics = model_answer_correction(
            self, batch, logits=logits, gathered_logps=_gl, gathered_selected=_gs
        )
        if answer_loss is not None:
            loss = loss + answer_loss
        outputs.update(answer_metrics)
        if 'answer_correction/s4_pairs' in answer_metrics:
            if getattr(self, '_s4_window_counters', None) is None:
                from model.molpo import S4WindowCounters
                self._s4_window_counters = S4WindowCounters()
            self._s4_window_counters.add(answer_metrics)
        if getattr(self, '_molpo_window_enabled', False):
            s1, s2, s3 = self._molpo_window_term_sums
            weight = float(getattr(self.args, 'answer_correction_weight', 0.0) or 0.0)
            answer_sum = (
                answer_loss * int(batch['answer_correction_chosen_count']) / weight
                if answer_loss is not None and weight
                else s1 * 0.0
            )
            counts = count_molpo_terms(
                batch,
                division=molpo_batch_division(self.args),
                pad_token_id=self.blip2model.llm_tokenizer.pad_token_id,
            )
            term_sums = torch.stack((s1, s2, s3, answer_sum))
            loss = self._molpo_window_reducer.add(
                term_sums, counts, window_id=int(batch['molpo_window']['window_start'])
            )
            del self._molpo_window_term_sums
        self.log('train_total_loss', loss.detach(), on_step=True, batch_size=self.args.batch_size)
        return loss

    def task_specific_logging(
        self,
        outputs,
        tasks,
        mode,
        epoch_end,
        task_specific_outputs: Dict[str, Dict[str, list]],
        num_moving_samples=32,
    ):

        if mode == "train":
            assert epoch_end == False

        if (mode == "train") or not epoch_end:
            # log dataset specific losses
            new_outputs = {
                k: v for k, v in outputs.items() if v.shape != torch.Size([])
            }

            for task in tasks:
                task_specific_outputs.setdefault(
                    task, {k: [] for k in new_outputs.keys()}
                )

            for metric, v in new_outputs.items():

                for i in range(v.shape[0]):
                    if torch.isnan(v[i]):
                        continue

                    task = tasks[i]
                    task_specific_outputs[task][metric].append(v[i].item())

                    if num_moving_samples is not None and (
                        len(self.task_specific_outputs[task][metric])
                        > num_moving_samples
                    ):
                        task_specific_outputs[task][metric].pop(0)

        if mode == "train" or epoch_end:
            for task, metric_dict in task_specific_outputs.items():
                for metric, vs in metric_dict.items():
                    if vs:
                        self.log(
                            f"{mode}/{task}/{metric}",
                            sum(vs) / len(vs),
                            batch_size=len(vs),
                            sync_dist=False,
                        )

    def on_train_epoch_start(self) -> None:
        self.task_specific_outputs = {}
        # if not hasattr(self, "task_specific_chosen_reward"):
        self.task_specific_chosen_reward = {}

        self.train_list_predictions = []
        self.train_list_targets = []
        self.train_list_prompts = []
        self.train_list_tasks = []
        self.train_list_probs = []
        self.train_total_avg_loss = 0.0
        self.train_total_seen_data_size = 0

        self.trainer.train_dataloader.collate_fn.current_epoch = (
            self.trainer.current_epoch
        )

        self.log_model_parameters()

    def on_evaluation_epoch_start(self):
        self.list_logs = {
            "predictions": [],
            "targets": [],
            "tasks": [],
            "probs": [],
            "prompts": [],
            "input_mol_strings": [],
        }

        self.total_avg_loss = 0.0
        self.total_seen_data_size = 0
        # self.task_subtask_name_pairs = self.trainer.datamodule.dataset_split[
        #     "test"
        # ].task_subtask_name_pairs

        self.task_subtask_name_pairs = self.trainer.datamodule.task_subtask_name_pairs

        self.eval_dataset_losses = {
            task_subtask_pair: {"avg_loss": 0.0, "num_instances": 0}
            for task_subtask_pair in self.task_subtask_name_pairs
        }
        self.eval_task_specific_outputs = {}

        if not hasattr(self, "task_specific_chosen_reward"):
            self.task_specific_chosen_reward = {}

        self.log_model_parameters()

    def log_model_parameters(self):
        mean_params = []
        for name, param in self.state_dict().items():
            try:
                mean_val = param.float().mean()
                name += "_mean"
            except:
                mean_val = param

            mean_params.append((name, mean_val))

        if not mean_params:
            return

        current_step = self.global_step

        if self.global_rank == 0:
            for logger in self.trainer.loggers:
                if isinstance(logger, pl.loggers.TensorBoardLogger):
                    for name, val in mean_params:
                        logger.experiment.add_scalar(
                            f"parameters/{name}", val, current_step
                        )

                elif isinstance(logger, pl.loggers.WandbLogger):
                    run = logger.experiment
                    run.define_metric("parameters/*", step_metric="global_step")
                    log_dict = {f"parameters/{name}": val for name, val in mean_params}
                    log_dict["global_step"] = current_step
                    logger.experiment.log(log_dict)

                elif isinstance(logger, pl.loggers.CSVLogger):
                    for name, val in mean_params:
                        logger.log_metrics(
                            {f"parameters/{name}": val}, step=current_step
                        )

    def evaluation_step(self, batch, batch_idx, dataloader_idx, mode="val"):
        if "graph" in self.args.mol_representation:
            graphs = batch["graphs"]
            additional_graphs = batch["additional_graphs"]
            is_mol_token = batch["prompt_is_mol_token"]
        else:
            graphs = None
            additional_graphs = None
            is_mol_token = None

        gen_outputs = self.blip2model.generate(
            graphs=(graphs, additional_graphs),
            input_ids=batch.prompt_input_ids,
            attention_mask=batch.prompt_attention_mask,
            is_mol_token=is_mol_token,
            num_beams=self.num_beams,
            max_length=self.gen_max_len,
            min_length=self.min_len,
            output_attentions=self.args.log_attn_score,
        )
        gen_logits = gen_outputs.logits
        gen_labels = batch.gen_labels

        forward_outputs = self.blip2model(batch)
        forward_logits = forward_outputs.pop("logits")
        forward_labels = batch.labels
        forward_loss_dict = get_instance_loss(
            logits=forward_logits, labels=forward_labels
        )
        forward_instance_loss = forward_loss_dict["instance_loss"]
        forward_loss = forward_loss_dict["loss"]

        if self.args.eval_molpo:
            len_tuple = gen_labels.shape[0] // self.args.molpo_batch_division
            tasks = [id2task(task_id.item()) for task_id in batch.tasks][:len_tuple]

            compute_loss_context_manager = torch.amp.autocast
            with compute_loss_context_manager(device_type="cuda"):
                forward_loss, metrics = self.get_total_molpo_loss(
                    logits=forward_logits,
                    labels=batch.labels,
                    molpo_labels=batch.molpo_labels,
                    tasks=tasks,
                    instance_loss=forward_instance_loss,
                    is_train=False,
                    molpo_batch_division=self.args.molpo_batch_division,
                )

            gen_logits = gen_logits[:len_tuple]
            gen_labels = gen_labels[:len_tuple]
            forward_instance_loss = forward_instance_loss[:len_tuple]

            attentions = gen_outputs.attentions
            predictions = gen_outputs.predictions[:len_tuple]
            prompt_input_ids = batch.prompt_input_ids[:len_tuple]
            input_ids = batch.input_ids[:len_tuple]
        else:
            tasks = [id2task(task_id.item()) for task_id in batch.tasks]
            attentions = gen_outputs.attentions
            predictions = gen_outputs.predictions
            prompt_input_ids = batch.prompt_input_ids
            input_ids = batch.input_ids

        if self.args.log_attn_score:
            self.log_attn_score(
                prompt_input_ids=prompt_input_ids,
                mode=mode,
                is_mol_token=is_mol_token,
                attentions=attentions,
            )

        # address generation input and output for evaluation metric calculation
        prompts = self.blip2model.llm_tokenizer.batch_decode(
            input_ids, skip_special_tokens=False
        )
        predictions = [
            p.replace(self.blip2model.llm_tokenizer.pad_token, "") for p in predictions
        ]
        target_ids = torch.where(
            gen_labels == -100,
            self.blip2model.llm_tokenizer.pad_token_id,
            gen_labels,
        )
        targets = self.blip2model.llm_tokenizer.batch_decode(target_ids)
        targets = [
            t.replace(self.blip2model.llm_tokenizer.pad_token, "") for t in targets
        ]

        probs = convert_logit2binary_prob(
            logits=gen_logits,
            predictions=predictions,
            tokenizer=self.blip2model.llm_tokenizer,
        )
        prompts = [
            p.replace(self.blip2model.llm_tokenizer.pad_token, "") for p in prompts
        ]
        input_mol_strings = self.blip2model.llm_tokenizer.batch_decode(
            batch.input_mol_strings
        )
        input_mol_strings = [
            p.replace(self.blip2model.llm_tokenizer.pad_token, "")
            for p in input_mol_strings
        ]

        self.list_logs["predictions"].extend(predictions)
        self.list_logs["targets"].extend(targets)
        self.list_logs["tasks"].extend(tasks)
        self.list_logs["probs"].extend(probs)
        self.list_logs["prompts"].extend(prompts)
        self.list_logs["input_mol_strings"].extend(input_mol_strings)

        # address forward loss
        batch_size = input_ids.shape[0]

        new_data_weight = batch_size / (self.total_seen_data_size + batch_size)
        self.total_avg_loss += (
            forward_loss.item() - self.total_avg_loss
        ) * new_data_weight
        self.total_seen_data_size += batch_size

        for i in range(forward_instance_loss.shape[0]):
            # if i th item is nan, skip
            if forward_instance_loss[i] != forward_instance_loss[i]:
                continue
            task_subtask_pair = tasks[i]
            if task_subtask_pair not in self.eval_dataset_losses:
                self.eval_dataset_losses[task_subtask_pair] = {
                    "avg_loss": 0.0,
                    "num_instances": 0,
                }
            # calculate average loss
            self.eval_dataset_losses[task_subtask_pair][
                "avg_loss"
            ] *= self.eval_dataset_losses[task_subtask_pair]["num_instances"] / (
                self.eval_dataset_losses[task_subtask_pair]["num_instances"] + 1
            )
            self.eval_dataset_losses[task_subtask_pair][
                "avg_loss"
            ] += forward_instance_loss[i] / (
                self.eval_dataset_losses[task_subtask_pair]["num_instances"] + 1
            )

            self.eval_dataset_losses[task_subtask_pair]["num_instances"] += 1

        if self.args.eval_molpo:
            self.task_specific_logging(
                outputs=metrics,
                tasks=tasks,
                mode=mode,
                epoch_end=False,
                task_specific_outputs=self.eval_task_specific_outputs,
                num_moving_samples=None,
            )

            for k, v in self.task_specific_chosen_reward.items():
                self.log(
                    f"train/{k}/bar_reward",
                    v,
                    batch_size=self.args.batch_size,
                    sync_dist=False,
                )

        return forward_loss

    def log_attn_score(self, prompt_input_ids, mode, is_mol_token, attentions):
        num_steps = len(attentions)
        seq_lengths = prompt_input_ids.shape[1]
        all_layers_attn = [
            torch.stack(attentions[step_idx])[..., :seq_lengths]
            for step_idx in range(1, num_steps)
        ]

        # [num_steps, num_heads, max_generated_length, max_generated_length] -> [num_steps, batch_size, max_generated_length]
        full_attn_mean = torch.stack(all_layers_attn).mean(dim=(1, 3)).squeeze()

        selfies_start_token_id = 35743
        selfies_end_token_id = 35744

        selfies_mask = torch.zeros_like(prompt_input_ids, dtype=torch.bool)
        st_batch_indices, start_indices = (
            prompt_input_ids == selfies_start_token_id
        ).nonzero(as_tuple=True)
        end_batch_indices, end_indices = (
            prompt_input_ids == selfies_end_token_id
        ).nonzero(as_tuple=True)

        valid_start_mask = torch.isin(st_batch_indices, end_batch_indices)
        st_batch_indices = st_batch_indices[valid_start_mask]
        start_indices = start_indices[valid_start_mask]

        start_indices += 1
        end_indices -= 1

        seq_range = torch.arange(seq_lengths, device=prompt_input_ids.device).unsqueeze(
            0
        )

        broadcasted_start = start_indices.unsqueeze(1)
        broadcasted_end = end_indices.unsqueeze(1)

        sequence_masks = (seq_range >= broadcasted_start) & (
            seq_range <= broadcasted_end
        )
        temp_mask = torch.zeros_like(prompt_input_ids, dtype=torch.bool)
        temp_mask.scatter_add_(
            0,
            end_batch_indices.unsqueeze(1).expand(-1, seq_lengths),
            sequence_masks,
        )
        selfies_mask = selfies_mask | temp_mask

        # mol_attn_score = full_attn_mean[:, mol_token_mask]

        mol_mean_scores = []
        mol_sum_scores = []
        selfies_mean_scores = []
        selfies_sum_scores = []

        # exception for batch size 1
        if len(full_attn_mean.shape) == 2:
            full_attn_mean = full_attn_mean.unsqueeze(1)

        for i in range(prompt_input_ids.shape[0]):
            if "graph" in self.args.mol_representation:
                mol_mean_scores.append(
                    full_attn_mean[:, i, is_mol_token[i]].mean(dim=-1).mean(dim=0)
                )
                mol_sum_scores.append(
                    full_attn_mean[:, i, is_mol_token[i]].sum(dim=-1).sum(dim=0)
                )
            if "string" in self.args.mol_representation:
                selfies_mean_scores.append(
                    full_attn_mean[:, i, selfies_mask[i]].mean(dim=-1).mean(dim=0)
                )
                selfies_sum_scores.append(
                    full_attn_mean[:, i, selfies_mask[i]].sum(dim=-1).sum(dim=0)
                )
        if "graph" in self.args.mol_representation:
            mol_mean_scores = torch.stack(mol_mean_scores)
            mol_mean_scores = torch.where(
                torch.isnan(mol_mean_scores),
                torch.zeros_like(mol_mean_scores),
                mol_mean_scores,
            )
            mol_sum_scores = torch.stack(mol_sum_scores)
            mol_sum_scores = torch.where(
                torch.isnan(mol_sum_scores),
                torch.zeros_like(mol_sum_scores),
                mol_sum_scores,
            )

            self.log(
                f"{mode}/graph_attn_mean_score",
                mol_mean_scores.mean().item(),
                sync_dist=False,
                batch_size=prompt_input_ids.shape[0],
            )
            self.log(
                f"{mode}/graph_attn_sum_score",
                mol_sum_scores.mean().item(),
                sync_dist=False,
                batch_size=prompt_input_ids.shape[0],
            )

        if "string" in self.args.mol_representation:
            selfies_mean_scores = torch.stack(selfies_mean_scores)
            selfies_mean_scores = torch.where(
                torch.isnan(selfies_mean_scores),
                torch.zeros_like(selfies_mean_scores),
                selfies_mean_scores,
            )
            selfies_sum_scores = torch.stack(selfies_sum_scores)
            selfies_sum_scores = torch.where(
                torch.isnan(selfies_sum_scores),
                torch.zeros_like(selfies_sum_scores),
                selfies_sum_scores,
            )

            self.log(
                f"{mode}/selfies_attn_mean_score",
                selfies_mean_scores.mean().item(),
                sync_dist=False,
                batch_size=prompt_input_ids.shape[0],
            )
            self.log(
                f"{mode}/selfies_attn_sum_score",
                selfies_sum_scores.mean().item(),
                sync_dist=False,
                batch_size=prompt_input_ids.shape[0],
            )

    def on_evaluation_epoch_end(self, mode="val") -> None:
        print(f"\nDevice {self.device} on_evaluation_epoch_end start")

        if self.args.eval_molpo:
            self.task_specific_logging(
                outputs=None,
                tasks=None,
                mode=mode,
                epoch_end=True,
                task_specific_outputs=self.eval_task_specific_outputs,
                num_moving_samples=None,
            )

        evaluation_results, failed_cases = per_device_evaluate(
            predictions=self.list_logs["predictions"],
            targets=self.list_logs["targets"],
            tasks=self.list_logs["tasks"],
            prompts=self.list_logs["prompts"],
            input_mol_strings=self.list_logs["input_mol_strings"],
            tokenizer=self.blip2model.llm_tokenizer,
            total_task_subtask_pairs=self.task_subtask_name_pairs,
        )

        self.save_predictions(
            predictions=self.list_logs["predictions"],
            targets=self.list_logs["targets"],
            tasks=self.list_logs["tasks"],
            prompts=self.list_logs["prompts"],
            probs=self.list_logs["probs"],
            input_mol_strings=self.list_logs["input_mol_strings"],
            filename=(
                f"{self.args.mode}-step{self.global_step}-{self.global_rank}-outputs.json"
                if self.args.mode == "val"
                else f"{self.args.mode}-{self.global_rank}-outputs.json"
            ),
        )

        self.save_predictions(
            predictions=failed_cases["predictions"],
            targets=failed_cases["targets"],
            tasks=failed_cases["tasks"],
            prompts=failed_cases["prompts"],
            input_mol_strings=failed_cases["input_mol_strings"],
            filename=(
                f"{self.args.mode}-step{self.global_step}-{self.global_rank}-failed_cases.json"
                if self.args.mode == "val"
                else f"{self.args.mode}-{self.global_rank}-failed_cases.json"
            ),
        )

        self.log(
            f"{mode}/total_loss",
            self.total_avg_loss,
            sync_dist=True,
            batch_size=self.total_seen_data_size,
        )

        # evaluate classification tasks
        self.cls_task_subtask_name_pair = [
            task_subtask_pair
            for task_subtask_pair in self.task_subtask_name_pairs
            if task_subtask_pair.split("/")[0] in CLASSIFICATION_BENCHMARKS
        ]
        # sort classification task_subtask_name_pairs in alphabetical order
        self.cls_task_subtask_name_pair.sort()
        self.cls_task_subtask_name_pair_dict = {
            task_subtask_pair: idx
            for idx, task_subtask_pair in enumerate(self.cls_task_subtask_name_pair)
        }
        # get inverse of self.cls_task_subtask_name_pair_dict
        self.cls_task_subtask_name_pair_dict_inv = {
            idx: task_subtask_pair
            for task_subtask_pair, idx in self.cls_task_subtask_name_pair_dict.items()
        }
        self.num_per_device_cls = 10000
        self.per_device_cls_tensor = torch.zeros(
            size=(self.num_per_device_cls, 4), device=self.device, dtype=torch.float
        )
        non_zero_count = 0
        cls_idx = 0
        for i in range(len(self.list_logs["tasks"])):
            task_subtask_pair = self.list_logs["tasks"][i]
            if task_subtask_pair in self.cls_task_subtask_name_pair_dict.keys():
                probs = self.list_logs["probs"][i]
                label = int(
                    "True" in self.list_logs["targets"][i]
                    or "true" in self.list_logs["targets"][i]
                )
                pair_ids = self.cls_task_subtask_name_pair_dict[task_subtask_pair]
                self.per_device_cls_tensor[cls_idx] = torch.tensor(
                    [probs[0], probs[1], pair_ids, label],
                    device=self.device,
                    dtype=torch.float,
                )
                non_zero_count += 1
                cls_idx += 1

        # evaluate the other tasks
        flattened_metric_keys = []
        flattened_metric_tensors = torch.empty(size=(0, 2), device=self.device)

        # tied to order of self.task_subtask_name_pairs
        for task_subtask_pair in evaluation_results:
            for metric in evaluation_results[task_subtask_pair]:
                flattened_metric_keys.append(f"{mode}/{task_subtask_pair}/{metric}")
                metric_value = evaluation_results[task_subtask_pair][metric]
                num_instance = evaluation_results[task_subtask_pair]["num_instances"]
                metric_count_pair = [metric_value * num_instance, num_instance]

                flattened_metric_tensors = torch.cat(
                    [
                        flattened_metric_tensors,
                        torch.tensor(
                            metric_count_pair,
                            device=self.device,
                        ).unsqueeze(0),
                    ],
                    dim=0,
                )

        # tied to order of self.task_subtask_name_pairs
        for dataset in self.eval_dataset_losses.keys():
            flattened_metric_keys.append(f"{mode}/{dataset}/avg_loss")
            metric_value = self.eval_dataset_losses[dataset]["avg_loss"]
            num_instance = self.eval_dataset_losses[dataset]["num_instances"]
            metric_count_pair = [metric_value * num_instance, num_instance]
            flattened_metric_tensors = torch.cat(
                [
                    flattened_metric_tensors,
                    torch.tensor(
                        metric_count_pair,
                        device=self.device,
                    ).unsqueeze(0),
                ],
                dim=0,
            )

        assert flattened_metric_tensors.shape[0] == len(
            flattened_metric_keys
        ), f"flattened_metric_tensors.shape[0]: {flattened_metric_tensors.shape[0]}, len(flattened_metric_keys): {len(flattened_metric_keys)}"
        # prepare idx to sort the flattened_metric_keys in alphabetical order
        indexed_flattened_metric_keys = list(enumerate(flattened_metric_keys))
        # get indices to sort the flattened_metric_keys in alphabetical order
        sorted_idx = [
            idx
            for idx, key in sorted(indexed_flattened_metric_keys, key=lambda x: x[1])
        ]
        flattened_metric_keys = [flattened_metric_keys[idx] for idx in sorted_idx]
        flattened_metric_tensors = flattened_metric_tensors[sorted_idx]

        if self.trainer.world_size > 1:
            print("gather the metrics across devices")
            raw_gathered_flattened_metric_tensors = self.all_gather(
                flattened_metric_tensors
            )  # [world_size, num_metrics, metric_value * per_device_instance_count, per_device_instance_count]

            # get rid of nan values (nan )
            gathered_flattened_metric_tensors = torch.where(
                torch.isnan(raw_gathered_flattened_metric_tensors),
                torch.zeros_like(raw_gathered_flattened_metric_tensors),
                raw_gathered_flattened_metric_tensors,
            )
            scaled_flattened_metric_tensors = gathered_flattened_metric_tensors[
                :, :, 0
            ].sum(dim=0)

            # total_instance_count = gathered_flattened_metric_tensors[:, :, 1].sum(dim=0)
            total_instance_count = torch.where(
                torch.isnan(gathered_flattened_metric_tensors[:, :, 0]),
                torch.zeros_like(gathered_flattened_metric_tensors[:, :, 1]),
                gathered_flattened_metric_tensors[:, :, 1],
            ).sum(dim=0)
            total_instance_count_include_nan = gathered_flattened_metric_tensors[
                :, :, 1
            ].sum(dim=0)

            gathered_cls_tensor = self.all_gather(self.per_device_cls_tensor)
            uniform_cls_tensor = torch.cat(
                [cls_tensor for cls_tensor in gathered_cls_tensor], dim=0
            )
        else:
            scaled_flattened_metric_tensors = flattened_metric_tensors[:, 0]
            total_instance_count = flattened_metric_tensors[:, 1]
            total_instance_count_include_nan = total_instance_count

            uniform_cls_tensor = self.per_device_cls_tensor

        # if total_instance_count is 0, set the metric to null value
        averaged_flattened_metric_tensors = torch.where(
            total_instance_count > 0,
            scaled_flattened_metric_tensors / total_instance_count,
            torch.tensor(float("nan"), device=self.device),
        )

        # evaluate classification tasks
        # get total_cls_tensor only where total_cls_tensor[:, :2].sum(-1) > 0
        actual_cls_tensor = uniform_cls_tensor[uniform_cls_tensor[:, :2].sum(-1) > 0]

        total_probs = actual_cls_tensor[:, :2].cpu()
        total_labels = actual_cls_tensor[:, 3].cpu().to(torch.long)
        tasks_subtask_idx = actual_cls_tensor[:, 2].to(torch.int32).tolist()
        # get task names using self.cls_task_subtask_name_pair_dict_inv
        total_tasks = [
            self.cls_task_subtask_name_pair_dict_inv[idx] for idx in tasks_subtask_idx
        ]
        classification_evaluation_result = total_device_evaluate(
            total_labels=total_labels,
            total_probs=total_probs,
            total_tasks=total_tasks,
            classification_task_subtask_pairs=self.cls_task_subtask_name_pair,
        )

        # convert classification_evaluation_result to flattened_metric_keys and flattened_metric_tensors
        cls_flattened_metric_keys = []
        cls_flattented_metric_tensors = torch.empty(size=(0, 1), device=self.device)
        for task_subtask_pair in classification_evaluation_result:
            for metric in classification_evaluation_result[task_subtask_pair]:
                cls_flattened_metric_keys.append(f"{mode}/{task_subtask_pair}/{metric}")
                metric_value = classification_evaluation_result[task_subtask_pair][
                    metric
                ]

                cls_flattented_metric_tensors = torch.cat(
                    [
                        cls_flattented_metric_tensors,
                        torch.tensor(
                            [metric_value],
                            device=self.device,
                        ).unsqueeze(0),
                    ],
                    dim=0,
                )
        cls_flattented_metric_tensors = cls_flattented_metric_tensors.squeeze(-1)
        flattened_metric_keys += cls_flattened_metric_keys
        averaged_flattened_metric_tensors = torch.cat(
            [averaged_flattened_metric_tensors, cls_flattented_metric_tensors], dim=0
        )

        # sort flattened_metric_keys in alphabetical order
        indexed_flattened_metric_keys = list(enumerate(flattened_metric_keys))
        # get indices to sort the flattened_metric_keys in alphabetical order
        sorted_idx = [
            idx
            for idx, key in sorted(indexed_flattened_metric_keys, key=lambda x: x[1])
        ]
        flattened_metric_keys = [flattened_metric_keys[idx] for idx in sorted_idx]
        averaged_flattened_metric_tensors = averaged_flattened_metric_tensors[
            sorted_idx
        ]
        print(
            "============================== Evaluation Results =============================="
        )
        for i, key in enumerate(flattened_metric_keys):
            if (
                "num_instances" in key
            ):  # num_instance here is actually mean of quadratic of num_instance
                continue
            print(f"{key}: {averaged_flattened_metric_tensors[i]} ")
            self.log(
                key,
                averaged_flattened_metric_tensors[i],
                sync_dist=False,
                rank_zero_only=True,
            )
        print(
            "================================================================================="
        )

        result_path = os.path.join(
            self.logger.log_dir,
            f"{mode}-step{self.global_step}-{self.global_rank}-results.json",
        )
        # zip the flattened metrics and averaged_flattened_metric_tensors
        result_dict = {}
        for i in range(len(flattened_metric_keys)):
            result_dict[flattened_metric_keys[i]] = averaged_flattened_metric_tensors[
                i
            ].item()
        # save result_dict in result_path
        with open(result_path, "w") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=4)

        print(f"\nDevice {self.device} on_evaluation_epoch_end end")

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None, **kwargs):

        scales = {
            "embed_tokens": getattr(self.args, "embed_pretrained_lr_scale", 1.0) or 1.0,
            "lm_head": getattr(self.args, "lm_head_pretrained_lr_scale", 1.0) or 1.0,
        }
        targets = [(tag, w, n) for tag, w, n in self._row_split_matrices()
                   if scales.get(tag, 1.0) != 1.0]
        if not targets:
            return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure, **kwargs)

        with torch.no_grad():
            snaps = [(tag, w, n, w[:n].detach().clone()) for tag, w, n in targets]
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure, **kwargs)

        interval = max(1, int(getattr(self.args, "log_every_n_steps", 50) or 50))
        measure = (self.global_step % interval == 0)
        logs = {}
        CH = 4096
        with torch.no_grad():
            for tag, weight, n_pre, snap in snaps:
                num_sq = den_sq = 0.0
                sc = scales[tag]
                for i in range(0, n_pre, CH):
                    j = min(i + CH, n_pre)
                    s = snap[i:j].float()
                    d = (weight[i:j].float() - s) * sc
                    weight[i:j] = (s + d).to(weight.dtype)
                    if measure:
                        den_sq += float(d.pow(2).sum())
                        num_sq += float((weight[i:j].float() - s).pow(2).sum())
                if measure and den_sq > 0:
                    logs[f"embed_scale/{tag}_realized"] = (num_sq / den_sq) ** 0.5
        if logs:
            self.log_dict(logs, on_step=True, on_epoch=False,
                          batch_size=self.args.batch_size, sync_dist=False)


    def on_fit_start(self):
        if getattr(self, '_molpo_window_enabled', False):

            if self.args.loss_type != 'v7':
                raise ValueError('global_molpo_terms_v1 supports only the v7 kappa-scaled objective')
            from pytorch_lightning.strategies import DDPStrategy, SingleDeviceStrategy
            from model.molpo import MolPOWindowDataLoader
            if (not self.automatic_optimization or len(self.trainer.optimizers) != 1
                    or not isinstance(self.trainer.strategy,(DDPStrategy,SingleDeviceStrategy))
                    or str(self.trainer.precision) not in ('bf16-mixed','bf16-true','32-true','64-true')):
                raise ValueError(
                    'global MolPO requires automatic ordinary DDP/single device, one optimizer, BF16/FP32/FP64'
                )
            if not isinstance(self.trainer.train_dataloader,MolPOWindowDataLoader):
                raise ValueError('global MolPO loader is missing')
            loader = self.trainer.train_dataloader
            if loader.drop_last:
                raise ValueError('global MolPO requires train_drop_last=false to preserve the global tail population')
            loader.window_batch_limit = int(self.trainer.num_training_batches)

            self._molpo_window_reducer = MolPOWindowReducer(
                accumulation=self.trainer.accumulate_grad_batches,
                coefficients=[float(self.args.sft_weight),float(self.args.molpo_weight),
                    float(getattr(self.args,'sft_reject_weight',0.) or 0.),
                    float(getattr(self.args,'answer_correction_weight',0.) or 0.)])
            self._sync_molpo_window_kappa_logs()

    def _row_split_matrices(self):
        n_new = getattr(self.blip2model, "n_added_tokens", 0)
        if not n_new:
            return []
        out = []
        for tag, getter in (
            ("embed_tokens", "get_input_embeddings"),
            ("lm_head", "get_output_embeddings"),
        ):
            try:
                module = getattr(self.blip2model.llm_model, getter)()
            except AttributeError:
                continue
            if module is None or not module.weight.requires_grad:
                continue
            out.append((tag, module.weight, module.weight.shape[0] - n_new))
        return out

    def on_before_optimizer_step(self, optimizer):
        if getattr(self, '_molpo_window_enabled', False):
            stats = self._molpo_window_reducer.finish()
            self._molpo_window_kappa.finish()
            self._sync_molpo_window_kappa_logs()
            self._last_molpo_reduction_stats = dict(stats,
                optimizer_step=int(self.trainer.global_step),kappa_updates=self._molpo_window_kappa.updates)
            for i, term in enumerate(('s1','s2','s3','s4')):
                for key in ('sums','denominators','means','weighted_terms'):
                    self.log(f'train/window_{term}_{key}',stats[key][i],on_step=True,
                             on_epoch=False,sync_dist=False,batch_size=1)

            from model.molpo import S4_POSITIONAL, S4WindowCounters, s4_scoring_policy
            if (float(getattr(self.args, 'answer_correction_weight', 0.) or 0.)
                    and s4_scoring_policy(self.args) == S4_POSITIONAL):
                if getattr(self, '_s4_window_counters', None) is None:
                    self._s4_window_counters = S4WindowCounters()
                s4 = self._s4_window_counters.finish(device=self.device)
                self._last_s4_window_counters = s4
                for name, value in s4.items():
                    self.log(f'train/window_s4/{name}', value, on_step=True, on_epoch=False,
                             sync_dist=False, batch_size=1)
            return


    def _sync_molpo_window_kappa_logs(self):
        values = dict(self._molpo_window_kappa.values)
        self._task_chosen_logps_ema = values
        self._task_chosen_ema = {t:float(self.args.beta)*v for t,v in values.items()}

def check_model_parameters(model, keyword):
    from collections import OrderedDict

    # save as ordered dict
    trainable_params_dict = OrderedDict()
    for name, param in model.named_parameters():
        if param.requires_grad and keyword in name:
            trainable_params_dict[name] = param
    return trainable_params_dict


def get_instance_loss(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    shift_logits = shift_logits.view(-1, logits.shape[-1])
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
    loss = (
        loss_not_reduced * instance_non_pad_tokens
    ).sum() / instance_non_pad_tokens.sum()
    return {"loss": loss, "instance_loss": instance_loss}


import torch
from typing import Tuple
from torch.nn import functional as F


def molpo_loss(
    chosen_rewards: torch.FloatTensor,
    chosen_loss_counts: torch.LongTensor,
    rejected_rewards: torch.FloatTensor,
    rejected_loss_counts: torch.LongTensor,
    loss_type="sigmoid",
    beta=1.0,
    gamma_beta_ratio=0.0,
    per_task_scaling=None,




    instance_ok=None,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the molpo loss for a batch of policy model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the molpo loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    # calculate molpo loss
    margin = chosen_rewards - rejected_rewards


    logits = margin - beta * gamma_beta_ratio
    if loss_type == "sigmoid":
        losses = -F.logsigmoid(beta * logits)
    elif loss_type == "v7":


        if per_task_scaling is None:
            raise ValueError(
                "v7 needs `per_task_scaling`; there is deliberately no fallback to "
                "`gamma_beta_ratio`. The two are different quantities and a silent substitution "
                "trains the task toward the wrong margin size."
            )
        s = per_task_scaling
        if torch.is_tensor(s):
            assert bool((s.abs() > 1e-6).all()), f"{loss_type} needs a nonzero task scale for every instance"
        else:
            assert abs(float(s)) > 1e-6, f"{loss_type} needs a nonzero task scale, got {s}"
        losses = s * (-F.logsigmoid(margin / s))
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. Should be one of "
            f"['sigmoid', 'v7']. "
        )
    loss_mask = torch.where(
        (rejected_loss_counts > 0) & (chosen_loss_counts > 0),
        True,
        False,
    )


    if instance_ok is not None:
        ok = instance_ok.to(device=loss_mask.device)
        if ok.shape != loss_mask.shape:
            raise ValueError(
                f"molpo_ok has {tuple(ok.shape)} entries for {tuple(loss_mask.shape)} pairs"
            )
        loss_mask = loss_mask & ok.bool()
    if not bool(loss_mask.any()):


        return losses.sum() * 0.0, losses, loss_mask
    loss = losses[loss_mask].mean()


    return loss, losses, loss_mask

def segment_ids(position_ids):
    from model.molpo_shared_prefix import get_active_segments

    _explicit = get_active_segments()
    if _explicit is not None:
        return _explicit
    if position_ids is None or position_ids.size(0) != 1:
        return None
    step = position_ids[0, 1:] - position_ids[0, :-1]
    if not bool((step < 0).any()):
        return None


    back = position_ids[0, 1:][step < 0]
    if bool((back != 0).any()):
        raise ValueError(
            "position_ids step backwards to a non-zero value, so the packed segments cannot be "
            "derived from them. This is the shared-prefix layout; it needs an explicit segment "
            "tensor, which is not wired yet."
        )
    return torch.cumsum((position_ids[0] == 0).long(), dim=0) - 1

def instance_label_counts(labels, position_ids=None):
    mask = labels[:, 1:] != -100
    seg = segment_ids(position_ids)
    if seg is None:
        return mask.sum(-1)
    index = seg[1:][mask[0]]
    n_instances = int(seg[-1]) + 1
    counts = torch.zeros(n_instances, dtype=torch.long, device=labels.device)
    counts.index_add_(0, index, torch.ones_like(index))
    return counts


def shifted_per_token_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    label_pad_token_id: int = -100,
    is_encoder_decoder: bool = False,
) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
    if logits.shape[:-1] != labels.shape:
        raise ValueError(
            "Logits (batch and sequence length dim) and labels must have the same shape."
        )
    if not is_encoder_decoder:
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id
    labels[labels == label_pad_token_id] = 0
    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)
    return (per_token_logps, loss_mask)


def reduce_logps(
    per_token_logps: torch.FloatTensor,
    loss_mask: torch.BoolTensor,
    position_ids: torch.LongTensor = None,
    average_log_prob: bool = True,
) -> Tuple[torch.FloatTensor, torch.LongTensor]:
    seg = segment_ids(position_ids)
    if seg is not None:


        index = seg[1:][loss_mask[0]]
        n_instances = int(seg[-1]) + 1
        sums = torch.zeros(
            n_instances, dtype=per_token_logps.dtype, device=per_token_logps.device
        )
        denom = torch.zeros_like(sums)
        sums.index_add_(0, index, per_token_logps[0][loss_mask[0]])
        denom.index_add_(0, index, torch.ones_like(index, dtype=sums.dtype))


        counts = torch.zeros(n_instances, dtype=torch.long, device=per_token_logps.device)
        counts.index_add_(0, index, torch.ones_like(index))
        if not average_log_prob:
            return sums, counts

        return sums / (denom + (denom == 0) * 1e-6), counts

    counts = loss_mask.sum(-1)
    target_truncation_mask = torch.where(counts > 0, True, False)


    numerically_stable_mask = counts + ~target_truncation_mask * 1e-6

    summed = (per_token_logps * loss_mask).sum(-1)
    if average_log_prob:
        return summed / numerically_stable_mask, counts
    else:
        return summed, counts

def get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = True,
    label_pad_token_id: int = -100,
    is_encoder_decoder: bool = False,
    position_ids: torch.LongTensor = None,
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
        label_pad_token_id: The label pad token id.
        is_encoder_decoder: Whether the model is an encoder-decoder model.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    per_token_logps, loss_mask = shifted_per_token_logps(
        logits, labels, label_pad_token_id, is_encoder_decoder
    )
    logps, _ = reduce_logps(per_token_logps, loss_mask, position_ids, average_log_prob)
    return logps

def split_chosen_rejected(x: torch.Tensor, division: int = 2):
    n = x.shape[0] // division


    return x[:n], x[n : 2 * n]

def reduce_logps_gathered(
    flat_logps: torch.FloatTensor,
    selected: torch.BoolTensor,
    position_ids: torch.LongTensor = None,
    average_log_prob: bool = True,
    sub_mask: torch.BoolTensor = None,
):
    seg = segment_ids(position_ids)
    if seg is not None:
        index = seg[1:][selected[0]]
        n_instances = int(seg[-1]) + 1
    else:
        rows = torch.arange(selected.size(0), device=selected.device)
        index = rows.unsqueeze(1).expand_as(selected)[selected]
        n_instances = selected.size(0)

    vals = flat_logps
    if sub_mask is not None:
        keep = sub_mask[selected]
        index = index[keep]
        vals = vals[keep]

    sums = torch.zeros(n_instances, dtype=vals.dtype, device=vals.device)
    sums.index_add_(0, index, vals)


    counts = torch.zeros(n_instances, dtype=torch.long, device=vals.device)
    counts.index_add_(0, index, torch.ones_like(index))
    if not average_log_prob:
        return sums, counts
    denom = counts.to(sums.dtype).clamp(min=1)
    return sums / denom, counts

def concatenated_forward(
    all_logits: torch.FloatTensor,
    all_labels: torch.LongTensor,
    instance_loss: torch.FloatTensor = None,
    label_pad_token_id: int = -100,
    position_ids: torch.LongTensor = None,
    gathered_logps: torch.FloatTensor = None,
    gathered_selected: torch.BoolTensor = None,
    division: int = 2,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

    We do this to avoid doing two forward passes, because it's faster for FSDP.
    """
    if gathered_logps is not None:
        per_token_logps = None
        loss_mask = None


        all_logps, all_counts = reduce_logps_gathered(
            gathered_logps,
            gathered_selected,
            position_ids,
            average_log_prob=True,
            sub_mask=all_labels[:, 1:] != label_pad_token_id,
        )
    else:
        per_token_logps, loss_mask = shifted_per_token_logps(
            all_logits, all_labels, label_pad_token_id
        )


        all_logps, all_counts = reduce_logps(
            per_token_logps, loss_mask, position_ids, average_log_prob=True
        )
    len_tuple = all_logps.shape[0] // division

    sft_instance_loss = instance_loss[:len_tuple]

    chosen_logps, rejected_logps = split_chosen_rejected(all_logps, division)
    chosen_loss_counts, rejected_loss_counts = split_chosen_rejected(all_counts, division)

    out_dict = {
        "sft_instance_loss": sft_instance_loss,
        "chosen_logps": chosen_logps,
        "chosen_loss_counts": chosen_loss_counts,
        "rejected_logps": rejected_logps,
        "rejected_loss_counts": rejected_loss_counts,
    }

    if division == 3:


        out_dict["reject_own_logps"] = all_logps[2 * len_tuple : 3 * len_tuple]
        out_dict["reject_own_counts"] = all_counts[2 * len_tuple : 3 * len_tuple]

    return out_dict

def _with_shared_prefix_plan(fn):
    def inner(self, batch, *a, **kw):
        from model.molpo_shared_prefix import ensure_hook, plan_from_batch, set_active_plan
        plan = seg = None
        if getattr(self.args, "molpo_share_prefix", False):
            plan, seg = plan_from_batch(batch, self.device)
            if plan is not None:
                plan.validate(int(batch["input_ids"].shape[-1]))
                ensure_hook()
        set_active_plan(plan, seg)
        try:
            return fn(self, batch, *a, **kw)
        finally:
            set_active_plan(None, None)
    return inner
Blip2Stage3.training_step = _with_shared_prefix_plan(Blip2Stage3.training_step)
Blip2Stage3.evaluation_step = _with_shared_prefix_plan(Blip2Stage3.evaluation_step)
