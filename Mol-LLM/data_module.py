# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import json
from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from datasets import load_from_disk
from data.storage import load_partition
from data_utils import DataCollator


class Stage3DM(LightningDataModule):

    def __init__(self, tokenizer, mode: str="to_be_removed", num_workers: int=0, args=None):
        super().__init__()
        self.args = args
        self.mode = mode
        self.num_workers = num_workers

        self.max_length = args.max_length
        self.batch_size = args.batch_size
        self.inference_batch_size = args.inference_batch_size

        self.mol_representation = args.mol_representation
        self.tokenizer = tokenizer

        if self.mode in ["test"]:
            self.test_dataset = _sort_by_task(get_dataset("test", tokenizer, args))
            present = set(self.test_dataset['task'])
        else:
            self.train_dataset = get_dataset("train", tokenizer, args)
            self.val_dataset = _sort_by_task(get_dataset('validation', tokenizer, args))
            present = set(self.train_dataset['task']) | set(self.val_dataset['task'])
        self.task_subtask_name_pairs = []
        self.task_subtask_name_pairs = sorted(present)
        tokenizer.padding_side = "left"
        self.train_collator = DataCollator(
            tokenizer=tokenizer,
            padding=True,
            max_length=args.max_length,
            return_tensors="pt",
            train=True,
            args=args,
        )
        val_max_length = getattr(args, 'val_max_length', None) or args.max_length
        test_max_length = getattr(args, 'test_max_length', None)
        self.eval_collator = DataCollator(tokenizer=tokenizer, padding=True, max_length=val_max_length, return_tensors='pt', train=False, args=args, truncate=True)
        self.test_collator = DataCollator(tokenizer=tokenizer, padding=True, max_length=test_max_length or args.max_length, return_tensors="pt", train=False, args=args, truncate=test_max_length is not None)

    def train_dataloader(self):
        window_mode = getattr(self.args, 'molpo_loss_reduction', None) == 'global_molpo_terms_v1'
        loader_class = DataLoader
        extra = {}
        if window_mode:
            from model.molpo import MolPOWindowDataLoader
            from data_utils import molpo_batch_division
            loader_class = MolPOWindowDataLoader
            self.train_collator.current_epoch = int(getattr(self.trainer, 'current_epoch', 0))
            extra = dict(accumulation=int(self.trainer.accumulate_grad_batches), window_batch_limit=self.trainer.limit_train_batches, molpo_division=molpo_batch_division(self.args), pad_token_id=self.tokenizer.pad_token_id)
        rows = self.train_dataset
        from data_utils import IndexedRows
        if getattr(self.args, 'train_molpo', False):
            rows = IndexedRows(self.train_dataset)
        loader = loader_class(rows, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=bool(getattr(self.args, 'train_drop_last', True)), persistent_workers=self.args.num_workers > 0 and (not window_mode), collate_fn=self.train_collator, **extra)
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True if self.args.num_workers > 0 else False,
            collate_fn=self.eval_collator,
        )
        return loader

    def test_dataloader(self):
        if not hasattr(self, 'test_dataset'):
            self.test_dataset = _sort_by_task(get_dataset('test', self.tokenizer, self.args))
            self.task_subtask_name_pairs = sorted(set(self.test_dataset['task']))
        loader = DataLoader(self.test_dataset, batch_size=self.inference_batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, drop_last=False, persistent_workers=True if self.args.num_workers > 0 else False, collate_fn=self.test_collator)
        return loader
def _sort_by_task(ds):
    import numpy as np

    tasks = np.asarray(ds["task"])
    order = np.argsort(tasks, kind="stable")
    return ds.select(order.tolist())

def get_dataset(split, tokenizer, args):
    if split not in ('train', 'validation', 'test'):
        raise ValueError(split)
    paths = getattr(args, 'split_paths', None)
    if paths is not None:
        if split not in paths:
            raise ValueError(f'Missing explicit {split} dataset path')
        path = Path(paths[split])
    else:
        path = Path(args.raw_data_root)
        if not (path / 'dataset_dict.json').is_file():
            partition = 'test' if split == 'validation' else split
            name = f"{args.llm_model.replace('/', '-')}_string+graph_q{args.num_query_token}_{partition}"
            path = path / name
            tag = getattr(args, 'data_tag', None)
            if tag is not None:
                tagged = path.with_name(f'{path.name}_{tag}')
                if tagged.exists() or not getattr(args, 'tasks', None):
                    path = tagged
    metadata = path / 'dataset_dict.json'
    if metadata.is_file():
        if split not in json.loads(metadata.read_text()).get('splits', []):
            raise ValueError(f'Missing {split} partition')
        path = path / split
    dataset = load_partition(path)
    tasks = getattr(args, 'tasks', None)
    if tasks:
        allowed = set(tasks)
        dataset = dataset.filter(lambda row: row["task"] in allowed)
    return dataset
