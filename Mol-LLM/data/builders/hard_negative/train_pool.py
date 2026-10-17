import re

from data_utils import input_mol_string_pattern

_SIDER_TASK = 'smol-property_prediction-sider'
_SIDER_SOC_PATTERNS = (
    ('congenital', 'congenital'),
    ('neoplasms', 'neoplasm'),
    ('hepatobiliary', 'hepatobiliary'),
    ('endocrine', 'endocrine|hormones'),
    ('musculoskeletal', 'musculoskeletal|connective tissue'),
    ('skin', 'skin and subcutaneous|the skin'),
    ('pregnancy', 'pregnan|puerperium|postpartum|perinatal'),
    ('cardiac', 'cardiac|the heart|heart functions'),
    ('psychiatric', 'psychiatric|mental health'),
    ('reproductive', 'reproductive system and breast|the breasts'),
    ('metabolism', 'metabolism and nutrition'),
    ('gastrointestinal', 'gastrointestinal'),
    ('ear', 'auditory|ear disorder|the ears|ear functions'),
    ('immune', 'immune'),
    ('respiratory', 'respiratory|mediastina'),
    ('nervous', 'nervous system'),
    ('eye', 'eye disorder|the eyes|eye functions'),
    ('blood', 'blood and lymphatic|lymphatic'),
    ('vascular', 'blood vessels|vascular'),
    ('renal', 'kidney|renal|urinary'),
)
_SIDER_SOC_RE = tuple((name, re.compile(pattern, re.I)) for name, pattern in _SIDER_SOC_PATTERNS)

def _sider_soc(prompt_text):
    t = input_mol_string_pattern.sub(' ', prompt_text or '')
    for name, rx in _SIDER_SOC_RE:
        if rx.search(t):
            return name
    return None

def pool_key(row):
    task = row['task']
    if task in {'tdc-ppbr_az', 'tdc-clearance_hepatocyte_az'}:
        from instructions_smol import tdc_prompt_endpoint
        return tdc_prompt_endpoint(row)
    if task != _SIDER_TASK:
        return task
    soc = _sider_soc(row.get('prompt_text', ''))
    if soc is None:
        raise ValueError('Cannot determine SIDER assay from prompt')
    return f'{task}/{soc}'


from dataclasses import dataclass
import json
from pathlib import Path
from multiprocessing import Pool

from datasets import Dataset
from data.storage import load_partition, partition_digest



@dataclass(frozen=True, init=False)
class TrainPool:
    _rows: object
    fingerprint: str

    def __init__(self):
        raise TypeError('Use TrainPool.load with a DatasetDict directory')

    @classmethod
    def load(cls, source, workers=8):
        if type(workers) is not int or workers < 1:
            raise ValueError('Workers must be a positive integer')
        source = Path(source)
        metadata = source / 'dataset_dict.json'
        if not metadata.is_file():
            raise ValueError('Reject construction requires a DatasetDict with a train split')
        splits = json.loads(metadata.read_text()).get('splits', [])
        if 'train' not in splits:
            raise ValueError('TRAIN split is missing')
        rows = load_partition(source, split='train')
        if not isinstance(rows, Dataset):
            raise ValueError('Expected a single TRAIN partition')
        required = {'task', 'input_mol_string', 'prompt_text', 'target_text',
                    'x', 'edge_index', 'edge_attr'}
        if required - set(rows.column_names):
            raise ValueError(f'Missing TRAIN columns: {sorted(required - set(rows.column_names))}')
        for name in ('orig_split', 'split'):
            if name in rows.column_names and set(rows.unique(name)) != {'train'}:
                raise ValueError('TRAIN contains rows marked as a different split')
        if any('rejected' in name or name.startswith('v8_') for name in rows.column_names):
            raise ValueError('Expected canonical TRAIN rows without prebuilt rejects')
        additional = {'additional_x', 'additional_edge_index', 'additional_edge_attr'}
        present = additional & set(rows.column_names)
        if present and present != additional:
            raise ValueError('Incomplete additional graph columns')
        result = object.__new__(cls)
        object.__setattr__(result, '_rows', rows)
        object.__setattr__(result, '_known_forward', None)
        object.__setattr__(result, '_workers', workers)
        object.__setattr__(result, 'fingerprint', partition_digest(source / 'train'))
        return result

    def __len__(self):
        return len(self._rows)

    def row(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise ValueError(f'Donor index is outside TRAIN: {index}')
        return self._rows[index]

    def check_forward_answer(self, anchor, answer_text):
        known = self._known_forward
        if known is None:
            known = {}
            rows = self._rows.select_columns(['task', 'input_mol_string', 'target_text'])
            def collect(records):
                for record in records:
                    if record is not None:
                        key, answer = record
                        known.setdefault(key, set()).add(answer)
            if self._workers > 1 and len(rows) >= 10000:
                with Pool(self._workers) as workers:
                    collect(workers.imap(forward_record, rows, chunksize=512))
            else:
                collect(map(forward_record, rows))
            object.__setattr__(self, '_known_forward', known)
        key, answer = forward_key(anchor['input_mol_string']), forward_key(answer_text)
        if key is None or answer is None:
            raise ValueError('Forward synthetic answer cannot be decoded')
        if answer in known.get(key, ()):
            raise ValueError('Forward synthetic reject is a recorded TRAIN answer')

    def check_answer_pair(self, anchor_index, donor_index):
        anchor, donor = self.row(anchor_index), self.row(donor_index)
        if anchor_index == donor_index:
            raise ValueError('A row cannot be its own reject')
        if pool_key(anchor) != pool_key(donor):
            raise ValueError('Anchor and donor belong to different endpoints')
        if anchor['target_text'] == donor['target_text']:
            raise ValueError('Anchor and donor have identical answers')
        return anchor, donor

    def check_pair(self, anchor_index, donor_index):
        anchor, donor = self.check_answer_pair(anchor_index, donor_index)
        same_additional = all(anchor.get(name) == donor.get(name) for name in
                              ('additional_x', 'additional_edge_index', 'additional_edge_attr'))
        if anchor['input_mol_string'] == donor['input_mol_string'] and same_additional:
            raise ValueError('Anchor and donor have identical molecular inputs')
        return anchor, donor


def forward_key(text):
    from rdkit import Chem
    from .recipes.reactions import split_text, dec
    parts = split_text(text)
    smiles = dec(parts[1]) if parts else None
    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(molecule) if molecule else None


def forward_record(row):
    if row['task'] not in FORWARD_TASKS:
        return None
    anchor = forward_key(row['input_mol_string'])
    answer = forward_key(row['target_text'])
    return (anchor, answer) if anchor and answer else None


FORWARD_TASKS = frozenset(('forward_reaction_prediction', 'smol-forward_synthesis'))
