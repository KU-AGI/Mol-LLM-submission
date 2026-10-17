from collections import Counter, defaultdict
import math
import re

import numpy as np

from .train_pool import pool_key, forward_key


def select(pool, task, inputs):
    from .train_pool import TrainPool
    from .recipes import basic, esol, lipo, properties, reactions, reagents, tdc, text
    if not isinstance(pool, TrainPool):
        raise TypeError('Selection requires a canonical TrainPool')
    tasks = pool._rows['task']
    indices = [i for i, value in enumerate(tasks) if value == task]
    if not indices:
        raise ValueError(f'Task is absent from TRAIN: {task}')
    inputs = dict(inputs, task=task, indices=indices, rows=pool._rows.select(indices))
    reaction_tasks = {'forward_reaction_prediction', 'smol-forward_synthesis',
                      'retrosynthesis', 'smol-retrosynthesis'}
    if task in reaction_tasks:
        inputs['reaction_rows'] = pool._rows.select([i for i, value in enumerate(tasks) if value in reaction_tasks])
    if task.startswith('drugood-'):
        records = basic.select_basic(inputs)
    elif task == 'smol-property_prediction-esol':
        records = esol.select_esol(inputs)
    elif task == 'smol-property_prediction-lipo':
        records = lipo.select_lipo(inputs)
    elif task.startswith('tdc-'):
        records = tdc.select_tdc(inputs)
    elif task in {'forward_reaction_prediction', 'smol-forward_synthesis'}:
        records = reactions.select_forward(inputs)
    elif task in {'retrosynthesis', 'smol-retrosynthesis'}:
        records = reactions.select_retro(inputs)
    elif task == 'reagent_prediction':
        records = reagents.select_reagent(inputs)
    elif task in {'chebi-20-mol2text', 'smol-molecule_captioning'}:
        records = text.select_caption(inputs)
    elif task in {'chebi-20-text2mol', 'smol-molecule_generation'}:
        records = text.select_text2mol(inputs)
    else:
        records = properties.select_properties(inputs)
    for entry in records:
        anchor = entry['anchor_index']
        if pool.row(anchor)['task'] != task:
            raise ValueError('Selected anchor belongs to a different task')
        for donor in entry['donor_indices']:
            if donor is not None:
                pool.check_pair(anchor, donor)
        if entry.get('answer_index') is not None:
            pool.check_answer_pair(anchor, entry['answer_index'])
        yield dict(entry, train_sha256=pool.fingerprint)


def global_records(records, indices):
    result = []
    for record in records:
        record = dict(record)
        for field in ('anchor_index', 'answer_index'):
            if record.get(field) is not None:
                index = int(record[field])
                if not 0 <= index < len(indices):
                    raise ValueError('Selection index is outside TRAIN task rows')
                record[field] = indices[index]
        donors = record['donor_indices']
        if any(d is not None and not 0 <= int(d) < len(indices) for d in donors):
            raise ValueError('Donor index is outside TRAIN task rows')
        record['donor_indices'] = [None if d is None else indices[int(d)] for d in donors]
        result.append(record)
    return result


def balanced_assign(candidates, similarities, slots):
    taken = Counter()
    donors = np.full((len(candidates), slots), -1, dtype=np.int64)
    selected = np.full(donors.shape, np.nan)
    cap = 1
    for slot in range(slots):
        remaining = list(range(len(candidates)))
        assigned = {}
        while remaining and cap <= 4096:
            users = defaultdict(list)
            for row in remaining:
                held = set(donors[row])
                for donor, similarity in zip(candidates[row], similarities[row]):
                    if donor not in held:
                        users[donor].append((similarity, row))
            for donor in sorted(users, key=lambda index: (len(users[index]), index)):
                if taken[donor] >= cap:
                    continue
                for similarity, row in sorted(users[donor], key=lambda item: -item[0]):
                    if row not in assigned:
                        assigned[row] = donor
                        taken[donor] += 1
                        selected[row, slot] = similarity
                        break
            remaining = [row for row in remaining if row not in assigned]
            if remaining:
                cap += 1
        for row, donor in assigned.items():
            donors[row, slot] = donor
    return donors, selected


def priority_assign(candidates, originals):
    unique = {}
    for donor in candidates + originals:
        unique.setdefault(int(donor['donor_local']), donor)
    pool = list(unique.values())
    return [pool[index % len(pool)] for index in range(len(originals))]


def rank_candidates(anchor_fingerprint, fingerprints, eligible, scan_limit=50000):
    from rdkit import DataStructs
    similarities = DataStructs.BulkTanimotoSimilarity(anchor_fingerprint, fingerprints)
    order = [int(i) for i in np.argsort(similarities)[::-1] if similarities[int(i)] < .999]
    scanned, candidates = [], []
    for index in order:
        if len(candidates) >= 32 or len(scanned) >= scan_limit:
            break
        if scanned and max(DataStructs.BulkTanimotoSimilarity(
                fingerprints[index], [fingerprints[i] for i in scanned])) >= .999:
            continue
        scanned.append(index)
        if eligible(index):
            candidates.append((similarities[index], index))
    candidates.sort(reverse=True)
    return candidates


def number(text):
    match = re.fullmatch(r'.*?<FLOAT>(.*?)</FLOAT>.*', text, re.S)
    if match is None:
        raise ValueError('Expected a complete FLOAT answer')
    body = re.sub(r'\s+', '', match[1])
    token = r'<\|([+\-0-9.])\|>'
    numeric = ''.join(re.findall(token, body))
    if re.sub(token, '', body) or not re.fullmatch(r'[+-]?\d+\.\d+', numeric):
        raise ValueError('Invalid FLOAT tokens')
    value = float(numeric)
    if not math.isfinite(value):
        raise ValueError('Nonfinite FLOAT answer')
    return value


def typed_value(text):
    if '<FLOAT>' in text:
        return number(text)
    match = re.fullmatch(r'.*?<BOOLEAN>(.*?)</BOOLEAN>.*', text, re.S)
    return {'True': True, 'False': False}.get(match[1].strip()) if match else None


class TypedAnswers:
    def __init__(self, rows, canonical):
        self.rows, self.canonical = rows, canonical
        self.endpoints = [pool_key(row) for row in rows]
        self.known = defaultdict(set)
        for row, endpoint, molecule in zip(rows, self.endpoints, canonical):
            value = typed_value(row['target_text'])
            if value is not None:
                self.known[endpoint, molecule].add(value)

    def __call__(self, anchor, donor):
        from model.molpo import format_matched_rejected_answer
        molecule, other = self.canonical[anchor], self.canonical[donor]
        endpoint = self.endpoints[anchor]
        if not molecule or not other or molecule == other or endpoint != self.endpoints[donor]:
            return False
        target, text = self.rows[anchor]['target_text'], self.rows[donor]['target_text']
        try:
            chosen, raw = typed_value(target), typed_value(text)
            value = typed_value(format_matched_rejected_answer(target, text))
        except (ValueError, TypeError):
            return False
        if chosen is None or raw is None or value is None or value != raw:
            return False
        known = self.known[endpoint, molecule] | {chosen}
        if '<BOOLEAN>' in target:
            return value not in known
        body = re.search(r'<FLOAT>(.*?)</FLOAT>', target, re.S)[1]
        numeric = ''.join(re.findall(r'<\|([+\-0-9.])\|>', body))
        precision = len(numeric.split('.')[1])
        return not any(value == item or float(f'{value:.{precision}f}') == float(f'{item:.{precision}f}')
                       for item in known)


def _canonical(text):
    return forward_key(text if '<SELFIES>' in text else '<SELFIES>' + text + '</SELFIES>')



def _result(anchor, donors, similarities, answer, recipe):
    return dict(anchor_index=int(anchor), occurrence=0, donor_indices=[None if d is None else int(d) for d in donors],
                similarities=[None if s is None else float(s) for s in similarities],
                answer_index=None if answer is None else int(answer), answer_weight=0.0 if answer is None else 1.0,
                answer_type='inactive' if answer is None else 'donor_answer_rejection', recipe=recipe)



def _assigned(anchor):
    originals = anchor['original_slots']
    similarities = anchor['original_similarities']
    copies = len(originals)
    existing = [dict(donor_local=originals[i][k], similarity=similarities[i][k])
                for k in range(3) for i in range(copies)]
    candidates = [dict(donor_local=d, similarity=s)
                  for d, s in zip(anchor['candidate_indices'], anchor['candidate_similarities'])]
    assigned = priority_assign(candidates, existing) if candidates else existing
    chosen = [assigned[k * copies] for k in range(3)]
    return [d['donor_local'] for d in chosen], [d['similarity'] for d in chosen]



def _scored(tokenizer, text):
    mask = set(tokenizer.molpo_mask_ids) | {tokenizer.pad_token_id}
    return tuple(t for t in tokenizer.encode(text, add_special_tokens=False) if t not in mask)
