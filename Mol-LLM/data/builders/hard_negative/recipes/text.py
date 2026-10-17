from collections import Counter, defaultdict
import re

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from model.molpo import format_matched_rejected_answer
from ..selection import balanced_assign, global_records, rank_candidates, _canonical, _result, _scored


def _caption_tokens(text):
    text = (text or '').replace('</s>', ' ').replace('<s>', ' ').lower()
    return [word for word in re.findall(r'[a-z0-9]+', text) if len(word) > 2]


def _rouge_l(left, right):
    if not left or not right:
        return 0.0
    previous = [0] * (len(right) + 1)
    for a in left:
        current = [0]
        for j, b in enumerate(right):
            current.append(previous[j] + 1 if a == b else max(current[j], previous[j + 1]))
        previous = current
    precision, recall = previous[-1] / len(right), previous[-1] / len(left)
    return 2 * precision * recall / (precision + recall) if previous[-1] else 0.0


def _caption_candidates(rows, canonical, anchors, candidates, similarities, excluded=()):
    excluded = set(excluded)
    tokens = [_caption_tokens(row['target_text']) for row in rows]
    pool = [i for i, words in enumerate(tokens) if words and canonical[i] is not None and i not in excluded]
    frequencies = Counter(word for i in pool for word in set(tokens[i]))
    dropped = {word for word, count in frequencies.items() if count >= 0.5 * len(pool)}
    content = [tuple(word for word in words if word not in dropped) for words in tokens]
    kept, scores = [], []
    if candidates is None:
        fingerprints = {i: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(canonical[i]), 2, 2048)
                        for i in set(pool) | set(anchors)}
        pool_fingerprints = [fingerprints[i] for i in pool]
        for anchor in anchors:
            ranked = rank_candidates(fingerprints[anchor], pool_fingerprints,
                                     lambda j: _rouge_l(content[anchor], content[pool[j]]) < .3, scan_limit=1000)
            if len(ranked) != 32:
                raise ValueError('The released caption selection requires 32 candidates within the scan limit')
            kept.append([pool[j] for _, j in ranked])
            scores.append([sim for sim, _ in ranked])
        return kept, scores
    for anchor, choices, values in zip(anchors, candidates, similarities):
        selected = [(donor, value) for donor, value in zip(choices, values)
                    if _rouge_l(content[anchor], content[donor]) < 0.3]
        kept.append([donor for donor, _ in selected])
        scores.append([value for _, value in selected])
    return kept, scores


def select_caption(inputs):
    rows, tokenizer = inputs['rows'], inputs['tokenizer']
    canonical = [_canonical(row['input_mol_string']) for row in rows]
    known = defaultdict(set)
    for row, canon in zip(rows, canonical):
        if canon is not None:
            text = format_matched_rejected_answer(row['target_text'], row['target_text'])
            known[row['task'], canon].add(_scored(tokenizer, text))
    candidates, values = _caption_candidates(rows, canonical, inputs['anchor_indices'],
                                             inputs.get('candidates'), inputs.get('candidate_similarities'),
                                             inputs['excluded_indices'])
    donors, similarities = balanced_assign(candidates, values, 3)
    if np.any(donors < 0):
        raise ValueError('Caption assignment left an unfilled donor slot')
    selected = []
    for i, anchor in enumerate(inputs['anchor_indices']):
        row = rows[anchor]
        own = _scored(tokenizer, row['target_text'])
        answer = None
        for donor in donors[i]:
            donor = int(donor)
            if donor < 0 or not (rows[donor]['target_text'] or '').strip():
                continue
            if canonical[anchor] is not None and canonical[donor] == canonical[anchor]:
                continue
            try:
                wrong = format_matched_rejected_answer(row['target_text'], rows[donor]['target_text'])
            except ValueError:
                continue
            content = _scored(tokenizer, wrong)
            if not content or content == own or content in known.get((row['task'], canonical[anchor]), ()):
                continue
            answer = donor
            break
        if answer is not None and not inputs['accept_answer'](anchor, answer):
            answer = None
        selected.append(_result(anchor, donors[i], similarities[i], answer, 'donor'))
    return global_records(selected, inputs['indices'])


def select_text2mol(inputs):
    rows, tokenizer = inputs['rows'], inputs['tokenizer']
    canonical = [_canonical(row['target_text']) for row in rows]
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048)
                    if s else None for s in canonical] if 'similarities' not in inputs else None
    known = defaultdict(set)
    for row, canon in zip(rows, canonical):
        if canon:
            known[row['task'], row['prompt_text']].add(canon)
    selected = []
    for position, anchor in enumerate(inputs['anchor_indices']):
        row = rows[anchor]
        eligible = [i for i, donor in enumerate(rows) if donor['task'] == row['task'] and canonical[i] is not None
                    and canonical[i] not in inputs['excluded_outputs'].get(row['task'], set())]
        answer = None
        if canonical[anchor] is not None:
            similarities = (np.asarray(inputs['similarities'][position])[eligible] if fingerprints is None else
                            DataStructs.BulkTanimotoSimilarity(fingerprints[anchor], [fingerprints[i] for i in eligible]))
            order = np.lexsort((np.asarray(eligible), -np.asarray(similarities)))
            bad = known[row['task'], row['prompt_text']] | {canonical[anchor]}
            for position in order[:256]:
                donor = eligible[int(position)]
                if canonical[donor] not in bad:
                    answer = donor
                    break
        if answer is not None:
            try:
                wrong = format_matched_rejected_answer(row['target_text'], rows[answer]['target_text'])
                content = _scored(tokenizer, wrong)
                distinct = bool(wrong) and bool(content) and content != _scored(tokenizer, row['target_text'])
            except ValueError:
                distinct = False
            if not distinct or not inputs['accept_answer'](anchor, answer):
                answer = None
        selected.append(_result(anchor, [None] * 3, [None] * 3, answer, 'text_placeholder'))
    return global_records(selected, inputs['indices'])
