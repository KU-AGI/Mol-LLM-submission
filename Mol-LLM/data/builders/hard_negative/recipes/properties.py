import re

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from ..selection import TypedAnswers, balanced_assign, global_records, number, rank_candidates
from ..train_pool import pool_key
from .reactions import dec, split_text


def select_properties(inputs):
    task = inputs['task']
    rows = inputs['rows']
    anchors = inputs['anchor_indices']
    occurrences = inputs['initial_occurrences']
    accept_answer = inputs['accept_answer']
    if len(occurrences) != len(anchors):
        raise ValueError('Each anchor instance requires its original occurrence offset')
    classification = task in {'bace', 'smol-property_prediction-bbbp',
                              'smol-property_prediction-clintox', 'smol-property_prediction-sider'}
    endpoints, labels, fingerprints, keys, canonical = [], [], [], [], []
    for row in rows:
        endpoints.append(pool_key(row))
        keys.append(re.sub(r'\s+', '', row['input_mol_string'].replace(
            '<SELFIES>', '').replace('</SELFIES>', '')))
        label = (float('True' in row['target_text'])
                 if classification or task.endswith('-hiv') else number(row['target_text']))
        labels.append(label)
        parts = split_text(row['input_mol_string'])
        smiles = dec(parts[1]) if parts else None
        molecule = Chem.MolFromSmiles(smiles) if smiles else None
        canonical.append(Chem.MolToSmiles(molecule) if molecule is not None else None)
        fingerprints.append(None if molecule is None else (
            AllChem.GetHashedAtomPairFingerprintAsBitVect(molecule, nBits=2048)
            if task.startswith('qm9_')
            else AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)))
    normalized = dict(inputs, endpoints=endpoints, labels=labels, fingerprints=fingerprints)
    source_indices = list(range(len(rows)))
    if classification:
        unique, members = {}, []
        for i in anchors:
            key = endpoints[i], keys[i]
            if key not in unique:
                unique[key] = len(members)
                members.append([])
            members[unique[key]].append(i)
        source_indices = [group[0] for group in members]
        position = {i: unique[endpoints[i], keys[i]] for i in range(len(rows))
                    if (endpoints[i], keys[i]) in unique}
        normalized.update(
            endpoints=[endpoints[i] for i in source_indices],
            labels=[sum(labels[i] for i in group)/len(group) for group in members],
            fingerprints=[fingerprints[i] for i in source_indices],
            anchor_indices=[position[i] for i in anchors],
            initial_donors=[[position[i] for i in row] for row in inputs['initial_donors']],
            excluded_indices={position[i] for i in inputs['excluded_indices'] if i in position})
        donors, similarities = select_classification(normalized)
        donors = np.array([
            [inputs['initial_donors'][r][slot]
             if donor == normalized['initial_donors'][r][slot] else source_indices[donor]
             for slot, donor in enumerate(row)] for r, row in enumerate(donors)])
        source_indices = list(range(len(rows)))
    elif task == 'smol-property_prediction-hiv':
        slots = 15
    elif task.startswith('qm9_'):
        slots = 3
    elif task.startswith('moleculeace-'):
        slots = 30
    else:
        raise ValueError(f'Unsupported property task: {task}')
    if not classification:
        candidates, similarities = property_candidates(normalized)
        donors, similarities = balanced_assign(candidates, similarities, slots)
        if np.any(donors < 0):
            raise ValueError('Property assignment left an unfilled donor slot')
    typed_answer = TypedAnswers(rows, canonical)
    records = []
    for instance, (anchor, row, sims) in enumerate(zip(anchors, donors, similarities)):
        for j in range(0, len(row), 3):
            selected = [source_indices[i] for i in row[j:j+3]]
            answer = inputs['initial_donors'][instance][0] if classification else selected[0]
            if task.startswith('moleculeace-'):
                from model.molpo import format_matched_rejected_answer
                same_graph = all(rows[anchor][key] == rows[answer][key]
                                 for key in ('x', 'edge_index', 'edge_attr'))
                try:
                    formatted = format_matched_rejected_answer(
                        rows[anchor]['target_text'], rows[answer]['target_text'])
                    active = (abs(labels[answer] - labels[anchor]) >= .3 - 1e-12
                              and not same_graph and canonical[anchor] != canonical[answer]
                              and number(formatted) == labels[answer]
                              and number(formatted) != labels[anchor])
                except (ValueError, TypeError):
                    active = False
            else:
                active = typed_answer(anchor, answer)
            active = active and accept_answer(anchor, answer)
            records.append(dict(
                anchor_index=int(anchor), occurrence=int(occurrences[instance]) + j//3,
                donor_indices=selected, similarities=sims[j:j+3].tolist(),
                answer_index=int(answer) if active else None,
                answer_weight=float(active),
                answer_type='donor_answer_rejection' if active else 'inactive'))
    return global_records(records, inputs['indices'])


def property_candidates(inputs):
    fingerprints = inputs['fingerprints']
    labels = np.asarray(inputs['labels'], dtype=float)
    endpoints = inputs['endpoints']
    excluded = set(inputs['excluded_indices'])
    groups = {}
    for i, endpoint in enumerate(endpoints):
        if i not in excluded and fingerprints[i] is not None and np.isfinite(labels[i]):
            groups.setdefault(endpoint, []).append(i)
    pools = {endpoint: (indices, [fingerprints[i] for i in indices],
                        labels[indices], float(np.std(labels[indices])))
             for endpoint, indices in groups.items()}
    candidates, similarities = [], []
    for anchor in inputs['anchor_indices']:
        indices, fps, values, margin = pools[endpoints[anchor]]
        differs = lambda j: (values[j] != labels[anchor] if inputs['task'] == 'smol-property_prediction-hiv'
                             else abs(values[j] - labels[anchor]) > margin)
        found = rank_candidates(fingerprints[anchor], fps, differs)
        if len(found) != 32:
            raise ValueError('The released property selection requires 32 distinct label-different candidates')
        candidates.append([indices[j] for _, j in found])
        similarities.append([sim for sim, _ in found])
    return candidates, similarities


def select_classification(inputs):
    fingerprints = inputs['fingerprints']
    labels = np.asarray(inputs['labels'], dtype=float)
    endpoints = inputs['endpoints']
    anchors = np.asarray(inputs['anchor_indices'], dtype=np.int64)
    donors = np.array(inputs['initial_donors'], dtype=np.int64, copy=True)
    similarities = np.array(inputs['initial_similarities'], dtype=float, copy=True)
    excluded = set(inputs['excluded_indices'])
    size = len(fingerprints)
    if len(labels) != size or len(endpoints) != size or donors.shape != (len(anchors), 3):
        raise ValueError('Inconsistent SAR input dimensions')
    if similarities.shape != donors.shape:
        raise ValueError('Initial donor similarities must match donor slots')
    if (np.any(anchors < 0) or np.any(anchors >= size)
            or np.any(donors < 0) or np.any(donors >= size)
            or any(i < 0 or i >= size for i in excluded)):
        raise ValueError('SAR indices must refer to the TRAIN molecule table')
    groups = {}
    for i, endpoint in enumerate(endpoints):
        groups.setdefault(endpoint, []).append(i)
    consistent = np.zeros(size, dtype=bool)
    aligned = np.zeros(size, dtype=bool)
    classes = labels >= .5
    for indices in groups.values():
        fps = [fingerprints[i] for i in indices]
        for position, i in enumerate(indices):
            sims = np.array(DataStructs.BulkTanimotoSimilarity(fingerprints[i], fps))
            sims[position] = -1
            fraction = labels[np.asarray(indices)[np.argsort(-sims)[:5]]].mean()
            consistent[i] = fraction >= .8 or fraction <= .2
            aligned[i] = classes[i] == (fraction >= .5)
    exception = consistent & ~aligned
    for row, anchor in enumerate(anchors):
        current = donors[row]
        endpoint = endpoints[anchor]
        if any(endpoints[i] != endpoint for i in current):
            raise ValueError('Initial SAR donors cross endpoints')
        if exception[anchor] or not np.any(exception[current]):
            continue
        eligible = [i for i in groups[endpoint]
                    if i not in excluded and classes[i] != classes[anchor] and not exception[i]]
        primary = [i for i in eligible if consistent[i] and aligned[i]]
        secondary = [i for i in eligible if not (consistent[i] and aligned[i])]
        candidates = []
        for tier in (primary, secondary):
            sims = np.array(DataStructs.BulkTanimotoSimilarity(
                fingerprints[anchor], [fingerprints[i] for i in tier]))
            candidates.extend((tier[j], sims[j]) for j in np.argsort(-sims))
        used = set(current)
        for slot, donor in enumerate(current.copy()):
            if not exception[donor]:
                continue
            candidate = next(((i, sim) for i, sim in candidates
                              if i not in used and i != anchor), None)
            if candidate is not None:
                donors[row, slot], similarities[row, slot] = candidate
                used.add(candidate[0])
    return donors, similarities
