from collections import Counter

import numpy as np

from ..selection import global_records
from ..train_pool import forward_key


def _reagent_parts(row):
    answer = forward_key(row['target_text'])
    if answer is None:
        raise ValueError('Unparseable reagent answer')
    marker = row['prompt_text'].find('|>>|')
    reactant = forward_key(row['input_mol_string'])
    product = forward_key(row['prompt_text'][marker + 4 :]) if marker >= 0 else None
    if reactant is None or product is None:
        raise ValueError('Unparseable reagent reaction')
    return reactant + '>>' + product, frozenset(answer.split('.'))


def _qualifying(golds, neighbors, similarities):
    result = []
    for anchor, candidates, scores in zip(golds, neighbors, similarities):
        counts = Counter(
            component for donor in candidates[:16] for component in golds[int(donor)]
        )
        valid = []
        for donor, similarity in zip(candidates, scores):
            donor = int(donor)
            added, removed = golds[donor] - anchor, anchor - golds[donor]
            if not (added or removed) or len(added) > 1 or len(removed) > 1:
                continue
            if all(1 - counts[c] / 16 >= 0.8 for c in added) and all(
                counts[c] / 16 >= 0.8 for c in removed
            ):
                valid.append((donor, float(similarity)))
        result.append(valid)
    return result


def _neighbors(reactions):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    if len(reactions) <= 32:
        raise ValueError('Reagent neighbor search requires more than 32 TRAIN rows')
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = []
    for reaction in reactions:
        vector = np.concatenate(
            [
                generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(smiles))
                for smiles in reaction.split('>>')
            ]
        )
        fingerprint = DataStructs.ExplicitBitVect(4096)
        fingerprint.SetBitsFromList(np.flatnonzero(vector).tolist())
        fingerprints.append(fingerprint)
    keys = np.asarray(reactions)
    neighbors, similarities = [], []
    for index, fingerprint in enumerate(fingerprints):
        scores = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints),
            dtype=np.float32,
        )
        scores[keys == keys[index]] = -1.0
        order = np.argpartition(-scores, 32)[:32]
        order = order[np.lexsort((order, -scores[order]))]
        neighbors.append(order.astype(np.int32))
        similarities.append(scores[order])
    return np.asarray(neighbors), np.asarray(similarities)


def select_reagent(inputs):
    rows, indices = inputs['rows'], inputs['indices']
    parsed = [_reagent_parts(row) for row in rows]
    reactions, golds = zip(*parsed)
    if 'c0_neighbors' in inputs:
        neighbors = inputs['c0_neighbors'], inputs['c0_neighbor_similarities']
    else:
        neighbors = _neighbors(reactions)
    initial = _qualifying(golds, *neighbors)
    if 'hybrid_neighbors' in inputs:
        neighbors = inputs['hybrid_neighbors'], inputs['hybrid_neighbor_similarities']
    later = _qualifying(golds, *neighbors)
    covered = inputs['near1edit_covered']
    if len(covered) != len(rows) or len(inputs['baseline']) != len(rows):
        raise ValueError('Reagent intermediate inputs differ from TRAIN population')
    selected = []
    for index, baseline in enumerate(inputs['baseline']):
        entry = dict(baseline)
        if int(entry['anchor_index']) != index:
            raise ValueError('Baseline reagent assignments do not follow TRAIN order')
        if bool(initial[index]) != bool(covered[index]):
            raise ValueError('Original near1edit neighbors differ from pinned coverage')
        if not covered[index]:
            selected.append(entry)
            continue
        donor, similarity = initial[index][0]
        if reactions[donor] == reactions[index]:
            raise ValueError('Near1edit donor has the anchor reaction')
        donors = [donor]
        scores = [similarity]
        seen = {reactions[donor]}
        for candidate, score in later[index]:
            if reactions[candidate] in seen:
                continue
            if reactions[candidate] == reactions[index]:
                raise ValueError('Hybrid donor has the anchor reaction')
            seen.add(reactions[candidate])
            donors.append(candidate)
            scores.append(score)
            if len(donors) == 3:
                break
        while len(donors) < 3:
            donors.append(donor)
            scores.append(similarity)
        entry.update(
            donor_indices=donors,
            similarities=scores,
            answer_index=donor,
            answer_weight=1.0,
            answer_type='donor_answer_rejection',
            recipe='donor',
        )
        selected.append(entry)
    return global_records(selected, indices)
