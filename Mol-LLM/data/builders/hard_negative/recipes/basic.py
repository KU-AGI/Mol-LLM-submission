from collections import Counter

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from ..selection import global_records, typed_value
from ..train_pool import forward_key


def select_basic(inputs):
    rows = inputs['rows']
    smiles = [forward_key(row['input_mol_string']) for row in rows]
    labels = [typed_value(row['target_text']) for row in rows]
    first = {}
    for i, molecule in enumerate(smiles):
        if molecule is None or labels[i] is None:
            raise ValueError('Invalid TRAIN molecule or label')
        if molecule in first and labels[first[molecule]] != labels[i]:
            raise ValueError('Conflicting TRAIN labels for a molecule')
        first.setdefault(molecule, i)
    fingerprints = {s: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048)
                    for s in first}
    baseline = inputs['baseline']
    used = [Counter(smiles[entry['donor_indices'][slot]] for entry in baseline) for slot in range(3)]
    caps = [max(count.values()) for count in used]
    result = [dict(entry, donor_indices=list(entry['donor_indices']),
                   similarities=list(entry['similarities'])) for entry in baseline]
    cache = {}
    order = sorted(range(len(baseline)), key=lambda i: (smiles[baseline[i]['anchor_index']],
                   baseline[i]['occurrence'], i))
    for row in order:
        entry = result[row]
        anchor = entry['anchor_index']
        molecule = smiles[anchor]
        if molecule not in cache:
            candidates = [s for s, i in first.items() if labels[i] != labels[anchor] and s != molecule]
            similarities = DataStructs.BulkTanimotoSimilarity(
                fingerprints[molecule], [fingerprints[s] for s in candidates])
            ranking = np.lexsort((candidates, -np.asarray(similarities)))
            cache[molecule] = [(candidates[i], similarities[i]) for i in ranking]
        taken = set()
        for slot in range(3):
            old = smiles[entry['donor_indices'][slot]]
            for candidate, similarity in cache[molecule]:
                if candidate in taken or (candidate != old and used[slot][candidate] >= caps[slot]):
                    continue
                if candidate != old:
                    used[slot][old] -= 1
                    used[slot][candidate] += 1
                taken.add(candidate)
                entry['donor_indices'][slot] = first[candidate]
                entry['similarities'][slot] = float(similarity)
                break
    return global_records(result, inputs['indices'])
