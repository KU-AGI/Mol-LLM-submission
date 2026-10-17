import heapq
import hashlib
from collections import Counter, defaultdict

import numpy as np

from ..selection import TypedAnswers, balanced_assign, global_records, number, _canonical, _result


def _select_regression(inputs):
    rows = inputs['rows']
    canonical = [_canonical(row['input_mol_string']) for row in rows]
    y = np.asarray([number(row['target_text']) for row in rows])
    sim = np.asarray(inputs['similarities'], dtype=float)
    donors = np.array(inputs['initial_donors'], dtype=np.int64, copy=True)
    original = donors.copy()
    n = len(rows)
    edges = np.unique(np.quantile(y, (0.2, 0.4, 0.6, 0.8)))
    bins = np.searchsorted(edges, y, side='left')
    shares = np.bincount(bins, minlength=len(edges) + 1) / n
    quota = np.floor(1.25 * shares * n + 1e-9).astype(int)
    label_sd = float(np.std(y))
    sim_scale = float(np.std(sim[np.arange(n)[:, None], original]))
    gap_scale = float(np.std(np.abs(y[original] - y[:, None]) / label_sd))
    typed = TypedAnswers(rows, canonical)
    for k in range(3):
        orig = donors[:, k].copy()
        count = np.bincount(bins[donors[:, k]], minlength=len(edges) + 1)
        sim_order = np.argsort(-sim, axis=1, kind='stable')
        cache = {}
        pointers = np.zeros(n, dtype=int)

        def ranked(i):
            if i not in cache:
                gap = np.abs(y - y[i]) / label_sd
                cost = np.sqrt(((sim[i] - sim[i, orig[i]]) / sim_scale) ** 2 +
                               ((gap - gap[orig[i]]) / gap_scale) ** 2)
                tie = np.empty(n, dtype=int)
                tie[sim_order[i]] = np.arange(n)
                cache[i] = np.lexsort((tie, cost)), cost
            return cache[i]

        def best(i):
            while pointers[i] < n:
                j = int(ranked(i)[0][pointers[i]])
                if (j != i and j not in donors[i] and count[bins[j]] < quota[bins[j]] and typed(i, j)
                        and canonical[j] not in {canonical[int(d)] for d in donors[i]}):
                    return j
                pointers[i] += 1
            return None

        heap = []
        for i in range(n):
            if count[bins[donors[i, k]]] > quota[bins[donors[i, k]]]:
                j = best(i)
                if j is not None:
                    heapq.heappush(heap, (float(ranked(i)[1][j]), i, j))
        while heap and np.any(count > quota):
            cost, i, j = heapq.heappop(heap)
            old_bin = bins[donors[i, k]]
            if count[old_bin] <= quota[old_bin]:
                continue
            if count[bins[j]] >= quota[bins[j]]:
                pointers[i] += 1
                next_j = best(i)
                if next_j is not None:
                    heapq.heappush(heap, (float(ranked(i)[1][next_j]), i, next_j))
                continue
            donors[i, k] = j
            count[old_bin] -= 1
            count[bins[j]] += 1
    similarities = np.asarray(inputs['initial_similarities'], dtype=float).copy()
    changed = donors != original
    similarities[changed] = sim[np.arange(n)[:, None], donors][changed]
    selected = [_result(i, donors[i], similarities[i],
                    int(donors[i, 0]) if typed(i, int(donors[i, 0])) and inputs['accept_answer'](i, int(donors[i, 0])) else None,
                    'donor') for i in range(n)]
    return global_records(selected, inputs['indices'])


def select_tdc(inputs):
    rows = inputs['rows']
    if '<FLOAT>' in rows[0]['target_text']:
        selected = _select_regression(inputs)
        key = lambda row: hashlib.sha256('\x1f'.join(row[name] for name in
                            ('task', 'input_mol_string', 'target_text', 'prompt_text')).encode()).digest()[:16]
        by_key = defaultdict(list)
        for row, record in zip(rows, selected):
            by_key[key(row)].append(record)
        seen = Counter()
        result = []
        for anchor, occurrence in zip(inputs['destination_anchor_indices'], inputs['destination_occurrences']):
            identity = key(rows[anchor])
            record = dict(by_key[identity][seen[identity] // 10])
            seen[identity] += 1
            record.update(anchor_index=inputs['indices'][anchor], occurrence=int(occurrence))
            result.append(record)
        if any(seen[identity] != 10 * len(records) for identity, records in by_key.items()):
            raise ValueError('TDC regression source occurrences must each expand exactly ten times')
        return result
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['target_text'].split('<BOOLEAN>')[1].split('</BOOLEAN>')[0].strip()].append(i)
    groups = sorted(groups.values(), key=lambda group: -len(group))
    anchors = [group[j % len(group)] for group in groups for j in range(len(groups[0]))]
    candidates, similarities = inputs['candidates'], inputs['candidate_similarities']
    if len(candidates) != len(anchors):
        raise ValueError('Candidate cache must match the balanced TDC anchor order')
    depth = np.asarray([len(row) for row in candidates])
    if np.any(depth == 0):
        raise ValueError('TDC anchor has no real donor candidates')
    full, short = np.flatnonzero(depth >= 15), np.flatnonzero(depth < 15)
    donors = np.full((len(anchors), 15), -1, dtype=np.int64)
    sims = np.full(donors.shape, np.nan)
    if len(full):
        donors[full], sims[full] = balanced_assign([candidates[i] for i in full], [similarities[i] for i in full], 15)
    for i in short:
        for cardinal in range(15):
            donors[i, cardinal] = candidates[i][cardinal % depth[i]]
            sims[i, cardinal] = similarities[i][cardinal % depth[i]]
    canonical = [_canonical(row['input_mol_string']) for row in rows]
    valid = TypedAnswers(rows, canonical)
    occurrences, duplicates = Counter(), []
    for anchor in anchors:
        duplicates.append(occurrences[anchor])
        occurrences[anchor] += 1
    selected = []
    for copy in range(5):
        for i, anchor in enumerate(anchors):
            chosen = donors[i, 3 * copy:3 * copy + 3]
            answer = int(chosen[0])
            active = valid(anchor, answer) and inputs['accept_answer'](anchor, answer)
            record = _result(anchor, chosen, sims[i, 3 * copy:3 * copy + 3], answer if active else None, 'donor')
            record['occurrence'] = duplicates[i] * 5 + copy
            selected.append(record)
    return global_records(selected, inputs['indices'])
