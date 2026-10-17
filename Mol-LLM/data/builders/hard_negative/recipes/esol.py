from ..selection import global_records, number, _canonical, _result, _assigned


def select_esol(inputs):
    rows = inputs['rows']
    canonical = [_canonical(row['input_mol_string']) for row in rows]
    values = {}
    for anchor in inputs['anchors']:
        i = anchor['anchor_index']
        values.setdefault(canonical[i], number(rows[i]['target_text']))
    mean = sum(values.values()) / len(values)
    threshold = 0.25 * (sum((y - mean) ** 2 for y in values.values()) / len(values)) ** 0.5
    selected = []
    for anchor in inputs['anchors']:
        i = anchor['anchor_index']
        donors, similarities = _assigned(anchor)
        active = canonical[donors[0]] != canonical[i] and abs(number(rows[donors[0]]['target_text']) - number(rows[i]['target_text'])) >= threshold
        selected.append(_result(i, donors, anchor['record_similarities'], donors[0] if active else None, 'donor'))
    return global_records(selected, inputs['indices'])
