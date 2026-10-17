from ..selection import TypedAnswers, global_records, _canonical, _result, _assigned


def select_lipo(inputs):
    rows = inputs['rows']
    canonical = [_canonical(row['input_mol_string']) for row in rows]
    valid = TypedAnswers(rows, canonical)
    selected = []
    for anchor in inputs['anchors']:
        i = anchor['anchor_index']
        donors, similarities = _assigned(anchor)
        active = valid(i, donors[0]) and inputs['accept_answer'](i, donors[0])
        selected.append(_result(i, donors, similarities, donors[0] if active else None, 'donor'))
    return global_records(selected, inputs['indices'])
