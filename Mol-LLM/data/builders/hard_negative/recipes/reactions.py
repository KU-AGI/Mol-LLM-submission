import re
from collections.abc import Mapping
from typing import Any

import numpy as np
import selfies as sf
from rdkit import Chem

from augment_dataset import mol2graph

CW = Chem.ChiralType.CHI_TETRAHEDRAL_CW

TOK = re.compile('\\[[^\\]]*\\]|\\.')
NONATOM = set('=#/\\-()0123456789%.')
ELEM = re.compile('^\\[?(Cl|Br|Si|Se|Sn|Zn|Mg|Li|Na|Al|As|Te|[BCNOPSFIKHbcnops])')
MAXRET = 5.0

def split_text(t):
    if not isinstance(t, str) or '<SELFIES>' not in t:
        return None
    k = t.index('<SELFIES>') + len('<SELFIES>')
    m = TOK.search(t, k)
    if not m:
        return None
    j = t.index('</SELFIES>') if '</SELFIES>' in t else len(t)
    b = t[m.start():j]
    toks = TOK.findall(b)
    return (t[:m.start()], toks, b[len(''.join(toks)):] + t[j:], ' ' in b[:40])

def dec(toks):
    try:
        return sf.decoder(''.join(toks))
    except Exception:
        return None

def canon(smi, strip=False):
    if smi is None:
        return None
    o = []
    for c in smi.split('.'):
        m = Chem.MolFromSmiles(c)
        if m is None:
            return None
        if strip:
            Chem.RemoveStereochemistry(m)
        o.append(Chem.MolToSmiles(m))
    return '.'.join(sorted(o)) if o else None

def nst(smi):
    if smi is None:
        return None
    n = 0
    for c in smi.split('.'):
        m = Chem.MolFromSmiles(c)
        if m is None:
            return None
        n += sum((1 for a in m.GetAtoms() if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED))
    return n

def atom_token_index(toks):
    try:
        smi, att = sf.decoder(''.join(toks), attribute=True)
    except Exception:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    atoms = [a for a in att if a.token and a.token[0] not in NONATOM]
    if len(atoms) != m.GetNumAtoms():
        return None
    nondot = [j for j, t in enumerate(toks) if t != '.']
    out = []
    for k, a in enumerate(atoms):
        e = ELEM.match(a.token)
        if not e or m.GetAtomWithIdx(k).GetSymbol() != e.group(1).capitalize():
            return None
        if not a.attribution:
            return None
        ai = a.attribution[-1].index
        if ai >= len(nondot):
            return None
        out.append(nondot[ai])
    return (smi, out)

def retention(a, b):
    n = max(len(a), len(b))
    return 100 * sum((1 for j in range(n) if (a[j] if j < len(a) else None) != (b[j] if j < len(b) else None))) / n

def encode(smi, spaced):
    try:
        e = sf.encoder(smi)
    except Exception:
        return None
    t = TOK.findall(e)
    return (' ' if spaced else '').join(t) if t else None

def make_y_minus(gold_toks, y_smiles, spaced):
    pm = Chem.MolFromSmiles(y_smiles)
    if pm is None:
        return None
    tgt = [a.GetIdx() for a in pm.GetAtoms() if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED]
    if not tgt:
        return None
    at = atom_token_index(gold_toks)
    if at:
        smi, tmap = at
        gm = Chem.MolFromSmiles(smi)
        skel = Chem.MolFromSmiles(canon(y_smiles, strip=True) or '')
        match = gm.GetSubstructMatch(skel) if gm and skel else None
        if match and tgt[0] < len(match):
            gi = match[tgt[0]]
            if gi < len(tmap):
                ti = tmap[gi]
                tok = gold_toks[ti]
                e = ELEM.match(tok)
                if e and gm.GetAtomWithIdx(gi).GetSymbol() == e.group(1).capitalize() and ('@' not in tok):
                    new = list(gold_toks)
                    new[ti] = tok[:e.end()] + '@' + tok[e.end():]
                    cand = dec(new)
                    if cand and canon(cand, strip=True) == canon(dec(gold_toks), strip=True) and ((nst(cand) or 0) > 0):
                        return ((' ' if spaced else '').join(new), 'inplace')
    body = encode(y_smiles, spaced)
    if body is None:
        return None
    if retention(gold_toks, TOK.findall(body)) > MAXRET:
        return None
    return (body, 'reencode')

def coherent_pair(reac_text, prod_text):
    rs, ps = (split_text(reac_text), split_text(prod_text))
    if not rs or not ps:
        return None
    reac, prod = (dec(rs[1]), dec(ps[1]))
    sr, sp = (nst(reac), nst(prod))
    if sr is None or sp is None:
        return None
    if sr != 0:
        return None
    if sp != 0:
        return None
    pm = Chem.MolFromSmiles(prod)
    if pm is None:
        return None
    comps = reac.split('.')
    for ci, comp in enumerate(comps):
        rm = Chem.MolFromSmiles(comp)
        if rm is None:
            continue
        cent = [a for a, _ in Chem.FindMolChiralCenters(rm, includeUnassigned=True, useLegacyImplementation=False)]
        if not cent:
            continue
        ri = cent[0]
        if rm.GetNumBonds() == 0:
            continue
        env = Chem.FindAtomEnvironmentOfRadiusN(rm, 2, ri)
        if not env:
            continue
        patt = Chem.MolFromSmarts(Chem.MolToSmarts(Chem.PathToSubmol(rm, env)))
        if patt is None:
            continue
        hits = pm.GetSubstructMatches(patt)
        if len(hits) != 1:
            return None
        rm2 = Chem.MolFromSmiles(comp)
        rm2.GetAtomWithIdx(ri).SetChiralTag(CW)
        pm2 = Chem.MolFromSmiles(prod)
        sub = Chem.PathToSubmol(rm, env)
        try:
            rpos = list(rm.GetSubstructMatches(Chem.MolFromSmarts(Chem.MolToSmarts(sub)))[0]).index(ri)
        except Exception:
            return None
        pidx = hits[0][rpos]
        if pm2.GetAtomWithIdx(pidx).GetSymbol() != rm.GetAtomWithIdx(ri).GetSymbol():
            return None
        pm2.GetAtomWithIdx(pidx).SetChiralTag(CW)
        g_minus = '.'.join(sorted([Chem.MolToSmiles(rm2)] + [c for k, c in enumerate(comps) if k != ci]))
        y_minus = Chem.MolToSmiles(pm2)
        if nst(g_minus) == 0 or nst(y_minus) == 0:
            return None
        if canon(g_minus, strip=True) != canon(reac, strip=True):
            return None
        if canon(y_minus, strip=True) != canon(prod, strip=True):
            return None
        eg, ey = (encode(g_minus, rs[3]), encode(y_minus, ps[3]))
        if eg is None or ey is None:
            return None
        return {'g_minus_text': rs[0] + eg + rs[2], 'y_minus_text': ps[0] + ey + ps[2], 'g_minus_smiles': g_minus, 'y_minus_smiles': y_minus}
    return None


def transform(row: Mapping[str, Any]) -> dict[str, Any] | None:
    pair = coherent_pair(row['input_mol_string'], row['target_text'])
    if pair is None:
        return None
    gold = split_text(row['target_text'])
    negative = make_y_minus(gold[1], pair['y_minus_smiles'], gold[3])
    if negative is None:
        return None
    graph_mol = Chem.MolFromSmiles(pair['g_minus_smiles'])
    if graph_mol is None:
        return None
    try:
        preliminary = mol2graph(graph_mol)
    except Exception:
        return None
    if np.asarray(preliminary['node_feat']).shape[1] != 9:
        return None
    smiles = dec(split_text(row['input_mol_string'])[1])
    anchor = Chem.MolFromSmiles(smiles)
    if anchor is None:
        return None
    target = None
    offset = 0
    for component in smiles.split('.'):
        molecule = Chem.MolFromSmiles(component)
        if molecule is None:
            break
        centers = [i for i, _ in Chem.FindMolChiralCenters(molecule, includeUnassigned=True, useLegacyImplementation=False)]
        if centers:
            target = offset + centers[0]
            symbol = molecule.GetAtomWithIdx(centers[0]).GetSymbol()
            break
        offset += molecule.GetNumAtoms()
    if target is None or target >= anchor.GetNumAtoms() or anchor.GetAtomWithIdx(target).GetSymbol() != symbol:
        return None
    anchor.GetAtomWithIdx(target).SetChiralTag(CW)
    try:
        graph = mol2graph(anchor)
    except Exception:
        return None
    x = np.asarray(graph['node_feat'], dtype=np.int64)
    if x.shape != np.asarray(row['x']).shape:
        return None
    edges = np.asarray(graph['edge_index'], dtype=np.int64)
    attributes = np.asarray(graph['edge_feat'], dtype=np.int64)
    if not np.array_equal(edges, row['edge_index']) or not np.array_equal(attributes, row['edge_attr']):
        return None
    difference = np.argwhere(x != np.asarray(row['x']))
    if len(difference) == 0 or not all(int(feature) == 1 for _, feature in difference):
        return None
    body, route = negative
    return {'y_minus_text': gold[0] + body + gold[2], 'route': route,
            'g_minus_x': x.tolist(), 'g_minus_edge_index': edges.tolist(),
            'g_minus_edge_attr': attributes.tolist(), 'n_feature_cells_changed': int(len(difference))}


def _identity(text):
    parts = split_text(text)
    smiles = dec(parts[1]) if parts else None
    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    return (canon(smiles), Chem.MolToInchi(molecule)) if molecule else (None, None)


def _known_reactions(rows):
    from collections import defaultdict

    directions = {
        'forward_reaction_prediction': 'forward',
        'smol-forward_synthesis': 'forward',
        'retrosynthesis': 'retro',
        'smol-retrosynthesis': 'retro',
    }
    known = defaultdict(set)
    for row in rows:
        direction = directions.get(row['task'])
        if direction is None:
            continue
        input_key, input_inchi = _identity(row['input_mol_string'])
        answer_key, answer_inchi = _identity(row['target_text'])
        if input_key is not None and answer_key is not None:
            known[direction, input_key].add(answer_inchi)
            inverse = 'retro' if direction == 'forward' else 'forward'
            known[inverse, answer_key].add(input_inchi)
    return known


def _distinct_answer(anchor, answer, tokenizer):
    from model.molpo import format_matched_rejected_answer

    try:
        formatted = format_matched_rejected_answer(anchor['target_text'], answer)
    except ValueError:
        return False
    from ..selection import _scored

    chosen = _scored(tokenizer, anchor['target_text'])
    rejected = _scored(tokenizer, formatted)
    return bool(rejected) and formatted.strip() != anchor['target_text'].strip() and rejected != chosen


def reaction_candidates(inputs):
    from ..selection import rank_candidates
    from rdkit import DataStructs
    from rdkit.Chem import AllChem

    rows = inputs['rows']
    excluded = set(inputs['excluded_indices'])
    fingerprints, outputs = [], []
    for row in rows:
        pair = []
        for field in ('input_mol_string', 'target_text'):
            parts = split_text(row[field])
            smiles = dec(parts[1]) if parts else None
            molecule = Chem.MolFromSmiles(smiles) if smiles else None
            pair.append(
                AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)
                if molecule is not None and molecule.GetNumAtoms() else None
            )
        fingerprints.append(pair[0])
        outputs.append(pair[1])
    indices = [i for i in range(len(rows))
               if i not in excluded and fingerprints[i] is not None and outputs[i] is not None]
    fps = [fingerprints[i] for i in indices]
    candidates, similarities = [], []
    for anchor in range(len(rows)):
        if fingerprints[anchor] is None or outputs[anchor] is None:
            raise ValueError('Reaction anchor has no molecular fingerprint')
        found = rank_candidates(
            fingerprints[anchor], fps,
            lambda donor: DataStructs.TanimotoSimilarity(
                outputs[indices[donor]], outputs[anchor]
            ) < .3,
        )
        candidates.append([indices[donor] for _, donor in found])
        similarities.append([score for score, _ in found])
    return candidates, similarities


def _select_reaction(inputs, direction):
    from ..selection import balanced_assign, global_records

    rows, indices = inputs['rows'], inputs['indices']
    if 'candidates' in inputs:
        candidates, scores = inputs['candidates'], inputs['candidate_similarities']
    else:
        candidates, scores = reaction_candidates(inputs)
    donors, similarities = balanced_assign(candidates, scores, 3)
    if bool((donors < 0).any()):
        raise ValueError('Reaction candidate assignment has unfilled slots')
    known = _known_reactions(inputs['reaction_rows'])
    parsed = [(_identity(row['input_mol_string']), _identity(row['target_text'])) for row in rows]
    selected = []
    for index, row in enumerate(rows):
        local_donors = [int(d) for d in donors[index]]
        slot_similarities = [float(s) for s in similarities[index]]
        donor_index = local_donors[0]
        donor = rows[donor_index]
        input_key, answer_inchi = parsed[index][0][0], parsed[index][1][1]
        donor_input, donor_inchi = parsed[donor_index][0][0], parsed[donor_index][1][1]
        same_graph = all(np.array_equal(row[field], donor[field]) for field in ('x', 'edge_index', 'edge_attr'))
        active = (
            not same_graph and input_key is not None and donor_input != input_key
            and answer_inchi is not None and donor_inchi is not None
            and donor_inchi != answer_inchi
            and donor_inchi not in known.get((direction, input_key), ())
            and _distinct_answer(row, donor['target_text'], inputs['tokenizer'])
        )
        if active:
            try:
                active = bool(inputs['accepts'](index, local_donors, slot_similarities, donor['target_text']))
            except ValueError:
                active = False
        selected.append({
            'anchor_index': index,
            'donor_indices': local_donors,
            'similarities': slot_similarities,
            'occurrence': 0,
            'answer_index': donor_index if active else None,
            'answer_weight': float(active),
            'answer_type': 'donor_answer_rejection' if active else 'inactive',
            'recipe': 'donor',
        })
    return global_records(selected, indices)


def select_retro(inputs):
    return _select_reaction(inputs, 'retro')


def select_forward(inputs):
    from ..train_pool import forward_key, forward_record

    selected = _select_reaction(inputs, 'forward')
    known = {}
    for row in inputs['reaction_rows']:
        record = forward_record(row)
        if record is not None:
            known.setdefault(record[0], set()).add(record[1])
    for row, entry in zip(inputs['rows'], selected):
        synthetic = transform(row)
        if synthetic is None:
            continue
        key = forward_key(row['input_mol_string'])
        if forward_key(synthetic['y_minus_text']) in known.get(key, ()):
            continue
        entry.update(
            donor_indices=[None] * 3,
            similarities=[1.0] * 3,
            answer_index=None,
            answer_weight=1.0,
            answer_type='invented_stereo_synthetic_v1',
            recipe='forward_stereo',
        )
    return selected
