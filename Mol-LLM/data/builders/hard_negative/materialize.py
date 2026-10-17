import gzip
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from datasets import Features, Value
import pyarrow as pa
import pyarrow.parquet as parquet

from data.storage import _file, _hash

from .train_pool import FORWARD_TASKS, TrainPool


GRAPH_COLUMNS = ('x', 'edge_index', 'edge_attr', 'additional_x',
                 'additional_edge_index', 'additional_edge_attr')
ANSWER_TYPES = frozenset(('donor_answer_rejection', 'invented_stereo_synthetic_v1', 'train_donor', 'inactive'))
TEXT_TASKS = frozenset(('chebi-20-text2mol', 'smol-molecule_generation'))


def reject_column(slot, name):
    if name.startswith('additional_'):
        return f'{slot}-th_additional_rejected_{name[11:]}'
    return f'{slot}-th_rejected_{name}'


def build_rows(pool, assignments):
    if not isinstance(pool, TrainPool):
        raise TypeError('A TrainPool is required')
    allowed = {'train_sha256', 'anchor_index', 'donor_indices', 'similarities',
               'occurrence', 'answer_index', 'answer_weight', 'answer_type', 'recipe'}
    for entry in assignments:
        if set(entry) - allowed:
            raise ValueError(f'Unknown pair fields: {sorted(set(entry) - allowed)}')
        if entry.get('train_sha256') != pool.fingerprint:
            raise ValueError('Pair table does not belong to this TRAIN partition')
        anchor_index = entry['anchor_index']
        donors, similarities = entry['donor_indices'], entry['similarities']
        if len(donors) != 3 or len(similarities) != 3:
            raise ValueError('Exactly three donors and similarities are required')
        row = dict(pool.row(anchor_index))
        row['train_src_idx'] = anchor_index
        occurrence = entry.get('occurrence', 0)
        if type(occurrence) is not int or occurrence < 0:
            raise ValueError('Occurrence must be a nonnegative integer')
        row['molpo_dup_idx'] = occurrence
        recipe = entry.get('recipe', 'donor')
        weight = float(entry.get('answer_weight', 1))
        answer_type = entry.get('answer_type', 'donor_answer_rejection')
        if not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError('Answer weight must be finite and in [0, 1]')
        if answer_type not in ANSWER_TYPES:
            raise ValueError('Unknown answer type')
        inactive = answer_type == 'inactive'
        if inactive and (weight != 0 or entry.get('answer_index') is not None):
            raise ValueError('Inactive answers require zero weight and no answer donor')
        synthetic = None
        if recipe == 'forward_stereo':
            from .recipes.reactions import transform
            if row['task'] not in FORWARD_TASKS or donors != [None] * 3 or entry.get('answer_index') is not None:
                raise ValueError('Invalid forward stereo assignment')
            if similarities != [1.0] * 3:
                raise ValueError('Forward stereo similarity must be one')
            synthetic = transform(row)
            if synthetic is None:
                raise ValueError('Anchor is ineligible for the forward stereo recipe')
            pool.check_forward_answer(row, synthetic['y_minus_text'])
            answer_text, answer_index = synthetic['y_minus_text'], -1
        elif recipe == 'text_placeholder':
            if row['task'] not in TEXT_TASKS or donors != [None] * 3 or similarities != [None] * 3:
                raise ValueError('Invalid text placeholder assignment')
            answer_index = entry['answer_index']
            if not inactive:
                _, answer = pool.check_answer_pair(anchor_index, answer_index)
                answer_text = answer['target_text']
        elif recipe == 'donor':
            answer_index = entry.get('answer_index', donors[0])
            if not inactive:
                _, answer = pool.check_answer_pair(anchor_index, answer_index)
                answer_text = answer['target_text']
        else:
            raise ValueError('Unknown recipe')
        for slot, (index, similarity) in enumerate(zip(donors, similarities)):
            if recipe == 'donor':
                _, donor = pool.check_pair(anchor_index, index)
                similarity = float(similarity)
                if not math.isfinite(similarity) or not 0 <= similarity <= 1:
                    raise ValueError('Similarity must be finite and in [0, 1]')
                target = donor['target_text']
            else:
                donor = row
                target = synthetic['y_minus_text'] if synthetic else ''
                index = -1
            for name in GRAPH_COLUMNS:
                if name in donor:
                    value = synthetic[f'g_minus_{name}'] if synthetic and not name.startswith('additional_') else donor[name]
                    row[reject_column(slot, name)] = value
            row[f'{slot}-th_rejected_target_text'] = target
            row[f'{slot}-th_rejected_src_idx'] = index
            row[f'{slot}-th_rejected_sim'] = similarity
        if inactive:
            answer_text, answer_index = '', -1
        row['v8_rejected_answer_text'] = answer_text
        row['v8_rejected_answer_src_idx'] = answer_index
        row['v8_answer_weight'] = weight
        row['v8_answer_type'] = answer_type
        yield row


def read_pairs(path, train_sha256=None):
    path = Path(path)
    if path.is_dir():
        manifest = json.loads(_file(path, 'manifest.json').read_text())
        if manifest.get('format') != 'train-pairs-v1' or not isinstance(manifest.get('files'), list):
            raise ValueError('Invalid pair directory manifest')
        fingerprint = manifest.get('train_sha256')
        if not isinstance(fingerprint, str) or (train_sha256 is not None and fingerprint != train_sha256):
            raise ValueError('Pair manifest does not belong to this TRAIN partition')
        files, seen = [], set()
        for item in manifest['files']:
            file = _file(path, item['path'])
            if file in seen or not str(file).endswith(('.jsonl', '.jsonl.gz')):
                raise ValueError('Duplicate or invalid pair file')
            seen.add(file)
            if type(item['rows']) is not int or item['rows'] < 0:
                raise ValueError('Invalid pair file row count')
            if file.stat().st_size != item['bytes'] or _hash(file) != item['sha256']:
                raise ValueError('Pair file checksum or size mismatch')
            files.append((file, item['rows']))
        if not files or type(manifest.get('rows')) is not int or sum(n for _, n in files) != manifest['rows']:
            raise ValueError('Pair manifest row total mismatch')
        for file, expected in files:
            count = 0
            for entry in read_pairs(file, fingerprint):
                count += 1
                entry.setdefault('train_sha256', fingerprint)
                yield entry
            if count != expected:
                raise ValueError('Pair file row count mismatch')
        return
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as handle:
        header = None
        seen_row = False
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get('format') == 'train-pairs-v1':
                if header is not None or seen_row or set(entry) != {'format', 'train_sha256'}:
                    raise ValueError('Invalid pair table header')
                header = entry['train_sha256']
                if train_sha256 is not None and header != train_sha256:
                    raise ValueError('Pair header does not belong to this TRAIN partition')
                continue
            if train_sha256 is not None and entry.get('train_sha256', train_sha256) != train_sha256:
                raise ValueError('Pair row does not belong to this TRAIN partition')
            fingerprint = header
            if fingerprint is not None:
                if entry.get('train_sha256', fingerprint) != fingerprint:
                    raise ValueError('Pair row overrides the header TRAIN hash')
                entry['train_sha256'] = fingerprint
            seen_row = True
            yield entry


def materialize(source, pair_table, output, workers=8, chunk_size=1024):
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('Chunk size must be a positive integer')
    pool = source if isinstance(source, TrainPool) else TrainPool.load(source, workers=workers)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    features = dict(pool._rows.features)
    features.update(train_src_idx=Value('int64'), molpo_dup_idx=Value('int64'),
                    v8_rejected_answer_text=Value('string'), v8_rejected_answer_src_idx=Value('int64'),
                    v8_answer_weight=Value('float64'), v8_answer_type=Value('string'))
    for slot in range(3):
        for name in GRAPH_COLUMNS:
            if name in features:
                features[reject_column(slot, name)] = features[name]
        features.update({f'{slot}-th_rejected_target_text': Value('string'),
                         f'{slot}-th_rejected_src_idx': Value('int64'),
                         f'{slot}-th_rejected_sim': Value('float64')})
    schema = Features(features).arrow_schema
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.molllm-build-', dir=output.parent) as temporary:
        root = Path(temporary)
        file = root / 'data.parquet'
        count, chunk = 0, []
        with parquet.ParquetWriter(file, schema, compression='zstd') as writer:
            assignments = read_pairs(pair_table, pool.fingerprint) if isinstance(pair_table, (str, Path)) else pair_table
            for row in build_rows(pool, assignments):
                chunk.append(row)
                count += 1
                if len(chunk) == chunk_size:
                    writer.write_table(pa.Table.from_pylist(chunk, schema=schema))
                    chunk.clear()
            if chunk:
                writer.write_table(pa.Table.from_pylist(chunk, schema=schema))
        manifest = dict(format='parquet', rows=count, train_sha256=pool.fingerprint,
                        source_split='train', files=[dict(path=file.name, rows=count,
                        sha256=_hash(file), bytes=file.stat().st_size)])
        (root / 'data.json').write_text(json.dumps(manifest, indent=2) + '\n')
        root.rename(output)
    return manifest
