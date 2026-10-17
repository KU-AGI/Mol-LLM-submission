"""Load and fingerprint explicit dataset partitions without opening sibling splits."""

import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]


def _file(root: Path, name: str) -> Path:
    path = Path(name)
    if not name or path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Invalid partition filename: {name}')
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError(f'Partition file escapes its directory or is missing: {name}')
    return resolved


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _partition(path: PathLike, split: Optional[str]) -> Path:
    root = Path(path)
    if (root / 'dataset_dict.json').exists():
        metadata = json.loads(_file(root, 'dataset_dict.json').read_text())
        if split is None or split not in metadata.get('splits', []):
            raise ValueError('A DatasetDict root requires an explicit listed split')
        if Path(split).name != split or split in ('.', '..'):
            raise ValueError('Invalid split name')
        selected = (root / split).resolve()
        if not selected.is_relative_to(root.resolve()) or not selected.is_dir():
            raise ValueError('Split escapes dataset root or is missing')
        return selected
    if split is not None:
        raise ValueError('split is only valid for a DatasetDict root')
    return root


def _manifest(root: Path):
    if not (root / 'data.json').exists():
        return None
    manifest = json.loads(_file(root, 'data.json').read_text())
    if manifest.get('format') != 'parquet' or not isinstance(manifest.get('files'), list):
        raise ValueError('Invalid portable parquet manifest')
    import pyarrow.parquet as parquet

    files = []
    seen = set()
    rows = 0
    for item in manifest['files']:
        name = item['path']
        file = _file(root, name)
        if file in seen or file.suffix != '.parquet':
            raise ValueError('Duplicate or non-parquet manifest entry')
        seen.add(file)
        if not isinstance(item['rows'], int) or item['rows'] < 0:
            raise ValueError('Invalid partition row count')
        if file.stat().st_size != item['bytes'] or _hash(file) != item['sha256']:
            raise ValueError(f'Partition file checksum or size mismatch: {name}')
        if parquet.read_metadata(file).num_rows != item['rows']:
            raise ValueError(f'Parquet row count mismatch: {name}')
        rows += item['rows']
        files.append(file)
    if not files or rows != manifest.get('rows'):
        raise ValueError('Partition manifest row total mismatch or empty files')
    return manifest, files


def load_partition(path: PathLike, split: Optional[str] = None):
    """Load one explicit HF or verified portable parquet partition."""
    from datasets import Dataset, load_dataset, load_from_disk

    root = _partition(path, split)
    if (root / 'dataset_dict.json').exists():
        raise ValueError('Nested DatasetDict is not a partition')
    portable = _manifest(root)
    if portable is not None:
        manifest, files = portable
        dataset = load_dataset('parquet', data_files={'partition': [str(file) for file in files]},
                               split='partition', cache_dir=os.environ.get('MOLLLM_DATASET_CACHE'))
        if len(dataset) != manifest['rows']:
            raise ValueError('Loaded row count does not match partition manifest')
        return dataset
    state = json.loads(_file(root, 'state.json').read_text())
    if state.get('_indices_data_files'):
        raise ValueError('Index-backed HF partitions must be flattened before loading')
    _file(root, 'dataset_info.json')
    for item in state['_data_files']:
        _file(root, item['filename'])
    dataset = load_from_disk(str(root))
    if not isinstance(dataset, Dataset):
        raise ValueError('Expected a Dataset partition')
    return dataset


def partition_digest(path: PathLike) -> str:
    """Hash partition data and schema/state files independently of their absolute root."""
    root = Path(path)
    if (root / 'dataset_dict.json').exists():
        raise ValueError('Fingerprint an explicit partition, not a DatasetDict root')
    portable = _manifest(root)
    if portable is not None:
        manifest, files = portable
        names = ['data.json'] + [item['path'] for item in manifest['files']]
        verified_hashes = {item['path']: item['sha256'] for item in manifest['files']}
    else:
        verified_hashes = {}
        state = json.loads(_file(root, 'state.json').read_text())
        if state.get('_indices_data_files'):
            raise ValueError('Index-backed HF partitions must be flattened before fingerprinting')
        names = ['state.json', 'dataset_info.json'] + [item['filename'] for item in state['_data_files']]
    digest = hashlib.sha256()
    for name in names:
        file = _file(root, name)
        encoded = name.encode('utf-8')
        digest.update(len(encoded).to_bytes(8, 'big'))
        digest.update(encoded)
        digest.update(file.stat().st_size.to_bytes(8, 'big'))
        digest.update(bytes.fromhex(verified_hashes[name] if name in verified_hashes else _hash(file)))
    return digest.hexdigest()
