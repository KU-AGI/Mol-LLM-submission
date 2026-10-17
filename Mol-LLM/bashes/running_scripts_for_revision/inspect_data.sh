#!/usr/bin/env bash
set -euo pipefail
: "${RELEASE:?Set RELEASE to the downloaded data and checkpoints directory}"
exec "${PYTHON:-python}" - "$RELEASE" <<'PY'
import sys
from pathlib import Path
import pyarrow.parquet as pq

path = Path(sys.argv[1]) / 'data/train/017-moleculeace-CHEMBL1862_Ki.parquet'
row = next(pq.ParquetFile(path).iter_batches(batch_size=1)).to_pylist()[0]
print('Task:', row['task'])
print('Answer:', row['target_text'])
print('Input graph nodes:', len(row['x']))
print('Paired graph nodes:', [len(row[f'{k}-th_rejected_x']) for k in range(3)])
print('Paired answers:', [row[f'{k}-th_rejected_target_text'] for k in range(3)])
PY
