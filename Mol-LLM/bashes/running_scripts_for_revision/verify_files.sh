#!/usr/bin/env bash
set -euo pipefail
: "${RELEASE:?Set RELEASE to the downloaded data and checkpoints directory}"
(cd "$RELEASE/data" && sha256sum -c SHA256SUMS)
(cd "$RELEASE/checkpoints" && sha256sum -c SHA256SUMS)
