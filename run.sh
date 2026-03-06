#!/usr/bin/env bash
set -e
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
exec "$(dirname "$0")/.venv/bin/python" "$@"
