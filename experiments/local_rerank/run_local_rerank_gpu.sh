#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${1:-${DATA_ROOT:-}}"
if [[ -z "${DATA_ROOT}" ]]; then
  echo "Usage: $0 <animal_clef_root> [extra args...]"
  echo "Or set DATA_ROOT=/path/to/animal-clef-2025"
  exit 1
fi
if [[ $# -gt 0 ]]; then
  shift
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
MATCHER="${MATCHER:-aliked}"
CANDIDATE_SIZE="${CANDIDATE_SIZE:-25}"
TRIALS_PER_QUERY="${TRIALS_PER_QUERY:-1}"
RESULTS_DIR="${RESULTS_DIR:-experiments/local_rerank/results}"
RUN_PREFIX="${RUN_PREFIX:-local_rerank_gpu}"

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}"
export MPLCONFIGDIR
export XDG_CACHE_HOME
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

if [[ ! -f "${DATA_ROOT}/metadata.csv" ]]; then
  echo "[Error] metadata.csv not found under: ${DATA_ROOT}" >&2
  exit 1
fi

echo "[Run] python=${PYTHON_BIN}, data_root=${DATA_ROOT}, device=cuda, matcher=${MATCHER}"
"${PYTHON_BIN}" experiments/local_rerank/run_local_rerank.py \
  --root "${DATA_ROOT}" \
  --matcher "${MATCHER}" \
  --device cuda \
  --candidate-size "${CANDIDATE_SIZE}" \
  --trials-per-query "${TRIALS_PER_QUERY}" \
  --results-dir "${RESULTS_DIR}" \
  --run-prefix "${RUN_PREFIX}" \
  "$@"
