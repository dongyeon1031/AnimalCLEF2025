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
# If MATCHERS is empty and MATCHER is also empty, run all candidates by default.
MATCHERS="${MATCHERS:-}"
MATCHER="${MATCHER:-}"
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

declare -a MATCHER_LIST
if [[ -n "${MATCHERS}" ]]; then
  read -r -a MATCHER_LIST <<<"${MATCHERS}"
elif [[ -n "${MATCHER}" ]]; then
  MATCHER_LIST=("${MATCHER}")
else
  MATCHER_LIST=("aliked" "loftr" "orb")
fi

for m in "${MATCHER_LIST[@]}"; do
  case "${m}" in
    aliked|loftr|orb) ;;
    *)
      echo "[Error] Unsupported matcher: ${m}. Use one of: aliked loftr orb." >&2
      exit 1
      ;;
  esac
done

echo "[Run] python=${PYTHON_BIN}, data_root=${DATA_ROOT}, device=cuda, matchers=${MATCHER_LIST[*]}"

failed=()
for m in "${MATCHER_LIST[@]}"; do
  run_prefix_for_matcher="${RUN_PREFIX}_${m}"
  echo "[Run] matcher=${m}, run_prefix=${run_prefix_for_matcher}"
  if "${PYTHON_BIN}" experiments/local_rerank/run_local_rerank.py \
    --root "${DATA_ROOT}" \
    --matcher "${m}" \
    --device cuda \
    --candidate-size "${CANDIDATE_SIZE}" \
    --trials-per-query "${TRIALS_PER_QUERY}" \
    --results-dir "${RESULTS_DIR}" \
    --run-prefix "${run_prefix_for_matcher}" \
    "$@"; then
    echo "[Done] matcher=${m}"
  else
    failed+=("${m}")
    echo "[Warn] matcher=${m} failed. Continuing with remaining matchers." >&2
  fi
done

if [[ ${#failed[@]} -gt 0 ]]; then
  echo "[Error] Failed matchers: ${failed[*]}" >&2
  exit 1
fi

echo "[Done] All matchers finished successfully."
