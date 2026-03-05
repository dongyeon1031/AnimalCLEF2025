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

AB_TEST_MODE=0
for arg in "$@"; do
  if [[ "${arg}" == "--ab-test" ]]; then
    AB_TEST_MODE=1
    break
  fi
done

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
# If MATCHERS is empty and MATCHER is also empty, run all candidates by default.
MATCHERS="${MATCHERS:-}"
MATCHER="${MATCHER:-}"
CANDIDATE_SIZE="${CANDIDATE_SIZE:-25}"
TRIALS_PER_QUERY="${TRIALS_PER_QUERY:-1}"
RESULTS_DIR="${RESULTS_DIR:-experiments/local_rerank/results}"
RUN_PREFIX="${RUN_PREFIX:-local_rerank_gpu}"
VIS_PER_DATASET="${VIS_PER_DATASET:-3}"
VIS_MAX_MATCHES="${VIS_MAX_MATCHES:-120}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LOFTR_BATCH_SIZE="${LOFTR_BATCH_SIZE:-4}"

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
  if [[ "${AB_TEST_MODE}" -eq 1 ]]; then
    MATCHER_LIST=("aliked")
  else
    MATCHER_LIST=("aliked" "loftr" "orb")
  fi
fi

for m in "${MATCHER_LIST[@]}"; do
  case "${m}" in
    aliked|loftr|roma|orb) ;;
    *)
      echo "[Error] Unsupported matcher: ${m}. Use one of: aliked loftr roma orb." >&2
      exit 1
      ;;
  esac
done

echo "[Run] python=${PYTHON_BIN}, data_root=${DATA_ROOT}, device=${DEVICE}, matchers=${MATCHER_LIST[*]}"

failed=()
for m in "${MATCHER_LIST[@]}"; do
  run_prefix_for_matcher="${RUN_PREFIX}_${m}"
  batch_size_for_matcher="${BATCH_SIZE}"
  if [[ "${m}" == "loftr" ]]; then
    batch_size_for_matcher="${LOFTR_BATCH_SIZE}"
  fi
  echo "[Run] matcher=${m}, run_prefix=${run_prefix_for_matcher}, batch_size=${batch_size_for_matcher}"
  if "${PYTHON_BIN}" experiments/local_rerank/run_local_rerank.py \
    --root "${DATA_ROOT}" \
    --matcher "${m}" \
    --device "${DEVICE}" \
    --batch-size "${batch_size_for_matcher}" \
    --candidate-size "${CANDIDATE_SIZE}" \
    --trials-per-query "${TRIALS_PER_QUERY}" \
    --results-dir "${RESULTS_DIR}" \
    --run-prefix "${run_prefix_for_matcher}" \
    --visualize-per-dataset "${VIS_PER_DATASET}" \
    --visualize-max-matches "${VIS_MAX_MATCHES}" \
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
