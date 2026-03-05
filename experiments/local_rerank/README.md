# Local Rerank-Only Experiments

This folder is isolated from the original fusion pipeline.

Goal:
- Build candidate pools of size `K` (default `25`) per query.
- Force-inject at least one ground-truth database sample into each pool.
- Rank candidates using a **local matcher only** (no MegaDescriptor/EVA fusion).
- Measure Top-1 hit rate.
- By default, candidates are sampled **within the same dataset/species only**.

## Script

- `experiments/local_rerank/run_local_rerank.py`
- `experiments/local_rerank/run_local_rerank_gpu.sh`

## GPU shortcut script

```bash
chmod +x experiments/local_rerank/run_local_rerank_gpu.sh
PYTHON_BIN=/opt/anaconda3/envs/animal_reid/bin/python \
experiments/local_rerank/run_local_rerank_gpu.sh /path/to/animal-clef-2025
```

You can override defaults with env vars:
- `MATCHERS` (space-separated list, e.g. `"aliked loftr roma orb"`)
- `MATCHER` (single matcher; used only when `MATCHERS` is empty)
- `CANDIDATE_SIZE` (default: `25`)
- `TRIALS_PER_QUERY` (default: `1`)
- `RESULTS_DIR`
- `RUN_PREFIX`

Default behavior:
- If neither `MATCHERS` nor `MATCHER` is set, `run_local_rerank_gpu.sh` runs all:
  `aliked`, `loftr`, `orb`
- Each matcher uses its own prefix: `${RUN_PREFIX}_<matcher>`
  so CSV/JSON are saved separately.
- Match-point visualization is enabled by default with:
  `VIS_PER_DATASET=3` (up to 3 good + 3 bad per dataset/species).
  Files are saved under:
  `experiments/local_rerank/results/visualizations/<run_prefix>_<matcher>/`

Visualization env vars:
- `VIS_PER_DATASET` (default: `3`, set `0` to disable)
- `VIS_MAX_MATCHES` (default: `120`)

Batch-size env vars:
- `BATCH_SIZE` (default: `64`, used for non-LoFTR matchers)
- `LOFTR_BATCH_SIZE` (default: `4`, used only for `loftr` to avoid OOM)

If `query` labels in `metadata.csv` do not overlap with `database` labels
(common in challenge test metadata), the script automatically falls back to
`db_self_eval` mode in `--query-source auto`.

## Matcher options

- `aliked` (default): ALIKED extractor + LightGlue matcher
- `loftr`: LoFTR matcher
- `roma`: RoMA dense matcher (`romatch`, outdoor/indoor pretrained)
- `orb`: OpenCV ORB local matcher (no pretrained weights, useful for offline smoke tests)

RoMA-specific CLI options:
- `--roma-variant {outdoor,indoor}` (default: `outdoor`)
- `--roma-coarse-res` (default: `560`)
- `--roma-upsample-res` (default: `864`)
- `--roma-coarse-res`는 `14`의 배수 권장/요구
- `--roma-cert-threshold` (default: `0.5`)
- `--roma-max-samples` (default: `1200`, visualization sampling limit)
- `--roma-score-mode {count,sum,sum_above}` (default: `sum_above`)

## Real dataset run (AnimalCLEF2025)

```bash
MPLCONFIGDIR=/tmp/mpl \
XDG_CACHE_HOME=/tmp \
/opt/anaconda3/envs/animal_reid/bin/python experiments/local_rerank/run_local_rerank.py \
  --root /path/to/animal-clef-2025 \
  --matcher aliked \
  --device cpu \
  --candidate-size 25 \
  --trials-per-query 1 \
  --max-queries 200
```

To force DB self-evaluation explicitly:

```bash
... --query-source db_self_eval --db-self-eval-per-id 1
```

RoMA example:

```bash
MATCHERS="roma" \
PYTHON_BIN=/opt/anaconda3/envs/animal_reid/bin/python \
experiments/local_rerank/run_local_rerank_gpu.sh /path/to/animal-clef-2025 \
  --max-queries 100 \
  --roma-variant outdoor \
  --roma-score-mode sum_above
```

To allow cross-species candidate sampling (not recommended):

```bash
... --cross-dataset-candidates
```

Optional dataset filter:

```bash
--dataset-filter LynxID2025,SeaTurtleID2022,SalamanderID2025
```

## Offline smoke test

When pretrained weights cannot be downloaded (offline environment), use:

```bash
MPLCONFIGDIR=/tmp/mpl \
XDG_CACHE_HOME=/tmp \
/opt/anaconda3/envs/animal_reid/bin/python experiments/local_rerank/run_local_rerank.py \
  --synthetic-smoke \
  --matcher orb \
  --candidate-size 5 \
  --trials-per-query 2 \
  --max-queries 6
```

This generates synthetic images in:
- `experiments/local_rerank/synthetic_data`

And outputs results to:
- `experiments/local_rerank/results`

## Notes on pretrained weights

`aliked`, `loftr`, and `roma` use pretrained checkpoints loaded by upstream libraries.
If internet is blocked, initialization can fail while downloading checkpoints.

You can either:
- run the offline smoke test with `--matcher orb`, or
- pre-populate `$TORCH_HOME/hub/checkpoints` with required files before execution.
