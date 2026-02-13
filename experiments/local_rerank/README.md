# Local Rerank-Only Experiments

This folder is isolated from the original fusion pipeline.

Goal:
- Build candidate pools of size `K` (default `25`) per query.
- Force-inject at least one ground-truth database sample into each pool.
- Rank candidates using a **local matcher only** (no MegaDescriptor/EVA fusion).
- Measure Top-1 hit rate.

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
- `MATCHER` (default: `aliked`)
- `CANDIDATE_SIZE` (default: `25`)
- `TRIALS_PER_QUERY` (default: `1`)
- `RESULTS_DIR`
- `RUN_PREFIX`

## Matcher options

- `aliked` (default): ALIKED extractor + LightGlue matcher
- `loftr`: LoFTR matcher
- `orb`: OpenCV ORB local matcher (no pretrained weights, useful for offline smoke tests)

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

## Notes on ALIKED / LoFTR weights

`aliked` and `loftr` use pretrained checkpoints loaded by upstream libraries.
If internet is blocked, initialization can fail while downloading checkpoints.

You can either:
- run the offline smoke test with `--matcher orb`, or
- pre-populate `$TORCH_HOME/hub/checkpoints` with required files before execution.
