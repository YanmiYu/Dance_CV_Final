# CV Tool for Dance Choreography Practice

A computer-vision tool for comparing a student's dance imitation against a
reference benchmark clip. It extracts single-person 2D pose per frame with a
pair of pose estimators, embeds the keypoint sequence with a SupCon-trained
graph encoder, aligns benchmark and learner via DTW, and produces
interpretable per-body-part / per-time-window scores plus human-readable
feedback. Results are viewable in a Streamlit dashboard.

## Final integrated pipeline

The pipeline composes five components into one run:

- **HRNet-W32** pose estimator (final keypoint model)
- **SimpleBaseline** pose estimator (secondary keypoint stream)
- **SupCon GNN** pose-encoder for embedding-space similarity (final embedding model)
- **LSTM** temporal error detector with calibrated probabilities (with a
  geometric-threshold fallback when no LSTM checkpoint is available)
- **Fusion / scoring / report** layer that produces JSON, Markdown, an
  overlay video, and per-stream curves
- **Streamlit frontend** that renders any report directory

## Required checkpoints

| Path | Format | Purpose |
| ---- | ------ | ------- |
| `data/processed/train_hrnet_w32/best.pt` | Git LFS | HRNet-W32 weights |
| `data/processed/simple_baseline/best.pt` | Git LFS | SimpleBaseline weights |
| `checkpoints/pose_gnn_encoder_basicdance_allgenre_c01_supcon.pt` | Git (raw) | **Final SupCon GNN encoder (default)** |
| `checkpoints/lstm/best_model.pt` | Git (raw) | Mia LSTM error detector |

The HRNet and SimpleBaseline checkpoints are tracked via Git LFS, so
**run `git lfs pull` after cloning** to download the real binaries.

The legacy GNN checkpoint at `checkpoints/pose_gnn_encoder_oscar.pt` is kept
as a backup. It is no longer the default — pass it explicitly to override.

## Setup

Python 3.10+. On macOS install `ffmpeg` via Homebrew: `brew install ffmpeg`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Pull LFS-tracked HRNet and SimpleBaseline weights:
git lfs pull
```

## Run the final demo

```bash
python run.py \
    --benchmark inputs/benchmark.mp4 \
    --learner   inputs/user.mp4 \
    --out       results/final_demo
```

Outputs in `results/final_demo/`:

- `report.md` — human-readable score + timestamped feedback
- `report.json` — overall score, intervals, fusion params, models used
- `report_curves.png` — similarity / confidence curve with error windows
- `streams.npz` — per-model error / similarity curves on the canonical time axis
- `aligned_side.mp4` — side-by-side overlay video with skeletons

The default config in `configs/integrate/pipeline.yaml` enables HRNet,
SimpleBaseline, and the SupCon GNN. The LSTM head is enabled when
`checkpoints/lstm/best_model.pt` exists; otherwise the runner falls back to
the geometric threshold head. Use `--require-lstm` to fail fast instead of
falling back, or `--no-lstm` to force the fallback.

## View results in Streamlit

```bash
streamlit run src/app/streamlit_app.py
```

The dashboard auto-discovers any directory containing a `report.json`
under `results/` or `data/reports/`. After running the demo above, select
**`results/final_demo`** in the sidebar.

## Override the GNN checkpoint (optional)

To run with the legacy Oscar GNN backup instead of the default SupCon one,
edit `configs/integrate/pipeline.yaml` and set:

```yaml
models:
  gnn:
    checkpoint: checkpoints/pose_gnn_encoder_oscar.pt
```

## Reproducibility — training surfaces

These are not needed to run the demo, but are kept so the final-submission
checkpoints can be reproduced.

### Train HRNet / SimpleBaseline

```bash
# Simple Baseline (Phase 6.1)
python -m src.train.train_pose --train configs/train/train.yaml

# HRNet-W32 (final pose model)
python -m src.train.train_pose --train configs/train/train_hrnet.yaml
```

### Train the SupCon GNN encoder

```bash
python -m src.train.train_pose_gnn_supcon \
    --config configs/train/train_pose_gnn_supcon_basicdance_allgenre_c01.yaml
```

The trainer copies the best checkpoint to
`checkpoints/pose_gnn_encoder_basicdance_allgenre_c01_supcon.pt` automatically.
See `docs/training_supcon_basicdance.md` for details.

### Train the Mia LSTM error detector

```bash
python scripts/build_lstm_dataset.py \
    --kp-dir   data/keypoints \
    --out-train data/lstm/train \
    --out-val   data/lstm/val \
    --out-test  data/lstm/test

python -m src.mia.train \
    --train-dir data/lstm/train \
    --val-dir   data/lstm/val \
    --checkpoint checkpoints/lstm/best_model.pt

python -m src.mia.evaluate \
    --test-dir data/lstm/test \
    --checkpoint checkpoints/lstm/best_model.pt
```

## Repository layout

```
configs/            # YAML configs; every file references docs/project_decisions.md
  data/             # data-pipeline configs
  model/            # model architecture configs
  train/            # training configs (HRNet, SimpleBaseline, SupCon GNN)
  integrate/        # final inference pipeline config
checkpoints/        # final SupCon GNN, LSTM, and legacy Oscar GNN checkpoints
data/               # runtime artifacts, manifests, labels, predictions, reports
docs/               # frozen decisions, training notes
inputs/             # demo benchmark + user clips
scripts/            # orchestration / curation CLIs (incl. AIST prepare pipeline)
src/
  data/             # CSV parsing, manifests, downloading, AIST++ converter
  datasets/         # AIST++ pose dataset, mixed-source sampler, SupCon dataset
  models/           # SimpleBaseline, HRNet-W32, PoseGNNEncoder, heads, decode
  losses/           # SupCon loss
  train/            # training engine, metrics, eval, SupCon GNN trainer
  infer/            # detector crop, video pose inference, temporal smoothing
  compare/          # normalize, embedding features, legacy DTW + render
  pose/             # integrated adapters for HRNet, SimpleBaseline, GNN
  error/            # per-model keypoint and embedding error streams
  fusion/           # final score / interval fusion + markdown report
  mia/              # LSTM temporal error detector + dataset / scoring
  pipeline/         # end-to-end integrated runner (used by run.py)
  app/              # Streamlit demo
  utils/            # io, video, viz, seed, config, checkpoints
tests/              # unit + smoke tests
```

## Generated artifacts (not tracked)

These directories are regenerated locally and intentionally git-ignored:

- `results/` — every pipeline run writes a fresh subdirectory here
- `data/reports/` — older comparison reports from preliminary experiments
- `logs/` — SLURM and training logs
- `data/processed/torch_cache/` — torchvision detector cache (auto-redownloaded)
- `data/processed/train_pose_gnn_supcon_basicdance*/` — SupCon training runs;
  the only thing promoted out of these dirs is the deployed checkpoint at
  `checkpoints/pose_gnn_encoder_basicdance_allgenre_c01_supcon.pt`
- `data/raw_frames/`, `data/raw_videos/`, `data/keypoints2d/`, `data/labels/` —
  source data and labels used by the training pipeline

**Read `docs/project_decisions.md` before touching anything.** All scope
decisions (COCO-17 joints, the HRNet backbone exception, detector-only crop
usage, upper-body weighting) are frozen there.

## Quick smoke commands

```bash
pytest -q
python run.py --help
python -m src.train.train_pose --help
python -m src.infer.run_pose_on_video --help
```

## What this project will NOT do (v1)

- No pretrained pose/keypoint weights, except the documented HRNet ImageNet
  backbone initialization.
- No multi-person / moving-camera pose estimation.
- No transformer as the first model.
- No UI before the CLI pipeline works.
- No single black-box score — every score is diagnosable by body-part / time.
