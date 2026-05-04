# CV Tool for Dance Choreography Practice

A computer-vision tool for comparing a student's dance imitation against a
reference benchmark clip. Pose heads are trained on AIST dance clips in
`data/raw_videos/` supervised by AIST++ 2D keypoints; the HRNet branch may
initialize its backbone from the whitelisted ImageNet checkpoint documented in
`docs/project_decisions.md`. The system extracts single-person 2D pose per
frame, aligns the benchmark and imitation with DTW, and produces interpretable
per-body-part / per-time-window scores plus human-readable feedback.

## Integrated pipeline (this branch)

This `integrate` branch wires four components developed on separate
branches into a single end-to-end pipeline:

- HRNet-W32 pose estimator (`max` branch)
- SimpleBaseline pose estimator (`simple-baseline-lynn` branch)
- GNN pose-encoder for embedding-space similarity (`stevenmerge` branch)
- LSTM temporal error detector + scoring + feedback (`Mia` branch)

The default preprocessing stage uses YOLOv8 person detection for a stable
full-body crop, matching the project pipeline diagram. Set
`preprocessing.detector_backend: torchvision` in
[`configs/integrate/pipeline.yaml`](configs/integrate/pipeline.yaml) to use the
older detector wrapper.

Run on a benchmark / learner pair:

```bash
python run.py \
    --benchmark data/raw_videos/<bench>.mp4 \
    --learner   data/raw_videos/<user>.mp4 \
    --out       results/integrate_run/
```

Outputs in `--out`:

- `report.md`     — human-readable score + timestamped feedback
- `report.json`   — overall score, intervals, fusion params, models used
- `report_curves.png` — similarity / confidence curve with error windows
- `streams.npz`   — per-model error / similarity curves on the canonical time axis

Configure which models/heads run via [`configs/integrate/pipeline.yaml`](configs/integrate/pipeline.yaml).
Default config: HRNet + GNN enabled; SimpleBaseline disabled (no checkpoint
shipped); LSTM disabled (no checkpoint shipped). To enable either, drop the
checkpoint into the path the config points to and flip `enabled: true`.

Required local artifacts:

| Path                                                | Source                       |
| --------------------------------------------------- | ---------------------------- |
| `data/processed/train_hrnet_w32/best.pt`            | trained on `max` branch      |
| `checkpoints/pose_gnn_encoder_oscar.pt`             | committed by `stevenmerge`   |
| `data/processed/simple_baseline/best.pt` (optional) | drop SB ckpt here to enable  |
| `checkpoints/lstm/best_model.pt` (optional)         | drop Mia LSTM ckpt to enable |
| `data/external/pretrained/hrnetv2_w32_imagenet.pth` | `python scripts/download_hrnet_imagenet.py` |



**Read `docs/project_decisions.md` before touching anything.** All scope
decisions (COCO-17 joints, the HRNet backbone exception, detector-only crop
usage, upper-body weighting) are frozen there.

## Repository layout

```
configs/            # YAML configs; every file references docs/project_decisions.md
  data/             # data-pipeline configs
  model/            # model architecture configs
  train/            # training config (single stage, AIST++-only)
data/               # runtime artifacts, manifests, labels, predictions, reports
docs/               # frozen decisions, recording protocol
scripts/            # orchestration / curation CLIs (incl. AIST prepare pipeline)
src/
  data/             # CSV parsing, manifests, downloading, AIST++ converter
  datasets/         # AIST++ pose dataset + mixed-source sampler
  models/           # Simple Baseline, HRNet-W32, heads, losses, decode
  train/            # training engine, metrics, eval
  infer/            # motion crop, video pose inference, temporal smoothing
  compare/          # normalize, features, DTW, score, feedback, report
  pose/             # integrated adapters for HRNet, SimpleBaseline, GNN
  error/            # per-model keypoint and embedding error streams
  fusion/           # final score / interval fusion
  mia/              # namespaced LSTM temporal error detector utilities
  pipeline/         # end-to-end integrated runner
  app/              # Streamlit demo (built last)
  utils/            # io, video, viz, seed, config
tests/              # unit + smoke tests
```

## Setup

Python 3.10+. On macOS install `ffmpeg` via Homebrew: `brew install ffmpeg`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick smoke commands

```bash
pytest -q
python run.py --help
python -m src.train.train_pose --help
python -m src.infer.run_pose_on_video --help
```

## Training data pipeline (AIST++-only)

All supervised pose training runs on frames extracted from the clips in
`data/raw_videos/`, supervised by the matching AIST++ 2D keypoints.
There is no COCO, no CrowdPose, no pseudo-labeling.

1. Drop the AIST++ per-video 2D keypoint files (`.pkl` or `.npy`) under
   `data/labels/aistpp/keypoints2d_raw/<video_stem>.pkl`. Stems must
   match filenames in `data/raw_videos/`.
2. Build frames + train/val JSONL splits:

```bash
python -m scripts.prepare_aist_training_data \
    --raw-videos data/raw_videos \
    --keypoints-dir data/labels/aistpp/keypoints2d_raw \
    --frames-dir data/raw_frames/aistpp \
    --out-dir data/labels/aistpp \
    --frame-stride 8
```

3. Train:

```bash
python -m src.train.train_pose --train configs/train/train.yaml
```

## Milestones (follow in order, never skip)

1.  Repo + configs + manifests work.
2.  Videos in `data/raw_videos/` + AIST++ 2D keypoints on disk.
3.  `scripts/prepare_aist_training_data.py` produces train/val JSONL.
4.  Simple baseline overfits a tiny subset.
5.  Simple baseline trains on the full AIST++-labeled data.
6.  HRNet-style model beats baseline.
7.  Video pose inference on benchmark and imitation clips.
8.  DTW alignment and scoring.
9.  Feedback generation.
10. Streamlit demo end to end.

## What this project will NOT do (v1)

- No pretrained pose/keypoint weights, except the documented HRNet ImageNet
  backbone initialization.
- No multi-person / moving-camera pose estimation.
- No transformer as the first model.
- No UI before the CLI pipeline works.
- No single black-box score — every score is diagnosable by body-part / time.
