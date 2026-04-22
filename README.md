# CV Tool for Dance Choreography Practice

A computer-vision tool for comparing a student's dance imitation against a
reference benchmark clip, trained entirely from scratch (no pretrained
weights). The system extracts a single-person 2D pose per frame, temporally
smooths it, aligns the benchmark and imitation with DTW, and produces
interpretable per-body-part / per-time-window scores plus human-readable
feedback.

**Read `docs/project_decisions.md` before touching anything.** All scope
decisions (e.g. no pretrained models, COCO-17 joints, upper-body weighting)
are frozen there.

## Repository layout

```
configs/            # YAML configs; every file references docs/project_decisions.md
  data/             # data-pipeline configs
  model/            # model architecture configs
  train/            # training-stage configs (A/B/C)
data/               # runtime artifacts, manifests, labels, predictions, reports
docs/               # frozen decisions, recording protocol
scripts/            # orchestration / curation CLIs
src/
  data/             # CSV parsing, manifests, downloading, converters
  datasets/         # pose dataset loaders (COCO / CrowdPose / AIST++ / mixed)
  models/           # Simple Baseline, HRNet-W32, heads, losses, decode
  train/            # training engine, metrics, eval
  infer/            # motion crop, video pose inference, temporal smoothing
  compare/          # normalize, features, DTW, score, feedback, report
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
python -m src.train.train_pose --help
python -m src.infer.run_pose_on_video --help
```

## Milestones (follow in order, never skip)

1.  Repo + configs + manifests work.
2.  Pilot video download + ffprobe integrity.
3.  Unified labeled dataset loader works.
4.  Simple baseline overfits a tiny subset.
5.  Simple baseline trains on full labeled data.
6.  HRNet-style model beats baseline.
7.  Video pose inference on benchmark and imitation clips.
8.  DTW alignment and scoring.
9.  Feedback generation.
10. Streamlit demo end to end.

## What this project will NOT do (v1)

- No pretrained weights anywhere.
- No multi-person / moving-camera pose estimation.
- No transformer as the first model.
- No UI before the CLI pipeline works.
- No single black-box score — every score is diagnosable by body-part / time.
