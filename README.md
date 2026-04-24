# CV Tool for Dance Choreography Practice

A computer-vision tool for comparing a student's dance imitation against a
reference benchmark clip, trained entirely from scratch (no pretrained
weights) on the AIST dance clips in `data/raw_videos/` supervised by
AIST++ 2D keypoints. The system extracts a single-person 2D pose per
frame, temporally smooths it, aligns the benchmark and imitation with
DTW, and produces interpretable per-body-part / per-time-window scores
plus human-readable feedback.

**Read `docs/project_decisions.md` before touching anything.** All scope
decisions (e.g. no pretrained models, COCO-17 joints, upper-body weighting)
are frozen there.

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
  models/           # Simple Baseline, HRNet-W32, PersonDetector, heads, losses, decode
  train/            # training engine (pose + detector), metrics, eval
  infer/            # motion crop, learned person detector, video pose inference, temporal smoothing
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

## Single-person detector (optional, noisy-background robust)

The pose model assumes a tight person bbox. In clean AIST++ studio clips the
MOG2-based `MotionCropper` works, but for real user footage with messy
backgrounds we train a dedicated, **from-scratch** single-person detector
on the *same* AIST++ JSONL (the `bbox_xyxy` field on every row is the only
supervision needed). See `docs/project_decisions.md` sections 1, 2 and 6.

Architecture: CenterNet-style heatmap + size head on our own ResNet-like
backbone (`src/models/person_detector.py`). No pretrained weights.

Training:

```bash
python -m src.train.train_detector --train configs/train/train_detector.yaml
```

The training dataset (`src/datasets/detector_dataset.py`) includes a
**noisy-background synthesizer** that, on each training sample, isolates
the dancer with a fast color-distance silhouette (or optional GrabCut)
and pastes them onto a random background drawn from:

  - a real-world background-image library you provide (recommended),
  - gaussian noise,
  - a random solid color,
  - or, as a fallback, a random other AIST++ frame.

The real-world textures are what makes this work in arbitrary rooms;
without them the detector is mostly limited to AIST-like studios. To set
them up, drop ANY collection of unlabeled images into
`data/raw_backgrounds/` (any subdirectory structure is fine):

```bash
mkdir -p data/raw_backgrounds
# Recommended: ~5k+ varied indoor/outdoor photos, e.g.,
#   - Places365 validation set (~36k images, CC, https://places2.csail.mit.edu/)
#   - SUN397, OpenImages thumbnails, your own room shots, etc.
```

These images are used STRICTLY as augmentation textures -- they carry
no labels and produce no supervisory signal. See
`docs/project_decisions.md` section 6 (revision 2026-04-23). When the
folder is missing or empty, the dataset silently falls back to AIST
frames as backgrounds (less generalization).

Using the trained detector at inference time:

```bash
python -m src.infer.run_pose_on_video \
    --video path/to/clip.mp4 \
    --model-config configs/model/simple_baseline.yaml \
    --ckpt         data/processed/train/best.pt \
    --detector-config configs/model/person_detector.yaml \
    --detector-ckpt   data/processed/detector/best.pt \
    --out-dir data/predictions/clip
```

When `--detector-ckpt` is omitted, the pipeline falls back to the legacy
motion-based cropper. When provided, the detector predicts a bbox and the
motion cropper is only used as a last-resort fallback if the detector's
peak confidence is below threshold.

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

- No pretrained weights anywhere.
- No multi-person / moving-camera pose estimation.
- No transformer as the first model.
- No UI before the CLI pipeline works.
- No single black-box score — every score is diagnosable by body-part / time.
