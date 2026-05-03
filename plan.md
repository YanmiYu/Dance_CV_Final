# Dance Choreography Practice Tool — 3-Week Project Plan

## Project Overview

Given a 10–15 second benchmark dance video and a learner's video of the same phrase, the system:
1. Extracts body keypoints from both videos (MediaPipe)
2. Normalizes and aligns the two motion sequences (DTW)
3. **Runs a trained LSTM Temporal Error Detector** to output, per aligned frame and per body part, the probability that the learner is "off"
4. Outputs a timestamped report telling the learner exactly **when** and **which body parts** deviate from the benchmark

No data collection is required. Training and testing use open-source dance datasets (AIST++, Human3.6M).

---

## Team Roles

| Member | Primary Focus |
|--------|--------------|
| Member 1 | Pose estimation & keypoint extraction |
| Member 2 | Motion alignment, feature engineering & model training |
| Member 3 | Dataset preparation, label generation & model evaluation |
| Member 4 | System integration, visualization & Streamlit demo |

---

## Open-Source Data Used

| Dataset | Purpose |
|---------|---------|
| **AIST++** | Primary training/validation/test data — 1,408 paired dance sequences across 30 subjects and 10 genres; same choreography performed by different dancers gives natural benchmark/learner pairs |
| **Human3.6M** | Additional motion sequences for stress-testing the pipeline |
| **COCO Keypoints** | Validate that the pose estimator produces correct 17-joint output on still images |
| **YouTube MV / tutorial clips** | Real benchmark videos used in the final demo (BLACKPINK, etc.) |

---

## Model: What We Train and Why

### Problem Framing

The rule-based pipeline flags a body part as "off" whenever its per-frame error
exceeds a fixed threshold (0.25) for at least 0.5 s.  This is fragile:

- A single threshold cannot adapt to dance style, speed, or body-part dynamics
- It fires on every transient fluctuation (e.g. phrase_05 scored 0.0 with 36 spurious intervals)
- It has no sense of *temporal pattern* — it treats each frame independently

We replace it with a **learned LSTM Temporal Error Detector**.

### Approach: LSTM Temporal Error Detector

```
Input at each aligned frame t:
  diff_features(t)  =  [bench_kp_norm(t) − user_kp_norm(t)]  per body part
                     =  6 parts × 4 features = 24-dim vector
                        features per part: mean_x_err, mean_y_err,
                                           max_joint_err, mean_joint_err

Model (src/model.py — TemporalErrorDetector):
  LSTM(input=24, hidden=64, layers=2, unidirectional, dropout=0.3)
    → takes sequence of 24-dim difference vectors  shape (T', 24)
    → outputs per-frame hidden states              shape (T', 64)
  Linear(64 → 6)  →  logits for 6 body parts
  sigmoid(logits) →  P(off) ∈ [0, 1] per frame per body part

Interval detection:
  Replace:  error > 0.25  for ≥ 0.5 s
  With:     P(off) > 0.5  for ≥ 0.5 s
```

### Why LSTM?
- Captures **temporal patterns**: sustained drift looks different from a single-frame spike — the LSTM learns the difference
- Suppresses noise: a 2-frame blip will not produce high probability across a 0.5 s window
- Binary output: simpler and more interpretable than 3-class — the model answers "is this frame off?" directly
- Lightweight: ~65 K parameters; trains in minutes on CPU, runs in real time

### Label Generation (Weak Supervision)

AIST++ provides the same choreography performed by multiple subjects. We:
1. Pick one subject as the "benchmark" and another as the "learner" for each choreography
2. Compute normalized DTW-aligned difference vectors
3. Label a frame as **off (1)** if mean joint error ≥ 0.35 torso lengths, else **correct (0)**
4. Train with `BCEWithLogitsLoss`, upweighting the "off" class (pos_weight=3) for class imbalance

### Train / Validation / Test Split

| Split | Source | Size | Purpose |
|-------|--------|------|---------|
| Train | AIST++ — 20 choreographies, subjects 1–20 | ~70% of pairs | Learn LSTM weights |
| Validation | AIST++ — same choreographies, subjects 21–25 | ~15% | Tune lr, dropout, threshold |
| Test | AIST++ — 5 held-out choreographies (never seen during training) | ~15% | Final F1 and precision/recall |
| Demo | YouTube MV + AIST++ clips | 3–5 clips | Qualitative demo in Streamlit |

---

## Task Checklist

### Infrastructure & Code
- [x] Repo structure set up; `requirements.txt` complete with all dependencies
- [x] `src/video_io.py` — `load_video`, `standardize`, `export_side_by_side`
- [x] `src/pose_extraction.py` — `PoseEstimator` (MediaPipe), Savitzky-Golay smoothing, confidence interpolation
- [x] `src/normalization.py` — center on hip midpoint, scale by torso length
- [x] `src/alignment.py` — DTW alignment via `fastdtw`, `warping_path_to_timestamps`
- [x] `src/dataset.py` — `build_diff_features` (24-dim), `build_labels` (binary), `DanceDeviationDataset`, `collate_fn`
- [x] `src/model.py` — LSTM `TemporalErrorDetector` (input 24 → hidden 64 → output 6 probabilities)
- [x] `src/train.py` — training loop, BCEWithLogitsLoss (pos_weight=3), Adam optimizer, CSV logging, best-checkpoint saving
- [x] `src/test.py` — evaluation loop, per-part binary precision / recall / F1 / AUC, threshold baseline
- [x] `src/scoring.py` — `compute_joint_errors`, `per_part_error_over_time`, `find_off_moments`, `overall_score`
- [x] `src/visualization.py` — skeleton overlay, side-by-side comparison video, Plotly error timeline & bar chart
- [x] `src/feedback.py` — plain-English interval descriptions, Markdown report formatter
- [x] `demo/app.py` — Streamlit UI: upload → full pipeline → score + video + chart + feedback
- [x] `main.py` — CLI for `extract`, `extract_all`, `train`, `test`, `analyze`, `batch`
- [x] `slurm_run.sh` — SLURM job script for Oscar (all tasks)
- [x] `scripts/download_data.sh` — downloads 5 AIST++ benchmark/learner video pairs
- [x] `scripts/build_dataset.py` — builds train/val/test `.npz` files from AIST++ keypoints

### Data & Extraction
- [x] Download 5 AIST++ video pairs into `data/phrase_01/` – `data/phrase_05/`
- [x] Extract keypoints for all 5 pairs (`python main.py batch`); `.npy` files saved
- [x] Full inference pipeline verified end-to-end: normalization → DTW → scoring → JSON report + comparison video
- [x] Download 120 AIST++ videos via `scripts/filtered_gBR_sBM_c01.csv` to Oscar (`data/videos/`)
- [x] Extract keypoints for all 120 videos on Oscar → `data/keypoints/`
- [x] Run `scripts/build_dataset.py` → labeled `.npz` pairs in `data/train/`, `data/val/`, `data/test/`

### Model Training & Testing
- [x] Launch training on Oscar: `sbatch slurm_run.sh train` (job 2212860)
- [x] 30 epochs, train loss `1.030 → 0.684`, val loss `1.236 → 0.926`, val F1 `0.462 → 0.662`
- [x] Best checkpoint saved: `checkpoints/best_model.pt` (225 KB, epoch 30, val F1 = 0.662)
- [x] Run `python main.py test` on held-out test split → `results/test_metrics.json`
- [x] **Test results** (on ch10 — choreographies never seen during training):

| Body Part  | Precision | Recall | F1    | AUC   |
|------------|-----------|--------|-------|-------|
| RIGHT_ARM  | 0.978     | 1.000  | 0.989 | 0.827 |
| LEFT_LEG   | 0.943     | 1.000  | 0.971 | 0.873 |
| RIGHT_LEG  | 0.969     | 1.000  | 0.984 | 0.844 |
| LEFT_ARM   | 0.939     | 1.000  | 0.968 | 0.881 |
| HEAD       | 0.879     | 1.000  | 0.935 | 0.728 |
| TORSO      | 0.807     | 0.993  | 0.890 | 0.795 |
| **Overall**| **0.919** | **0.999** | **0.958** | — |

  TORSO and HEAD are the hardest parts (lowest AUC). Recall ≈ 1.0 everywhere — the model catches all "off" frames with no missed detections; some false positives remain (lower precision on TORSO/HEAD).

### Evaluation & Analysis
- [x] Per-body-part error analysis complete — TORSO and HEAD hardest (AUC 0.73–0.79); arm/leg parts easiest (AUC 0.83–0.88)
- [x] Run full pipeline locally with trained model: `python main.py batch --checkpoint checkpoints/best_model.pt`
- [x] Compare interval outputs vs. old fixed-threshold results — see table below
- [ ] Run full pipeline on 3–5 demo clips outside AIST++ (e.g. YouTube); write qualitative assessment for each

#### Model vs. Fixed-Threshold — Interval Comparison on phrase_01–05

Overall scores are identical (score is geometric, model only affects interval detection).
The model reduces total intervals from **74 → 37** (50% fewer) by merging fragmented
short detections into coherent longer windows and suppressing transient noise.

| Phrase | Score | Threshold intervals | Model intervals | Key difference |
|--------|-------|---------------------|-----------------|----------------|
| phrase_01 | 86 | 4 (HEAD×3, LEFT_ARM×1) | 2 (LEFT_ARM, RIGHT_ARM at start) | Model drops 3 short HEAD blips (< 0.5 s each); flags the arm deviation at clip start more precisely |
| phrase_02 | 84 | 12 (arms throughout) | 9 (arms throughout) | Model merges several consecutive short arm intervals into longer coherent windows |
| phrase_03 | 89 | 4 (HEAD+arms+leg at end) | 1 (LEFT_ARM at start) | Model suppresses end-of-clip artifacts entirely; catches only the real deviation |
| phrase_04 | 80 | 18 (arms, legs, head) | 7 (arms dominant) | Model collapses fragmented arm alerts (10.4–11.7 s split into 3 pieces) into one clean 7.9–11.9 s window |
| phrase_05 | 0 | **36** (all parts, every segment) | **18** (same parts, longer windows) | Biggest improvement: threshold fired on every transient fluctuation; model merges into sustained deviations only |

**phrase_05 detail** — the model correctly identifies large, sustained deviations (e.g. RIGHT_ARM 2.0–11.9 s, HEAD 5.1–11.9 s) whereas the threshold produced 36 noisy short intervals across the same time span. Both score 0/100 because the overall geometric error is extreme — this learner (d05 music mBR5) is genuinely very far from the benchmark throughout the clip.

<!-- ### Report & Presentation
- [ ] Write model training + results sections in `docs/final_report.md`
- [ ] Write ablation study section with clean F1 tables and confusion matrices
- [ ] Write qualitative evaluation section
- [ ] End-to-end test on a new demo pair not seen during any development
- [ ] Refine Streamlit UI: loading spinner, `MM:SS.f` timestamps, clear section headers
- [ ] Record a 2–3 minute screen-capture demo video
- [ ] Prepare slides: motivation → pipeline → training → results → ablation → demo → limitations → future work
- [ ] Rehearse live demo; confirm it runs without errors on the demo machine -->

---

## Deliverables Summary

| Deliverable | Owner | Target |
|------------|-------|--------|
| Pose extraction module | M1 | End of Week 1 |
| `dataset.py` + labeled `.npz` training data | M2 + M3 | End of Week 1 |
| Train/val/test split index (`splits.json`) | M3 | End of Week 1 |
| Trained LSTM checkpoint | M2 | End of Week 2 |
| Test-set evaluation metrics (precision / recall / F1 / AUC per part) | M3 | End of Week 2 |
| End-to-end Streamlit demo (with trained model) | M4 | End of Week 2 |
| Qualitative demo evaluation | M1 | End of Week 3 |
| Final report | All | End of Week 3 |
| Demo video + slides | M4 | End of Week 3 |

---

## Technical Architecture

### Full Pipeline (Inference)

```
benchmark.mp4 ──┐
                ├──► [1. Video Ingestion] ──► [2. Pose Extraction] ──► [3. Normalization]
learner.mp4   ──┘                                                              │
                                                                               ▼
                                                                    [4. DTW Alignment]
                                                                               │
                                                                               ▼
                                                              [5. Difference Feature Builder]
                                                                    (T', 24) per frame
                                                                               │
                                                                               ▼
                                                          ┌────────────────────────────────┐
                                                          │  6. LSTM Temporal Error Detector│  ← trained model
                                                          │  input:  (T', 24)               │
                                                          │  output: (T', 6) P(off) per part│
                                                          └───────────────┬─────────────────┘
                                                                          │
                                                                          ▼
                                                              [7. Interval Extraction]
                                                         P(off) > 0.5 for ≥ 0.5 s → Interval list
                                                                    ↙            ↘
                                                     [8a. Comparison video]  [8b. Timeline chart]
                                                                    ↘            ↙
                                                              [9. Streamlit UI]
                                                      "At 3.2s–5.8s your right arm is off"
```

### Training Pipeline

```
AIST++ keypoints (.npy)
        │
        ▼
[normalize + DTW-align pairs]
        │
        ▼
[build_diff_features()]  →  (T', 24) per pair
        │
        ▼
[build_labels()]         →  (T', 6) binary labels   ← weak supervision
                              1 = off  (mean_err ≥ 0.35 torso lengths)
                              0 = correct
        │
        ▼
[save .npz per pair]     →  data/train/ data/val/ data/test/
        │
        ▼
[PyTorch DataLoader]
        │
        ▼
[LSTM TemporalErrorDetector]
   BCEWithLogitsLoss (pos_weight=3 for class imbalance)
   Adam optimizer, lr=1e-3
   30 epochs, best checkpoint by val F1
        │
        ▼
[checkpoints/best_model.pt]
```

### Layer-by-Layer Detail

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — VIDEO INGESTION          src/video_io.py  (M4)                   │
│  load_video(path) → frames + fps                                             │
│  standardize(frames, fps=15, res=(720,1280))                                 │
│  export_side_by_side(frames_A, frames_B, out_path)                           │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │ BGR frames
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  LAYER 2 — POSE EXTRACTION          src/pose_extraction.py  (M1)            │
│  PoseEstimator(backend="mediapipe")                                          │
│    .extract_sequence(frames) → (T, 17, 3)  [x_px, y_px, confidence]         │
│  smooth()  interpolate_missing()  save_keypoints()  load_keypoints()         │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │ (T, 17, 3)
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  LAYER 3 — NORMALIZATION            src/normalization.py  (M2)              │
│  normalize(seq): center on hip midpoint, scale by torso length               │
│  → (T, 17, 3) in torso-length units, body-size independent                  │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │ normalized (T, 17, 3)
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  LAYER 4 — DTW ALIGNMENT            src/alignment.py  (M2)                  │
│  dtw_align(bench_norm, user_norm)                                            │
│    → bench_aligned, user_aligned  shape (T', 17, 3)                          │
│    → warping_path  List[(i_bench, i_user)]                                   │
│  Library: fastdtw — O(N) time/space                                          │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │ aligned sequences (T', 17, 3)
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  LAYER 5 — FEATURE ENGINEERING      src/dataset.py  (M2)                    │
│  build_diff_features(bench_aligned, user_aligned)                            │
│    For each of 6 body parts:                                                 │
│      [mean_x_err, mean_y_err, max_joint_err, mean_joint_err]                 │
│    → (T', 24)  — one 24-dim vector per aligned frame                         │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │ (T', 24) diff feature sequence
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  LAYER 6 — LSTM TEMPORAL ERROR DETECTOR   src/model.py                      │
│                                                                              │
│  class TemporalErrorDetector(nn.Module):                                     │
│    self.lstm = nn.LSTM(input_size=24, hidden_size=64,                        │
│                        num_layers=2, batch_first=True, dropout=0.3)          │
│    self.head = nn.Linear(64, 6)                                              │
│                                                                              │
│    forward(x: Tensor (B, T', 24))                                            │
│      h, _ = self.lstm(x)         → (B, T', 64)                              │
│      logits = self.head(h)        → (B, T', 6)                              │
│                                                                              │
│    predict_proba(x)                                                          │
│      return sigmoid(logits)       → (B, T', 6)  P(off) ∈ [0, 1]            │
│                                                                              │
│  Output: per frame per body part — probability the frame is "off"            │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │ (T', 6) P(off) per frame per part
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  LAYER 7 — INTERVAL EXTRACTION      src/scoring.py                           │
│  find_off_moments(part_signals, threshold=0.5, fps, min_duration_s=0.5)      │
│    → List[Interval(start_s, end_s, part, mean_error)]                        │
│    contiguous runs where P(off) > 0.5 per body part → one Interval each     │
│    minimum run length = 0.5 s to suppress transient spikes                   │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │ List[Interval]
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  LAYER 8 — VISUALIZATION & FEEDBACK                                          │
│  src/visualization.py  (M4)                                                  │
│    overlay_skeleton(), render_comparison_video(), plot_error_timeline()       │
│  src/feedback.py  (M4)                                                       │
│    generate_feedback(intervals) → plain-English timestamped sentences         │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  ▼
                           demo/app.py  (M4) — Streamlit UI
```

---

### Data Structures

```python
# Keypoint sequence — shape (T, 17, 3), axis-2 = [x_norm, y_norm, confidence]
KeypointSequence = np.ndarray

# COCO joint indices
#  0 nose | 1 L_eye | 2 R_eye | 3 L_ear | 4 R_ear
#  5 L_shldr | 6 R_shldr | 7 L_elbw | 8 R_elbw | 9 L_wrst | 10 R_wrst
# 11 L_hip | 12 R_hip | 13 L_knee | 14 R_knee | 15 L_ankl | 16 R_ankl

BODY_PARTS: dict[str, list[int]] = {
    "LEFT_ARM":   [5, 7, 9],
    "RIGHT_ARM":  [6, 8, 10],
    "LEFT_LEG":   [11, 13, 15],
    "RIGHT_LEG":  [12, 14, 16],
    "TORSO":      [5, 6, 11, 12],
    "HEAD":       [0],
}

# Diff feature vector per frame: shape (T', 24)
# 6 parts × 4 features: [mean_x_err, mean_y_err, max_joint_err, mean_joint_err]
DiffFeatures = np.ndarray

# Label tensor per frame: shape (T', 6)  — binary float32: 0.0=correct, 1.0=off
Labels = np.ndarray

@dataclass
class Interval:
    start_s:    float   # start time (seconds)
    end_s:      float   # end time (seconds)
    part:       str     # e.g. "RIGHT_ARM"
    severity:   int     # 1=moderate, 2=off
```

---

### Module Dependency Graph

```
video_io.py
    ↓
pose_extraction.py
    ↓
normalization.py
    ↓
alignment.py
    ↓
dataset.py          ←── also used by train.py and test.py
    ↓
model.py            ←── trained by train.py; loaded by scoring.py at inference
    ↓
scoring.py          (labels_to_intervals)
    ↓
visualization.py ──┐
feedback.py      ──┤
                   ↓
              demo/app.py  (orchestrates all)
```

---

### File & Folder Layout

```
Final_CV/
│
├── data/
│   ├── phrase_01/
│   │   ├── benchmark.mp4
│   │   ├── learner.mp4
│   │   └── keypoints/
│   │       ├── benchmark_kp.npy    # (T, 17, 3)
│   │       └── learner_kp.npy
│   ├── phrase_02/  ...
│   ├── phrase_03/  ...
│   ├── train/                      # .npz files: features + labels per pair
│   ├── val/
│   ├── test/
│   └── splits.json                 # choreography IDs → train/val/test
│
├── src/
│   ├── video_io.py            # M4 — frame loading, standardization, export
│   ├── pose_extraction.py     # M1 — PoseEstimator, smoothing, interpolation
│   ├── normalization.py       # M2 — centering & scaling
│   ├── alignment.py           # M2 — DTW alignment
│   ├── dataset.py             # M2 — DiffFeature builder + PyTorch Dataset
│   ├── model.py               # M2 — DeviationClassifier (BiGRU)
│   ├── train.py               # M2 — training loop, logging, checkpointing
│   ├── test.py                # M3 — evaluation loop, metrics, confusion matrix
│   ├── scoring.py             # M2 — labels_to_intervals, overall_score
│   ├── visualization.py       # M4 — skeleton overlay, video renderer, charts
│   └── feedback.py            # M4 — interval → plain-English sentences
│
├── checkpoints/
│   └── best_model.pt          # saved after training
│
├── results/
│   ├── training_log.csv       # epoch, train_loss, val_loss, val_f1
│   ├── test_metrics.json      # final test-set F1, accuracy, confusion matrix
│   └── phrase_XX/             # per-phrase analysis outputs (JSON + videos)
│
├── scripts/
│   └── build_dataset.py       # M3 — generates train/val/test .npz from AIST++
│
├── demo/
│   └── app.py                 # M4 — Streamlit UI
│
├── notebooks/
│   ├── 01_pose_exploration.ipynb       # M1
│   ├── 02_alignment_experiments.ipynb  # M2
│   ├── 03_model_training_curves.ipynb  # M2 — plot loss/F1 curves
│   └── 04_evaluation_analysis.ipynb    # M3 — confusion matrices, ablation table
│
├── docs/
│   └── final_report.md
│
├── main.py                    # CLI: extract / extract_all / train / test / analyze / batch
├── slurm_run.sh               # SLURM job script for Oscar
├── requirements.txt
└── README.md
```

---

### Key Algorithmic Choices

#### Pose Estimator
- **MediaPipe Pose** (default): CPU-friendly, no extra install
- **MMPose** (optional, GPU): higher accuracy; swap in via `backend="mmpose"`

#### Normalization
1. **Center** — hip midpoint → (0, 0) each frame
2. **Scale** — divide by torso length; removes height and camera-distance effects

#### DTW Alignment
- Cost per frame pair = mean Euclidean distance across all 17 joints
- Library: `fastdtw` — O(N) time/space; fast enough for 15 fps × 15 s clips

#### Difference Features (model input)
- Per body part and per aligned frame: `[mean_x_err, mean_y_err, max_joint_err, mean_joint_err]`
- 6 parts × 4 = **24-dimensional input vector per frame**
- Normalized so values are in torso-length units (scale-invariant)

#### BiGRU Classifier
| Hyperparameter | Value |
|----------------|-------|
| Input size | 24 |
| Hidden size | 64 |
| Layers | 2 |
| Bidirectional | Yes (effective hidden = 128) |
| Dropout | 0.3 |
| Output | 6 parts × 3 classes |
| Loss | Weighted CrossEntropy ("off" class weight × 3) |
| Optimizer | Adam, lr=1e-3, weight_decay=1e-4 |
| Epochs | 30 (early stop by val F1) |
| Batch size | 16 sequences (padded to same length) |

#### Ablation Study Design
| Condition | Change | Expected effect |
|-----------|--------|----------------|
| **Full model** (baseline) | BiGRU + DTW + norm | — |
| No temporal context | MLP per frame | Lower F1 on sustained deviations |
| Unidirectional GRU | Remove backward pass | Slightly lower F1 |
| No DTW | Naive frame-by-frame | Higher false positives from timing mismatch |
| No normalization | Skip normalize() | Higher false positives from scale differences |
| Fixed threshold | No model | Fragile, not adaptive to dance style |

#### Color Coding (visualization)
| Label | Class | Color |
|-------|-------|-------|
| Good | 0 | Green `#2ecc71` |
| Moderate | 1 | Yellow `#f39c12` |
| Off | 2 | Red `#e74c3c` |

---

## Risk Register & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| AIST++ download is slow / large | Medium | Use pre-extracted 2D keypoints (much smaller than raw video) |
| Geometric-threshold labels are too noisy for training | Medium | Increase "off" class weight in loss; inspect label quality in notebook |
| BiGRU overfits on training choreographies | Medium | Dropout 0.3 + weight decay; hold out 5 choreographies for test |
| Pose estimator fails on fast / sideways motion | Medium | Confidence filtering + interpolation; fall back to MMPose on Oscar |
| DTW too slow at 30 fps | Low | Downsample to 15 fps before alignment |
| Training time exceeds SLURM wall time | Low | 30 epochs on CPU ≈ 20 min; 1-hour wall time is sufficient |

---

## MVP Scope (must-have)

- Pose extraction from any two uploaded 10–15 second videos
- Normalization + DTW temporal alignment
- Trained BiGRU classifier predicting good / moderate / off per frame per body part
- `labels_to_intervals()` → timestamped "off" interval list
- Plain-English feedback: "At X s – Y s your RIGHT ARM is off"
- Side-by-side skeleton overlay video
- Error timeline chart in Streamlit

## Stretch Goals (nice-to-have)

- Live webcam capture instead of video upload
- Confidence scores on each predicted interval
- Exportable PDF summary report
- Multi-angle support

---

## Code Review Log — 2026-04-19

Full pass over every script to remove dead code, fix bugs, and align paths across
documentation, SLURM, and the filesystem. Changes are grouped by file.

### `helper/` → `scripts/` (directory rename)
The helper scripts were living in `helper/` while all documentation, SLURM, and
`build_dataset.py`'s own docstring referenced `scripts/`. Renamed the directory
so every path is consistent.

### `src/alignment.py`
- Removed `_frame_distance()` — dead code; `fastdtw` is called with
  `scipy.spatial.distance.euclidean` directly and `_frame_distance` was never
  called anywhere in the project.

### `src/scoring.py`
- Removed `AnalysisResult` dataclass — never instantiated anywhere. Downstream
  callers (`main.py`, `demo/app.py`) work directly with the individual return
  values (`score`, `intervals`, `part_errors`).
- Removed the now-unused `field` import from `dataclasses`.

### `src/test.py`
- **Bug fix — baseline accuracy crash**: `float(np.array(preds) == np.array(targets))`
  raises `TypeError` because `float()` cannot convert a boolean array. Fixed to
  `(np.array(...) == np.array(...)).mean()`.
- **Bug fix — f-string format crash**: `f"(val_f1={ckpt_meta.get('val_f1', '?'):.4f})"`
  raises `ValueError` when the key is absent and the default `'?'` (a string) is
  formatted with `:.4f`. Fixed by extracting the value first and branching on its
  type before formatting.

### `src/visualization.py`
- `render_comparison_video` — added optional `timestamps: np.ndarray | None`
  parameter. When provided (from `warping_path_to_timestamps`), the red "off"
  border and the per-frame timestamp label use DTW-accurate real time instead of
  the naive `t / fps` approximation, which was misaligned after DTW warping.
- `plot_error_timeline` — added optional `timestamps` parameter so the Plotly
  x-axis can use the same DTW-accurate time axis that `find_off_moments` uses for
  interval boundaries. Falls back to `np.arange(T) / fps` when not supplied.

### `main.py`
- Removed unused import `format_report` (from `feedback`) and `plot_error_timeline`
  (from `visualization`). Both were imported at the top level but never called.
- Removed the redundant `if args.checkpoint ... else` block in `task_analyze` that
  printed different labels but ran identical `find_off_moments` calls in both
  branches. Replaced with a single unconditional call.
- `render_comparison_video` call now passes `timestamps=timestamps` so the video
  border aligns correctly with the flagged intervals.

### `demo/app.py`
- Removed unused import `export_side_by_side` (from `video_io`).
- Removed unused import `AnalysisResult` (from `scoring`; the dataclass was also
  deleted from `scoring.py`).

### `scripts/build_dataset.py`
- Removed the `fps` parameter from `process_pair()` — it was declared and accepted
  but never used inside the function body. The `--fps` CLI argument is retained in
  `main()` for documentation purposes but is no longer forwarded.
- Added a `None`-guard in `load_aist_keypoints()`: when the dict contains none of
  the recognized keys (`keypoints2d`, `kps2d`, `joints2d`), the function now raises
  a descriptive `ValueError` listing the actual keys instead of silently passing
  `None` to `np.array()` and producing a confusing downstream error.
