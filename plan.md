# Dance Choreography Practice Tool — 3-Week Project Plan

## Project Overview

Given a 10–15 second benchmark dance video and a learner's video of the same phrase, the system extracts body keypoints from both, aligns the two sequences in time, and outputs a timestamped report telling the learner exactly **when** and **which body parts** deviate from the benchmark.

No data collection is required. All testing and validation uses open-source pose/dance datasets.

---

## Team Roles

| Member | Primary Focus |
|--------|--------------|
| Member 1 | Pose estimation & keypoint extraction |
| Member 2 | Motion alignment & deviation scoring |
| Member 3 | Evaluation, open-source data integration & ablation |
| Member 4 | System integration, visualization & Streamlit demo |

---

## Open-Source Data Used

| Dataset | Purpose |
|---------|---------|
| **COCO Keypoints** | Validate that the pose estimator produces correct 17-joint output |
| **AIST++ / AIOZ-GDance** | Source of short paired or solo dance sequences for end-to-end testing |
| **Human3.6M** | Additional motion sequences to stress-test the alignment and scoring |
| **Publicly available dance tutorial / MV clips** (YouTube) | Real benchmark videos (e.g., BLACKPINK choreography) used as inputs in the demo |

For the demo and development, we trim clips to 10–15 seconds and use a single frontal camera angle.

---

## Week 1 — Foundation: Pose Extraction Pipeline

**Goal:** Given any 10–15 second video, reliably extract a smooth keypoint sequence and save it to disk.

### Member 1 — Pose Estimation
- [ ] Evaluate MediaPipe Pose vs. MMPose on 2–3 sample dance clips; choose one based on accuracy and ease of setup (MediaPipe is the default — CPU-friendly, no extra install)
- [ ] Implement `extract_keypoints(video_path) → np.ndarray` returning shape `(T, 17, 3)` — `[x, y, confidence]` per joint per frame
- [ ] Add post-processing: Savitzky-Golay temporal smoothing to reduce jitter; confidence-based linear interpolation for occluded frames
- [ ] Save/load keypoints to `.npy`; write a quick visualization script that draws the skeleton on each frame to verify output visually

### Member 2 — Normalization
- [ ] Implement `normalize(seq)`: center all joints on hip midpoint, scale by torso length (neck-to-hip distance) so body size and camera distance no longer affect comparisons
- [ ] Verify that two clips of the same dancer at different distances produce near-identical normalized sequences
- [ ] Prototype DTW on a pair of short toy sequences; confirm it handles a learner who performs the move correctly but slightly slower

### Member 3 — Data Preparation & Validation
- [ ] Download and preprocess clips from AIST++ or AIOZ-GDance; trim to 10–15 second phrases, convert to 30 fps MP4
- [ ] Validate that Member 1's extractor produces sensible keypoints on these clips (visual spot-check, confidence histogram)
- [ ] Prepare at least 3 test pairs: `(benchmark_clip, learner_clip)` where the "learner" is a different clip of the same routine or a clip with an intentionally different pose — used to verify the scoring logic later

### Member 4 — Project Scaffold
- [ ] Set up the repo structure (see File Layout below) and `requirements.txt`
- [ ] Write `README.md` with setup and run instructions
- [ ] Build `video_io.py`: `load_video`, `standardize_fps`, `export_side_by_side`
- [ ] Sketch the Streamlit UI wireframe: two upload boxes, a "Run Analysis" button, a results panel

### End-of-Week-1 Milestone
- `extract_keypoints(video_path)` works correctly on any frontal dance clip
- 3 test pairs ready in `data/` with keypoints extracted
- All members can run the pipeline on their machine

---

## Week 2 — Core System: Alignment, Scoring & Timestamped Feedback

**Goal:** Given two extracted keypoint sequences, produce a timestamped list of "off" moments per body part.

### Member 1 — Robustness & Integration
- [ ] Harden the extractor: handle missing detections (no person in frame), very fast motion, and partial occlusion
- [ ] Expose a clean `PoseEstimator` class that wraps the chosen backend so swapping MediaPipe ↔ MMPose requires changing one line
- [ ] Profile runtime; downsample to 15 fps before DTW if needed for speed

### Member 2 — Alignment & Deviation Scoring
- [ ] Implement `dtw_align(bench_seq, user_seq)` using `fastdtw`; output the warping path and the two aligned sequences of equal length `T'`
- [ ] Implement `compute_joint_errors(aligned_bench, aligned_user) → dict[joint → np.ndarray (T',)]` — per-joint Euclidean error at each aligned frame
- [ ] Group joints into body parts and compute `per_part_error_over_time(joint_errors) → dict[part → np.ndarray (T',)]`
- [ ] Implement `find_off_moments(part_errors, threshold, fps) → List[Interval]` — scans each body part's error curve and returns time intervals (start_s, end_s) where error exceeds the threshold
- [ ] Compute an overall similarity score (0–100) as a summary statistic

### Member 3 — Scoring Validation & Ablation
- [ ] Run the end-to-end pipeline on the 3 test pairs; manually inspect whether the flagged timestamps correspond to visually obvious differences
- [ ] Run a simple ablation: compare results with and without normalization; compare DTW vs. naive frame-by-frame matching — document which produces more sensible intervals
- [ ] Adjust the deviation threshold empirically so that minor style variation is not flagged but clear positional errors are

### Member 4 — Visualization
- [ ] Implement `overlay_skeleton(frame, keypoints, part_errors)` — draws skeleton on a frame; joints colored green / yellow / red based on current error level
- [ ] Implement `render_comparison_video(bench_frames, user_frames, bench_kp, user_kp, intervals, out_path)` — side-by-side MP4 with colored skeletons and a timestamp ticker; "off" intervals highlighted with a red border
- [ ] Implement `plot_error_timeline(part_errors, intervals, fps)` — Plotly chart: x = time in seconds, y = error per body part, with shaded "off" regions
- [ ] Wire all components into `demo/app.py` so the full pipeline runs from the UI

### End-of-Week-2 Milestone
- Input two video files → output a list of `(time_start, time_end, body_part, severity)` intervals
- Comparison video and timeline chart render correctly
- Streamlit app runs end-to-end on at least one test pair

---

## Week 3 — Evaluation, Polish & Demo

**Goal:** Validate the system's outputs, fix failure cases, and prepare the final demo and report.

### Member 1 — Pipeline Finalization
- [ ] Final robustness pass: test on all 3+ test pairs; fix any systematic failures
- [ ] (Stretch) Add live webcam capture mode: user records through the app instead of uploading a file
- [ ] Write docstrings for all public functions in `pose_extraction.py` and `video_io.py`

### Member 2 — Scoring Finalization
- [ ] Tune threshold and body-part weights based on Member 3's validation findings
- [ ] Compute and report quantitative metrics across the test pairs:
  - Mean number of "off" intervals per clip
  - Distribution of interval durations
  - Correlation between overall score and visible deviation (sanity check)
- [ ] Document the final scoring formula and threshold choices in `docs/final_report.md`

### Member 3 — Evaluation Report
- [ ] For each test pair, write a short written assessment: does the system's flagged output agree with what a human viewer would notice?
- [ ] Compile an ablation table: normalization on/off × DTW on/off → how do flagged intervals change?
- [ ] Write the evaluation section of the final report

### Member 4 — Demo Polish & Presentation
- [ ] Refine Streamlit UI: loading spinner, clear section headers, timestamps formatted as `MM:SS.f`
- [ ] Add a **Feedback Summary** panel listing flagged intervals in plain English:
  > "At 3.2 s – 5.8 s your **right arm** is significantly off. At 9.0 s – 11.4 s your **left leg** deviates from the benchmark."
- [ ] Record a 2–3 minute screen-capture demo video
- [ ] Prepare presentation slides: motivation → pipeline → demo → results → limitations → future work

### All Members — Integration & Documentation
- [ ] End-to-end test on a new pair not seen during development
- [ ] Write assigned sections of the final report; proofread and merge
- [ ] Rehearse the live demo; confirm it runs without errors on the demo machine

### End-of-Week-3 Milestone
- `streamlit run demo/app.py` accepts two videos and outputs timestamped "off" intervals
- Evaluation results documented across ≥ 3 test pairs
- Presentation slides and demo video ready

---

## Deliverables Summary

| Deliverable | Owner | Target |
|------------|-------|--------|
| Pose extraction module (`pose_extraction.py`) | M1 | End of Week 1 |
| Normalization + DTW alignment module | M2 | End of Week 1–2 |
| 3+ open-source test pairs (trimmed, keypoints extracted) | M3 | End of Week 1 |
| Deviation scoring + `find_off_moments()` | M2 | End of Week 2 |
| Skeleton overlay + comparison video renderer | M4 | End of Week 2 |
| Streamlit app (basic, end-to-end) | M4 | End of Week 2 |
| Scoring validation & ablation | M3 | End of Week 3 |
| Polished demo + feedback text | M4 | End of Week 3 |
| Final report | All | End of Week 3 |
| Demo video + slides | M4 | End of Week 3 |

---

## Technical Architecture

### Pipeline at a Glance

```
benchmark.mp4 ──┐
                ├──► [1. Video Ingestion] ──► [2. Pose Extraction] ──► [3. Normalization]
learner.mp4   ──┘                                                              │
                                                                               ▼
                                                                    [4. DTW Alignment]
                                                                               │
                                                                               ▼
                                                              [5. Per-joint Error Computation]
                                                                               │
                                                                               ▼
                                                              [6. find_off_moments()]
                                                                     ↙            ↘
                                                      [7a. Comparison video]  [7b. Timeline chart]
                                                                     ↘            ↙
                                                               [8. Streamlit UI]
                                                         "At 3.2s–5.8s your right arm is off"
```

### Layer-by-Layer Detail

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — VIDEO INGESTION          src/video_io.py  (M4)                   │
│                                                                              │
│  load_video(path) → Iterator[frame]          BGR frames + timestamps         │
│  standardize(video, fps=30, res=(720,1280))  resample & resize               │
│  export_side_by_side(frames_A, frames_B, out_path)                           │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │ raw frames (np.ndarray)
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  LAYER 2 — POSE EXTRACTION          src/pose_extraction.py  (M1)            │
│                                                                              │
│  class PoseEstimator:                                                        │
│    backend: "mediapipe" | "mmpose"                                           │
│    .extract_sequence(video) → np.ndarray  shape (T, 17, 3)                  │
│                                           axis-2: [x_px, y_px, confidence]  │
│                                                                              │
│  smooth(seq)              Savitzky-Golay filter along time axis              │
│  interpolate_missing(seq) fill low-confidence frames by linear interp        │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │ KeypointSequence (T, 17, 3)
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  LAYER 3 — NORMALIZATION            src/normalization.py  (M2)              │
│                                                                              │
│  normalize(seq) → seq_norm                                                   │
│    1. Translate: hip midpoint → (0, 0)                                       │
│    2. Scale: divide by torso length (neck midpoint to hip midpoint)          │
│                                                                              │
│  Result: both sequences are comparable regardless of body size or            │
│  distance from camera                                                        │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │ normalized sequences
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  LAYER 4 — TEMPORAL ALIGNMENT       src/alignment.py  (M2)                  │
│                                                                              │
│  dtw_align(bench_norm, user_norm)                                            │
│    → warping_path: List[(i_bench, i_user)]                                   │
│    → bench_aligned, user_aligned  both shape (T', 17, 3)                     │
│                                                                              │
│  Library: fastdtw  (O(N) time and space)                                     │
│  Cost: mean Euclidean distance across all 17 joints per frame pair           │
│                                                                              │
│  Handles learners who perform the correct motion at a different speed        │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │ aligned sequences (T', 17, 3)
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  LAYER 5 — DEVIATION SCORING        src/scoring.py  (M2)                   │
│                                                                              │
│  compute_joint_errors(bench_aligned, user_aligned)                           │
│    → errors: np.ndarray  shape (T', 17)   per-joint error at each frame     │
│                                                                              │
│  BODY_PARTS grouping (COCO joint indices):                                   │
│    LEFT_ARM:  [5 shoulder, 7 elbow, 9 wrist]                                 │
│    RIGHT_ARM: [6 shoulder, 8 elbow, 10 wrist]                                │
│    LEFT_LEG:  [11 hip, 13 knee, 15 ankle]                                    │
│    RIGHT_LEG: [12 hip, 14 knee, 16 ankle]                                    │
│    TORSO:     [5, 6, 11, 12]                                                 │
│    HEAD:      [0 nose]                                                       │
│                                                                              │
│  per_part_error_over_time(errors)                                            │
│    → dict[part_name → np.ndarray (T',)]   mean error per part per frame     │
│                                                                              │
│  find_off_moments(part_errors, threshold=0.25, fps=30)                       │
│    → List[Interval(start_s, end_s, part_name, mean_error)]                  │
│      scans each part's error curve; flags contiguous runs > threshold        │
│      minimum interval length: 0.5 s (avoids single-frame noise)             │
│                                                                              │
│  overall_score(part_errors) → float  0–100                                  │
│    = 100 × (1 − mean_error / max_possible_error)                             │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │ List[Interval] + per_part_errors
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  LAYER 6 — VISUALIZATION & FEEDBACK                                          │
│                                                                              │
│  src/visualization.py  (M4)                                                  │
│    overlay_skeleton(frame, kp, part_errors_at_t)                             │
│      draws limbs; joints colored: green <0.15 / yellow 0.15–0.35 / red >0.35│
│    render_comparison_video(bench_frames, user_frames,                        │
│                            bench_kp, user_kp, intervals, out_path)          │
│      side-by-side MP4; red border during flagged intervals                   │
│    plot_error_timeline(part_errors, intervals, fps)                          │
│      Plotly line chart: time (s) vs. error per part; shaded "off" regions    │
│                                                                              │
│  src/feedback.py  (M4)                                                       │
│    generate_feedback(intervals) → List[str]                                  │
│      "At 3.2 s – 5.8 s your RIGHT ARM is significantly off."                │
│      "At 9.0 s – 11.4 s your LEFT LEG deviates from the benchmark."         │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  STREAMLIT APP    demo/app.py  (M4)                                          │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Upload Benchmark Video        Upload Your Video                    │    │
│  │  [  benchmark.mp4  ▼ ]         [  learner.mp4  ▼ ]                 │    │
│  │                    [ Run Analysis ]                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────┐  ┌────────────────────┐                            │
│  │  Benchmark          │  │  Your Performance  │  ← side-by-side playback  │
│  │  [skeleton overlay] │  │  [skeleton overlay]│    color-coded joints      │
│  └─────────────────────┘  └────────────────────┘                            │
│                                                                              │
│  Overall Similarity Score:  74 / 100                                         │
│                                                                              │
│  ──── Error Timeline ──────────────────────────────────────────────────     │
│  [Plotly chart: time vs. error per body part, shaded "off" intervals]        │
│                                                                              │
│  ──── Feedback ────────────────────────────────────────────────────────     │
│  • At 3.2 s – 5.8 s your RIGHT ARM is significantly off.                    │
│  • At 9.0 s – 11.4 s your LEFT LEG deviates from the benchmark.             │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

### Data Structures

```python
# Keypoint sequence for one video
# shape: (T, 17, 3)   axis-2 = [x_normalized, y_normalized, confidence]
KeypointSequence = np.ndarray

# COCO 17-joint index reference
#  0 nose   | 1 L_eye  | 2 R_eye  | 3 L_ear   | 4 R_ear
#  5 L_shldr| 6 R_shldr| 7 L_elbw | 8 R_elbw  | 9 L_wrst | 10 R_wrst
# 11 L_hip  |12 R_hip  |13 L_knee |14 R_knee  |15 L_ankl | 16 R_ankl

BODY_PARTS: dict[str, list[int]] = {
    "LEFT_ARM":   [5, 7, 9],
    "RIGHT_ARM":  [6, 8, 10],
    "LEFT_LEG":   [11, 13, 15],
    "RIGHT_LEG":  [12, 14, 16],
    "TORSO":      [5, 6, 11, 12],
    "HEAD":       [0],
}

@dataclass
class Interval:
    start_s:    float   # start time in seconds
    end_s:      float   # end time in seconds
    part:       str     # e.g. "RIGHT_ARM"
    mean_error: float   # average normalized deviation during this interval

@dataclass
class AnalysisResult:
    overall_score:      float               # 0–100
    intervals:          list[Interval]      # timestamped "off" segments
    part_errors:        dict[str, np.ndarray]  # part → (T',) error time series
    warping_path:       list[tuple[int, int]]  # DTW warping path
    feedback_lines:     list[str]           # human-readable sentences
```

---

### Module Dependency Graph

```
video_io.py  ←──────────────────────────── no external src deps
    ↓
pose_extraction.py  ←───────────────────── imports video_io
    ↓
normalization.py    ←───────────────────── imports nothing from src
    ↓
alignment.py        ←───────────────────── imports normalization
    ↓
scoring.py          ←───────────────────── imports alignment
    ↓
visualization.py    ←───────────────────── imports scoring, video_io
feedback.py         ←───────────────────── imports scoring
    ↓
demo/app.py         ←───────────────────── orchestrates everything
```

---

### File & Folder Layout

```
Final_CV/
│
├── data/
│   ├── phrase_01/
│   │   ├── benchmark.mp4        # open-source clip (AIST++ / MV)
│   │   ├── learner.mp4          # comparison clip
│   │   └── keypoints/
│   │       ├── benchmark_kp.npy   # (T, 17, 3)
│   │       └── learner_kp.npy
│   ├── phrase_02/  ...
│   └── phrase_03/  ...
│
├── src/
│   ├── video_io.py            # M4 — frame loading, standardization, export
│   ├── pose_extraction.py     # M1 — PoseEstimator, smoothing, interpolation
│   ├── normalization.py       # M2 — centering & scaling
│   ├── alignment.py           # M2 — DTW alignment
│   ├── scoring.py             # M2 — joint errors, find_off_moments, overall_score
│   ├── visualization.py       # M4 — skeleton overlay, video renderer, timeline chart
│   └── feedback.py            # M4 — interval → plain-English sentences
│
├── demo/
│   └── app.py                 # M4 — Streamlit orchestrator
│
├── notebooks/
│   ├── 01_pose_exploration.ipynb       # M1 — verify extractor output
│   ├── 02_alignment_experiments.ipynb  # M2 — tune DTW
│   └── 03_scoring_validation.ipynb     # M3 — ablation & threshold tuning
│
├── docs/
│   └── final_report.md
│
├── requirements.txt
└── README.md
```

---

### Key Algorithmic Choices

#### Pose Estimator
- **MediaPipe Pose** (default): runs on CPU, no extra install, 33 landmarks (we use the 17 COCO-compatible subset)
- **MMPose** (optional, GPU): higher accuracy on fast/partially occluded motion; swap in by changing `backend="mmpose"` in one place

#### Normalization
1. **Center** — shift all joints so hip midpoint = (0, 0) each frame
2. **Scale** — divide by torso length (neck midpoint to hip midpoint distance); eliminates height and camera-distance differences

#### DTW Alignment
- Allows the learner to perform the sequence at a different speed without being penalized
- Cost per frame pair = mean Euclidean distance across all 17 joints
- Library: `fastdtw` — O(N) time and space, suitable for 15–30 fps × 15 seconds

#### `find_off_moments` Logic
```
for each body part p:
    errors_p = per_part_error_over_time[p]   # shape (T',)
    mask = errors_p > threshold              # boolean array
    merge contiguous True runs into intervals
    discard intervals shorter than 0.5 s     # noise filter
    map frame indices back to timestamps     # using warping path
```

#### Deviation Threshold
- Default: `0.25` normalized units (roughly 25% of torso length)
- Tuned empirically in `03_scoring_validation.ipynb` using the 3 test pairs

#### Color Coding
| Level | Threshold | Color |
|-------|-----------|-------|
| Good | error < 0.15 | Green `#2ecc71` |
| Moderate | 0.15 – 0.35 | Yellow `#f39c12` |
| Off | > 0.35 | Red `#e74c3c` |

---

## Risk Register & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Pose estimator fails on fast / sideways motion | Medium | Confidence-based interpolation; fall back to MMPose on OSCAR |
| DTW too slow at 30 fps | Low | Downsample to 15 fps before alignment |
| Normalization fails when dancer is partially out of frame | Medium | Detect missing hip keypoints; skip normalization for those frames and flag to user |
| "Off" intervals too noisy (single-frame spikes flagged) | Medium | Minimum interval length filter (0.5 s) + Savitzky-Golay smoothing |
| Open-source clips have different camera angles | Medium | Restrict test pairs to frontal-camera clips; document as a known limitation |

---

## MVP Scope (must-have)

- Extract keypoints from any two uploaded 10–15 second videos
- Normalize for body size and camera distance
- DTW temporal alignment
- `find_off_moments()` → timestamped list of "off" intervals per body part
- Plain-English feedback: "At X s – Y s your RIGHT ARM is off"
- Side-by-side skeleton overlay video
- Error timeline chart

## Stretch Goals (nice-to-have)

- Live webcam capture instead of video upload
- Exportable PDF summary report
- Multi-angle support
