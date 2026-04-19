# Dance Choreography Practice Tool

A computer vision tool that compares a learner's dance video against a benchmark and tells them exactly **when** and **which body parts** are off.

## Setup

```bash
pip install -r requirements.txt
```

## Run the Demo

```bash
streamlit run demo/app.py
```

Upload a benchmark video and a learner video (10–15 seconds, frontal camera angle).  
The system outputs:
- A side-by-side comparison video with color-coded skeleton overlays
- A timeline chart showing error per body part over time
- Plain-English feedback: *"At 3.2 s – 5.8 s your RIGHT ARM is off."*

## Project Structure

```
src/
  video_io.py           Frame loading, standardization, export
  pose_extraction.py    MediaPipe-based keypoint extraction + smoothing
  normalization.py      Body-size and camera-distance normalization
  alignment.py          DTW temporal alignment
  scoring.py            Per-joint error, find_off_moments(), overall score
  visualization.py      Skeleton overlay, comparison video, timeline chart
  feedback.py           Timestamped intervals → plain-English sentences

demo/app.py             Streamlit UI (orchestrates the full pipeline)
data/phrase_XX/         Test video pairs + extracted keypoints (.npy)
notebooks/              Exploration and validation notebooks
docs/final_report.md    Final report
```

## Data Format

Keypoint arrays are saved as `.npy` files with shape `(T, 17, 3)`:
- axis 0: frame index
- axis 1: joint index (COCO 17-joint layout)
- axis 2: `[x_normalized, y_normalized, confidence]`

## COCO Joint Index Reference

```
 0 nose       1 L_eye    2 R_eye    3 L_ear    4 R_ear
 5 L_shoulder 6 R_shoulder 7 L_elbow 8 R_elbow
 9 L_wrist   10 R_wrist  11 L_hip  12 R_hip
13 L_knee    14 R_knee   15 L_ankle 16 R_ankle
```
