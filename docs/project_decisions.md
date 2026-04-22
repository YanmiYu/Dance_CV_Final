# Project Decisions (FROZEN)

This file is the single source of truth for the project scope. Every config,
script, and module MUST refer back to these decisions. Do not relax any of
these without updating this document first.

## 1. No pretrained weights — ever

- No pretrained pose model (OpenPose / MediaPipe / MMPose / ViTPose / ...).
- No pretrained object/person detector.
- No ImageNet / COCO-Detection initialization for any backbone.
- No self-supervised foundation-model checkpoints.
- All weights are initialized randomly (Kaiming / normal) and trained from
  scratch on labeled keypoint data we prepare ourselves.

Enforcement: model factory functions in `src/models/` MUST NOT load weights
from any external source. Loading state dicts is only permitted from our own
training checkpoints saved under `data/processed/checkpoints/`.

## 2. Task definition

- Single-person 2D pose extraction only.
- Short clips only: 10-20 seconds.
- Fixed camera (or near-fixed) only for v1.
- Downstream task: compare one benchmark clip vs one imitation clip.

## 3. Canonical joint format

- COCO 17-keypoint layout, everywhere:
  0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
  5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
  9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
  13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle.
- Flip map (for horizontal-flip augmentation) is defined in
  `src/datasets/common.py` and is the ONLY allowed source.

## 4. MVP scope

- Predict full body.
- In scoring, weight upper body more heavily than legs (see
  `configs/data/compare.yaml::body_part_weights`).

## 5. Training paradigm

- Per-frame supervised pose training FIRST (stages A and B).
- Video alignment / scoring / temporal refinement comes AFTER.
- Do NOT start with an end-to-end video transformer.

## 6. Data policy

- `data/raw_urls/*.csv`: download lists for unlabeled dance video (AIST).
- Supervised pose labels come only from COCO, CrowdPose, and AIST++.
- Our own paired benchmark/imitation videos are for downstream evaluation
  and calibration, NOT for supervised pose-label training (unless explicitly
  hand-annotated into `data/labels/custom_dance_val/`).

## 7. Pipeline order (never skip forward)

1. Repo skeleton + configs.
2. CSV ingestion + manifests.
3. Pilot downloads + ffprobe integrity.
4. MVP subset curation.
5. Paired benchmark/imitation manifest.
6. Label conversion (COCO / CrowdPose / AIST++).
7. Simple Baseline model — prove the training loop.
8. HRNet-W32 — final model.
9. Staged training A -> B -> (optional) C.
10. Motion-based video inference.
11. Temporal smoothing.
12. Normalization -> features -> DTW -> score -> feedback.
13. Streamlit demo (last, not first).

## 8. Explicit non-goals for v1

- No multi-person pose.
- No moving-camera clips.
- No training on raw unlabeled CSV videos without pseudo-labels.
- No transformer as the first model.
- No UI before CLI pipeline works.
- No single "black-box" score without per-part / per-window diagnostics.
