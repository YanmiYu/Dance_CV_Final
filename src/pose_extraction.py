"""
pose_extraction.py — PoseEstimator wrapper, smoothing, and interpolation.
Owner: Member 1

Supports MediaPipe Pose (default, CPU), MMPose (optional, GPU),
and SimpleBaseline (our trained model).
All backends produce keypoint arrays in the 17-joint COCO format.

COCO joint index reference:
  0 nose       1 L_eye    2 R_eye    3 L_ear    4 R_ear
  5 L_shoulder 6 R_shoulder 7 L_elbow 8 R_elbow
  9 L_wrist   10 R_wrist  11 L_hip  12 R_hip
 13 L_knee    14 R_knee   15 L_ankle 16 R_ankle
"""

from __future__ import annotations

import urllib.request
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter

# SimpleBaseline input/output sizes (must match training config)
_SB_IMAGE_SIZE  = (256, 192)   # (H, W)
_SB_HEATMAP_SIZE = (64, 48)    # (H, W)
_SB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_SB_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# MediaPipe landmark indices that map to COCO 17-joint order
_MP_TO_COCO: list[int] = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)
_MODEL_PATH = Path(__file__).parent.parent / "models" / "pose_landmarker_heavy.task"

# KeypointSequence shape: (T, 17, 3)  — axis-2: [x_px, y_px, confidence]
KeypointSequence = np.ndarray


def _ensure_model() -> Path:
    if not _MODEL_PATH.exists():
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"[pose_extraction] Downloading MediaPipe model → {_MODEL_PATH} ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[pose_extraction] Model download complete.")
    return _MODEL_PATH


class PoseEstimator:
    """Wraps a pose estimation backend and exposes a unified API.

    Parameters
    ----------
    backend : {"mediapipe", "mmpose"}
        Which pose estimator to use. MediaPipe is the default; it runs on
        CPU with no extra installation. MMPose requires a GPU and separate
        setup but gives higher accuracy on fast or partially occluded motion.
    """

    def __init__(self, backend: str = "mediapipe", checkpoint: str | None = None) -> None:
        self.backend = backend
        self.checkpoint = checkpoint  # path to .pth file, required for simple_baseline
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if self.backend == "mediapipe":
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            model_path = _ensure_model()
            base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
            options = mp_vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._model = mp_vision.PoseLandmarker.create_from_options(options)
        elif self.backend == "simple_baseline":
            import torch
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from models.simple_baseline import SimpleBaseline

            if self.checkpoint is None:
                raise ValueError("simple_baseline backend requires a checkpoint path.")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = SimpleBaseline(num_joints=17).to(device)
            state = torch.load(self.checkpoint, map_location=device)
            # support both raw state_dict and checkpoint dicts
            if "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
            model.eval()
            self._model = model
            self._device = device
        elif self.backend == "mmpose":
            raise NotImplementedError("MMPose backend not yet implemented.")
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")

    def extract(self, frame: np.ndarray) -> np.ndarray:
        """Extract keypoints from a single BGR frame.

        Returns
        -------
        np.ndarray, shape (17, 3)
            Each row is [x_px, y_px, confidence] in pixel coordinates.
            Confidence is 0.0 for joints not detected.
        """
        self._load_model()
        import cv2
        import mediapipe as mp
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.backend == "mediapipe":
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._model.detect(mp_image)
            kp = np.zeros((17, 3), dtype=np.float32)
            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                for coco_idx, mp_idx in enumerate(_MP_TO_COCO):
                    kp[coco_idx] = [lm[mp_idx].x * w,
                                    lm[mp_idx].y * h,
                                    lm[mp_idx].visibility]
            return kp

        if self.backend == "simple_baseline":
            import torch
            import cv2

            h, w = frame.shape[:2]
            # Resize full frame to network input size (single-person assumption)
            img_h, img_w = _SB_IMAGE_SIZE
            resized = cv2.resize(rgb, (img_w, img_h)).astype(np.float32) / 255.0
            resized = (resized - _SB_MEAN) / _SB_STD                        # (H, W, 3)
            tensor  = torch.from_numpy(resized.transpose(2, 0, 1)).unsqueeze(0)  # (1, 3, H, W)
            tensor  = tensor.to(self._device)

            with torch.no_grad():
                heatmaps = self._model(tensor)   # (1, 17, 64, 48)

            heatmaps = heatmaps[0].cpu().numpy()  # (17, 64, 48)
            hm_h, hm_w = _SB_HEATMAP_SIZE

            kp = np.zeros((17, 3), dtype=np.float32)
            for j in range(17):
                hm = heatmaps[j]
                flat_idx = hm.argmax()
                hm_x = float(flat_idx % hm_w)
                hm_y = float(flat_idx // hm_w)
                # Scale heatmap coords back to original pixel space
                kp[j, 0] = hm_x / hm_w * w
                kp[j, 1] = hm_y / hm_h * h
                kp[j, 2] = float(hm.max())   # peak value as confidence
            return kp

        raise NotImplementedError

    def extract_sequence(self, frames: list[np.ndarray]) -> KeypointSequence:
        """Extract keypoints from a list of frames.

        Returns
        -------
        KeypointSequence : np.ndarray, shape (T, 17, 3)
        """
        return np.stack([self.extract(f) for f in frames], axis=0)

    def close(self) -> None:
        if self._model is not None and self.backend == "mediapipe":
            self._model.close()
        self._model = None


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def smooth(seq: KeypointSequence, window: int = 7, polyorder: int = 2) -> KeypointSequence:
    """Apply Savitzky-Golay filter along the time axis to reduce jitter.

    Only smooths the x and y channels; confidence is left unchanged.
    """
    seq = seq.copy()
    T = seq.shape[0]
    win = min(window, T if T % 2 == 1 else T - 1)
    if win < polyorder + 1:
        return seq
    seq[:, :, 0] = savgol_filter(seq[:, :, 0], win, polyorder, axis=0)
    seq[:, :, 1] = savgol_filter(seq[:, :, 1], win, polyorder, axis=0)
    return seq


def interpolate_missing(
    seq: KeypointSequence,
    confidence_threshold: float = 0.3,
) -> KeypointSequence:
    """Fill frames where a joint's confidence is below threshold via linear interpolation."""
    seq = seq.copy()
    T, J, _ = seq.shape
    for j in range(J):
        low = seq[:, j, 2] < confidence_threshold
        if not low.any():
            continue
        good_idx = np.where(~low)[0]
        if len(good_idx) < 2:
            continue
        for ch in range(2):
            seq[:, j, ch] = np.interp(np.arange(T), good_idx, seq[good_idx, j, ch])
    return seq


def save_keypoints(seq: KeypointSequence, path: str | Path) -> None:
    """Save a keypoint sequence to a .npy file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), seq)


def load_keypoints(path: str | Path) -> KeypointSequence:
    """Load a keypoint sequence from a .npy file. Returns shape (T, 17, 3)."""
    return np.load(str(path))
