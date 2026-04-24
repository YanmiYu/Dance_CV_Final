"""Full-frame single-person detector dataset.

Reads the same internal JSONL schema produced by
``scripts/prepare_aist_training_data.py`` (see
``docs/project_decisions.md`` section 6), but -- unlike the pose dataset
-- does NOT crop to the top-down bbox. The detector has to localize the
person inside the WHOLE frame, so we feed the whole (resized) frame and
use ``bbox_xyxy`` as the only supervision signal.

Noisy-background synthesis
--------------------------
AIST++ backgrounds are clean / studio, but the downstream inference runs
on dance videos recorded by users in arbitrary environments. To make the
detector robust without introducing any new labeled dataset, every
training sample has a configurable probability of having its OUTSIDE-
bbox pixels replaced by one of:

  - solid random color
  - additive gaussian noise
  - a random other training frame (pasted background)

This gives the network many foreground/background combinations from the
same AIST++ frames and teaches it to localize the dancer rather than
memorize the studio pattern.

Returned numpy dict (one sample)::

    image            (3, H, W) float32 in [0, 1]
    center_heatmap   (1, H/4, W/4) float32 in [0, 1]
    size_target      (2, H/4, W/4) float32; only valid at ``size_mask > 0``
    size_mask        (1, H/4, W/4) float32; 1.0 at the GT center cell
    bbox_xyxy_input  (4,)        GT bbox in detector input pixels
    image_path       str
    image_id         str
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.io import read_jsonl


@dataclass
class DetectorAugConfig:
    color_jitter_brightness: float = 0.3
    color_jitter_contrast: float = 0.3
    gaussian_blur_prob: float = 0.2
    motion_blur_prob: float = 0.1
    jpeg_prob: float = 0.3
    jpeg_quality: Tuple[int, int] = (55, 95)
    horizontal_flip_prob: float = 0.5
    scale_jitter: Tuple[float, float] = (0.8, 1.2)
    translation_jitter_ratio: float = 0.1

    noisy_bg_prob: float = 0.5
    noisy_bg_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    feather_ratio: float = 0.05

    # External background-texture library. Strictly UNLABELED images used
    # only as composite backgrounds (see docs/project_decisions.md
    # section 6 revision 2026-04-23). When ``None`` or the directory is
    # empty, the ``"frame"`` background mode falls back to sampling
    # another AIST++ frame (the original behavior).
    external_bg_dir: Optional[str] = None

    # How to isolate the dancer from the AIST background before pasting
    # them onto a different background. ``"none"`` keeps the (legacy)
    # rectangular bbox+feather paste; ``"color"`` uses a fast HSV/color-
    # distance silhouette that exploits AIST's near-uniform background;
    # ``"grabcut"`` runs OpenCV's GrabCut with ``grabcut_iters``
    # iterations -- higher quality but ~10x slower per sample.
    silhouette_method: str = "color"
    silhouette_thresh: float = 28.0
    grabcut_iters: int = 2

    @classmethod
    def from_yaml(cls, d: Dict) -> "DetectorAugConfig":
        if not d:
            return cls()
        cj = d.get("color_jitter", {})
        return cls(
            color_jitter_brightness=float(cj.get("brightness", 0.3)),
            color_jitter_contrast=float(cj.get("contrast", 0.3)),
            gaussian_blur_prob=float(d.get("gaussian_blur_prob", 0.2)),
            motion_blur_prob=float(d.get("motion_blur_prob", 0.1)),
            jpeg_prob=float(d.get("jpeg_prob", 0.3)),
            jpeg_quality=tuple(d.get("jpeg_quality", (55, 95))),
            horizontal_flip_prob=float(d.get("horizontal_flip_prob", 0.5)),
            scale_jitter=tuple(d.get("scale_jitter", (0.8, 1.2))),
            translation_jitter_ratio=float(d.get("translation_jitter_ratio", 0.1)),
            noisy_bg_prob=float(d.get("noisy_bg_prob", 0.5)),
            noisy_bg_weights=tuple(d.get("noisy_bg_weights", (1.0, 1.0, 1.0))),
            feather_ratio=float(d.get("feather_ratio", 0.05)),
            external_bg_dir=d.get("external_bg_dir"),
            silhouette_method=str(d.get("silhouette_method", "color")).lower(),
            silhouette_thresh=float(d.get("silhouette_thresh", 28.0)),
            grabcut_iters=int(d.get("grabcut_iters", 2)),
        )


def _letterbox_bbox(bbox: np.ndarray, src_hw: Tuple[int, int], dst_hw: Tuple[int, int]) -> np.ndarray:
    """Scale an xyxy bbox from ``src_hw`` image space to ``dst_hw`` image space (simple resize)."""
    sh, sw = src_hw
    dh, dw = dst_hw
    sx = dw / float(sw)
    sy = dh / float(sh)
    x1, y1, x2, y2 = bbox
    return np.asarray([x1 * sx, y1 * sy, x2 * sx, y2 * sy], dtype=np.float32)


def make_center_heatmap(
    out_h: int, out_w: int, cx: float, cy: float, bw: float, bh: float,
) -> np.ndarray:
    """Gaussian splat at (cx, cy) with sigma derived from bbox size.

    Uses the CornerNet / CenterNet radius rule adapted to our needs: sigma
    scales with the smaller bbox side so that tiny people get a tight bump
    and tall dancers a wide one.
    """
    heatmap = np.zeros((out_h, out_w), dtype=np.float32)
    if not (0 <= cx < out_w and 0 <= cy < out_h):
        return heatmap
    sigma = max(1.0, 0.1 * min(bw, bh))
    radius = int(math.ceil(sigma * 3))
    x_lo = max(0, int(cx) - radius); x_hi = min(out_w - 1, int(cx) + radius)
    y_lo = max(0, int(cy) - radius); y_hi = min(out_h - 1, int(cy) + radius)
    if x_hi < x_lo or y_hi < y_lo:
        return heatmap
    xs = np.arange(x_lo, x_hi + 1, dtype=np.float32)
    ys = np.arange(y_lo, y_hi + 1, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    g = np.exp(-((xv - cx) ** 2 + (yv - cy) ** 2) / (2.0 * sigma * sigma))
    heatmap[y_lo : y_hi + 1, x_lo : x_hi + 1] = np.maximum(
        heatmap[y_lo : y_hi + 1, x_lo : x_hi + 1], g
    )
    return heatmap


class DetectorDataset:
    """Map-style, PyTorch-agnostic detector dataset."""

    def __init__(
        self,
        annotations_jsonl: str | Path,
        *,
        input_size: Tuple[int, int] = (256, 256),
        output_stride: int = 4,
        is_train: bool = True,
        aug: Optional[DetectorAugConfig] = None,
        max_items: Optional[int] = None,
    ) -> None:
        self.records: List[Dict] = list(read_jsonl(annotations_jsonl))
        if max_items is not None:
            self.records = self.records[: int(max_items)]
        self.input_size = tuple(input_size)
        self.output_stride = int(output_stride)
        assert self.input_size[0] % self.output_stride == 0
        assert self.input_size[1] % self.output_stride == 0
        self.is_train = is_train
        self.aug = aug or DetectorAugConfig()
        self._n = len(self.records)
        self._external_bgs: List[str] = self._scan_external_bgs(self.aug.external_bg_dir)
        if self.is_train and self._external_bgs:
            print(
                f"[DetectorDataset] using {len(self._external_bgs)} external "
                f"background textures from {self.aug.external_bg_dir}"
            )

    def __len__(self) -> int:
        return self._n

    # ------------------------------------------------------------------
    # loading + geometric transforms
    # ------------------------------------------------------------------
    def _load_frame(self, rec: Dict) -> np.ndarray:
        img = cv2.imread(rec["image_path"], cv2.IMREAD_COLOR)
        if img is None:
            img = np.full((480, 640, 3), 128, dtype=np.uint8)
        return img

    def _resize(self, img: np.ndarray) -> np.ndarray:
        H, W = self.input_size
        return cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

    def _apply_flip(self, img: np.ndarray, bbox: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_train or random.random() > self.aug.horizontal_flip_prob:
            return img, bbox
        w = img.shape[1]
        img = img[:, ::-1, :].copy()
        x1, y1, x2, y2 = bbox
        bbox = np.asarray([w - 1 - x2, y1, w - 1 - x1, y2], dtype=np.float32)
        return img, bbox

    def _apply_scale_translate(self, img: np.ndarray, bbox: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Random scale + translation via an affine warp in input-resolution space."""
        H, W = self.input_size
        if not self.is_train:
            return img, bbox
        s_lo, s_hi = self.aug.scale_jitter
        scale = random.uniform(s_lo, s_hi)
        tx = random.uniform(-1, 1) * self.aug.translation_jitter_ratio * W
        ty = random.uniform(-1, 1) * self.aug.translation_jitter_ratio * H
        cx, cy = W / 2.0, H / 2.0
        M = np.array(
            [
                [scale, 0.0, (1 - scale) * cx + tx],
                [0.0, scale, (1 - scale) * cy + ty],
            ],
            dtype=np.float32,
        )
        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        pts = np.array(
            [[bbox[0], bbox[1], 1.0], [bbox[2], bbox[3], 1.0]], dtype=np.float32
        )
        new_pts = pts @ M.T
        x1, y1 = new_pts[0]; x2, y2 = new_pts[1]
        bbox = np.asarray([
            float(max(0.0, min(W - 1, min(x1, x2)))),
            float(max(0.0, min(H - 1, min(y1, y2)))),
            float(max(0.0, min(W - 1, max(x1, x2)))),
            float(max(0.0, min(H - 1, max(y1, y2)))),
        ], dtype=np.float32)
        return img, bbox

    # ------------------------------------------------------------------
    # appearance augmentations
    # ------------------------------------------------------------------
    def _color_jitter(self, img: np.ndarray) -> np.ndarray:
        if not self.is_train:
            return img
        out = img.astype(np.float32)
        if self.aug.color_jitter_brightness > 0:
            b = random.uniform(1 - self.aug.color_jitter_brightness, 1 + self.aug.color_jitter_brightness)
            out = out * b
        if self.aug.color_jitter_contrast > 0:
            c = random.uniform(1 - self.aug.color_jitter_contrast, 1 + self.aug.color_jitter_contrast)
            mean = out.mean()
            out = (out - mean) * c + mean
        return np.clip(out, 0, 255).astype(np.uint8)

    def _maybe_blur(self, img: np.ndarray) -> np.ndarray:
        if not self.is_train:
            return img
        if random.random() < self.aug.gaussian_blur_prob:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)
        if random.random() < self.aug.motion_blur_prob:
            k = random.choice([5, 7, 9])
            kernel = np.zeros((k, k), dtype=np.float32)
            kernel[k // 2, :] = 1.0 / k
            img = cv2.filter2D(img, -1, kernel)
        return img

    def _maybe_jpeg(self, img: np.ndarray) -> np.ndarray:
        if not self.is_train or random.random() > self.aug.jpeg_prob:
            return img
        q = random.randint(self.aug.jpeg_quality[0], self.aug.jpeg_quality[1])
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            return img
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)

    # ------------------------------------------------------------------
    # noisy-background compositing (the whole point of this dataset)
    # ------------------------------------------------------------------
    _EXTERNAL_BG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    @classmethod
    def _scan_external_bgs(cls, root: Optional[str]) -> List[str]:
        """Cache the list of external background-texture image paths.

        Returns an empty list if ``root`` is ``None`` or does not exist;
        the caller silently falls back to the AIST-frame paste mode.
        See ``docs/project_decisions.md`` section 6 (revision 2026-04-23):
        these images are STRICTLY unlabeled augmentation textures.
        """
        if not root:
            return []
        p = Path(root)
        if not p.exists() or not p.is_dir():
            return []
        out: List[str] = []
        for f in sorted(p.rglob("*")):
            if f.is_file() and f.suffix.lower() in cls._EXTERNAL_BG_EXTS:
                out.append(str(f))
        return out

    def _sample_random_frame(self) -> Optional[np.ndarray]:
        if self._n <= 1:
            return None
        for _ in range(3):
            rec = self.records[random.randrange(self._n)]
            img = cv2.imread(rec["image_path"], cv2.IMREAD_COLOR)
            if img is not None:
                return self._resize(img)
        return None

    def _sample_external_bg(self) -> Optional[np.ndarray]:
        if not self._external_bgs:
            return None
        for _ in range(3):
            path = self._external_bgs[random.randrange(len(self._external_bgs))]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                return self._resize(img)
        return None

    def _make_bg(self, kind: str) -> np.ndarray:
        H, W = self.input_size
        if kind == "frame":
            # Prefer external textures (real-world variety); fall back to a
            # random AIST frame, which still helps a little even on AIST's
            # uniform studio.
            bg = self._sample_external_bg()
            if bg is None:
                bg = self._sample_random_frame()
            if bg is not None:
                return bg
            kind = "noise"
        if kind == "noise":
            base = np.random.randint(0, 256, size=3, dtype=np.uint8)
            noise = np.random.normal(loc=0.0, scale=40.0, size=(H, W, 3))
            out = np.clip(base.astype(np.float32)[None, None, :] + noise, 0, 255).astype(np.uint8)
            return out
        color = np.random.randint(0, 256, size=3, dtype=np.uint8)
        return np.full((H, W, 3), color, dtype=np.uint8)

    # ------------------------------------------------------------------
    # silhouette extraction (separating dancer from AIST background)
    # ------------------------------------------------------------------
    def _silhouette_color(self, img_bgr: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """Cheap silhouette: foreground = pixels far from background color.

        AIST videos have an almost-uniform background, so sampling the
        outer border of the frame gives a tight estimate of the background
        color. Anything inside the bbox that is more than
        ``silhouette_thresh`` away from that color (in BGR L2 distance) is
        treated as foreground. ~3 ms per call at 256x256.
        """
        H, W = img_bgr.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1 = max(0, min(W - 1, x1)); x2 = max(x1 + 1, min(W, x2))
        y1 = max(0, min(H - 1, y1)); y2 = max(y1 + 1, min(H, y2))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        border = max(2, min(H, W) // 32)
        samples = np.concatenate(
            [
                img_bgr[:border, :].reshape(-1, 3),
                img_bgr[-border:, :].reshape(-1, 3),
                img_bgr[:, :border].reshape(-1, 3),
                img_bgr[:, -border:].reshape(-1, 3),
            ],
            axis=0,
        ).astype(np.float32)
        bg_mean = samples.mean(axis=0)
        diff = img_bgr.astype(np.float32) - bg_mean
        dist = np.sqrt((diff * diff).sum(axis=-1))
        fg = (dist > self.aug.silhouette_thresh).astype(np.uint8)
        bbox_mask = np.zeros((H, W), dtype=np.uint8)
        bbox_mask[y1:y2, x1:x2] = 1
        fg = fg & bbox_mask
        if fg.sum() < 0.05 * (x2 - x1) * (y2 - y1):
            return None
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
        return fg.astype(np.float32)

    def _silhouette_grabcut(self, img_bgr: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """GrabCut silhouette. ~50-150 ms per call; far higher quality."""
        H, W = img_bgr.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1 = max(0, min(W - 1, x1)); x2 = max(x1 + 1, min(W, x2))
        y1 = max(0, min(H - 1, y1)); y2 = max(y1 + 1, min(H, y2))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        rect = (x1, y1, x2 - x1, y2 - y1)
        mask = np.zeros((H, W), dtype=np.uint8)
        bgd = np.zeros((1, 65), dtype=np.float64)
        fgd = np.zeros((1, 65), dtype=np.float64)
        try:
            cv2.grabCut(img_bgr, mask, rect, bgd, fgd,
                        max(1, int(self.aug.grabcut_iters)),
                        cv2.GC_INIT_WITH_RECT)
        except cv2.error:
            return None
        fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.float32)
        if fg.sum() < 0.05 * (x2 - x1) * (y2 - y1):
            return None
        return fg

    def _foreground_mask(self, img_bgr: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """Dispatch on ``silhouette_method``; returns a (H, W) float mask in [0, 1]."""
        method = self.aug.silhouette_method
        if method == "color":
            return self._silhouette_color(img_bgr, bbox)
        if method == "grabcut":
            return self._silhouette_grabcut(img_bgr, bbox)
        return None

    def _apply_noisy_bg(self, img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        if not self.is_train or random.random() > self.aug.noisy_bg_prob:
            return img
        weights = np.asarray(self.aug.noisy_bg_weights, dtype=np.float32)
        weights = weights / max(weights.sum(), 1e-6)
        kind = np.random.choice(["color", "noise", "frame"], p=weights)
        bg = self._make_bg(kind)
        H, W = self.input_size
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W, x2))
        y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H, y2))
        if x2 <= x1 or y2 <= y1:
            return img

        # Build the foreground mask. Try the configured silhouette method
        # first; if it fails (tiny bbox, GrabCut error, mostly-empty mask)
        # fall back to the legacy rectangular bbox+feather paste so
        # training never crashes.
        mask = self._foreground_mask(img, np.asarray([x1, y1, x2, y2], dtype=np.float32))
        if mask is None:
            mask = np.zeros((H, W), dtype=np.float32)
            mask[y1:y2, x1:x2] = 1.0

        bbox_min_side = min(x2 - x1, y2 - y1)
        feather = int(round(self.aug.feather_ratio * bbox_min_side))
        if feather > 0:
            k = max(3, feather * 2 + 1)
            if k % 2 == 0:
                k += 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)
        mask = np.clip(mask, 0.0, 1.0)[..., None]
        out = img.astype(np.float32) * mask + bg.astype(np.float32) * (1.0 - mask)
        return np.clip(out, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # target construction
    # ------------------------------------------------------------------
    def _build_targets(self, bbox: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        H, W = self.input_size
        s = self.output_stride
        out_h, out_w = H // s, W // s
        x1, y1, x2, y2 = bbox
        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))
        cx_in = (x1 + x2) * 0.5
        cy_in = (y1 + y2) * 0.5
        cx_out = cx_in / s
        cy_out = cy_in / s
        heatmap = make_center_heatmap(out_h, out_w, cx_out, cy_out, bw / s, bh / s)

        size_target = np.zeros((2, out_h, out_w), dtype=np.float32)
        size_mask = np.zeros((1, out_h, out_w), dtype=np.float32)
        ix = int(round(cx_out)); iy = int(round(cy_out))
        if 0 <= ix < out_w and 0 <= iy < out_h:
            size_target[0, iy, ix] = bw / W
            size_target[1, iy, ix] = bh / H
            size_mask[0, iy, ix] = 1.0
        return heatmap[None], size_target, size_mask

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        rec = self.records[idx]
        img = self._load_frame(rec)
        src_hw = img.shape[:2]
        img = self._resize(img)

        bbox_src = np.asarray(rec["bbox_xyxy"], dtype=np.float32)
        bbox = _letterbox_bbox(bbox_src, src_hw, self.input_size)

        img, bbox = self._apply_flip(img, bbox)
        img, bbox = self._apply_scale_translate(img, bbox)
        img = self._apply_noisy_bg(img, bbox)
        img = self._maybe_blur(img)
        img = self._color_jitter(img)
        img = self._maybe_jpeg(img)

        heatmap, size_target, size_mask = self._build_targets(bbox)

        img_chw = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)
        return {
            "image": img_chw,
            "center_heatmap": heatmap,
            "size_target": size_target,
            "size_mask": size_mask,
            "bbox_xyxy_input": bbox.astype(np.float32),
            "image_path": rec["image_path"],
            "image_id": rec.get("image_id", ""),
        }
