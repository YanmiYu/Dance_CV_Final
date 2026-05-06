"""GNN pose-encoder adapter — composes on top of an existing keypoint
PoseRunResult (typically HRNet's). Normalizes the keypoint sequence with
stevenmerge's normalize_sequence, then encodes with PoseGNNEncoder to
produce per-frame 128-dim embeddings.

This intentionally re-uses the upstream HRNet pass rather than running its
own detector + heatmap inference; the diagram's "GNN Pose Encoder" stage
operates on 2D keypoints, so we can borrow whichever pose model the
pipeline is already running.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from src.compare.embedding_features import encode_pose_sequence, load_pose_gnn_encoder
from src.compare.normalize_pose import NormalizeConfig, normalize_sequence
from src.pose.base import PoseRunResult


def encode_from_pose_result(
    upstream: PoseRunResult,
    *,
    checkpoint: str = "checkpoints/pose_gnn_encoder_basicdance_allgenre_c01_supcon.pt",
    device: str = "auto",
    normalize_cfg: Optional[NormalizeConfig] = None,
) -> PoseRunResult:
    if upstream.poses is None:
        raise ValueError("GNN adapter needs an upstream PoseRunResult with keypoints.")

    normalized, _mask = normalize_sequence(upstream.poses, normalize_cfg)
    # encode_pose_sequence expects (T, 17, 3); restore the conf channel.
    enc_input = np.concatenate(
        [normalized[..., :2].astype(np.float32),
         upstream.poses[..., 2:3].astype(np.float32)],
        axis=-1,
    )

    model, dev = load_pose_gnn_encoder(checkpoint, device=device)
    embeddings = encode_pose_sequence(model, enc_input, device=dev)

    return PoseRunResult(
        name="gnn_pose_encoder",
        poses=enc_input,           # normalized poses are still useful downstream
        embeddings=embeddings,     # (T, 128)
        bboxes=upstream.bboxes,
        fps=upstream.fps,
        num_frames=upstream.num_frames,
        width=upstream.width,
        height=upstream.height,
        extra={"checkpoint": checkpoint, "upstream": upstream.name},
    )
