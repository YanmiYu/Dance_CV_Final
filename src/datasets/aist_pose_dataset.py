"""AIST++ pose dataset (same internal JSONL schema as COCO/CrowdPose)."""
from __future__ import annotations

from .coco_pose_dataset import PoseJsonlDataset


class AistPoseDataset(PoseJsonlDataset):
    pass
