"""CrowdPose reads the SAME internal JSONL schema produced by
``src.data.convert_crowdpose``. We therefore just re-export the generic
loader with a different default ``dataset_name`` for logging.
"""
from __future__ import annotations

from .coco_pose_dataset import PoseJsonlDataset


class CrowdPoseDataset(PoseJsonlDataset):
    pass
