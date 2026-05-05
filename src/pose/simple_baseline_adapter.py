"""SimpleBaseline pose adapter — same wrapper as HRNet but with the
SB config and checkpoint. Ckpt path must live under data/processed/ to
satisfy the engine's safety check.
"""
from __future__ import annotations

from typing import Optional

from src.pose.base import PoseRunResult
from src.pose.hrnet_adapter import run_video as _run_video


def run_video(
    video_path: str,
    *,
    model_config: str = "configs/model/simple_baseline.yaml",
    checkpoint: str = "data/processed/simple_baseline/best.pt",
    device: Optional[str] = None,
    out_dir: Optional[str] = None,
    inference: Optional[dict] = None,
) -> PoseRunResult:
    return _run_video(
        video_path,
        model_config=model_config,
        checkpoint=checkpoint,
        device=device,
        name="simple_baseline",
        out_dir=out_dir,
        inference=inference,
    )
