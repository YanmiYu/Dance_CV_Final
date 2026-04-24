"""Training CLI for the single-person detector.

Trained FROM SCRATCH on the same AIST++ 2D keypoint labels used for pose
(``docs/project_decisions.md`` section 6); the ``bbox_xyxy`` present in
every row is the only supervision the detector needs.

Usage::

    python -m src.train.train_detector --train configs/train/train_detector.yaml
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.datasets.detector_dataset import DetectorAugConfig, DetectorDataset
from src.models.person_detector import PersonDetector
from src.train.engine import _TorchDatasetAdapter, _collate, build_optimizer, build_scheduler, _load_state_from_internal_ckpt
from src.utils.config import load_yaml
from src.utils.seed import seed_everything


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


class DetectorLoss(nn.Module):
    """MSE on center heatmap + masked L1 on size.

    Simple and robust; good enough for single-person supervision where the
    heatmap is already spatially unambiguous. Can be swapped for focal loss
    later without touching the dataset or the model.
    """

    def __init__(self, center_weight: float = 1.0, size_weight: float = 0.1) -> None:
        super().__init__()
        self.center_weight = float(center_weight)
        self.size_weight = float(size_weight)

    def forward(
        self,
        preds: torch.Tensor,
        center_gt: torch.Tensor,
        size_gt: torch.Tensor,
        size_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        center_pred = torch.sigmoid(preds[:, :1])
        size_pred = preds[:, 1:]

        center_loss = F.mse_loss(center_pred, center_gt)

        valid = size_mask.sum().clamp_min(1.0)
        size_l1 = (size_pred - size_gt).abs() * size_mask
        size_loss = size_l1.sum() / valid

        total = self.center_weight * center_loss + self.size_weight * size_loss
        return {"loss": total, "center_loss": center_loss.detach(), "size_loss": size_loss.detach()}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _decode_bbox(preds: torch.Tensor, stride: int, input_hw) -> torch.Tensor:
    """Batch-decode predictions -> (B, 4) xyxy in input pixels. Argmax only."""
    B = preds.shape[0]
    center = torch.sigmoid(preds[:, 0])
    H, W = input_hw
    out_h, out_w = center.shape[-2], center.shape[-1]
    flat = center.view(B, -1)
    _, idx = flat.max(dim=1)
    cy = (idx // out_w).float()
    cx = (idx % out_w).float()
    cx_in = cx * stride
    cy_in = cy * stride
    size = preds[:, 1:]
    batch_ix = torch.arange(B, device=preds.device)
    cy_i = cy.long().clamp(0, out_h - 1)
    cx_i = cx.long().clamp(0, out_w - 1)
    bw = size[batch_ix, 0, cy_i, cx_i] * W
    bh = size[batch_ix, 1, cy_i, cx_i] * H
    x1 = cx_in - bw / 2
    y1 = cy_in - bh / 2
    x2 = cx_in + bw / 2
    y2 = cy_in + bh / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


def _iou(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    ix1 = torch.maximum(pred[:, 0], gt[:, 0])
    iy1 = torch.maximum(pred[:, 1], gt[:, 1])
    ix2 = torch.minimum(pred[:, 2], gt[:, 2])
    iy2 = torch.minimum(pred[:, 3], gt[:, 3])
    iw = (ix2 - ix1).clamp_min(0)
    ih = (iy2 - iy1).clamp_min(0)
    inter = iw * ih
    pa = (pred[:, 2] - pred[:, 0]).clamp_min(0) * (pred[:, 3] - pred[:, 1]).clamp_min(0)
    ga = (gt[:, 2] - gt[:, 0]).clamp_min(0) * (gt[:, 3] - gt[:, 1]).clamp_min(0)
    union = pa + ga - inter
    return inter / union.clamp_min(1e-6)


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class DetectorTrainCtx:
    device: torch.device
    model: nn.Module
    optim: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler
    loss_fn: nn.Module
    output_dir: Path
    input_hw: tuple
    log_every_n_steps: int = 50
    save_every_n_epochs: int = 5
    eval_every_n_epochs: int = 2


def _require_no_external_weights(model_cfg: Dict) -> None:
    if model_cfg.get("pretrained", False):
        raise SystemExit(
            "Refusing to run: model_config.pretrained is true. "
            "docs/project_decisions.md section 1 forbids pretrained weights."
        )


def make_detector_ctx(train_cfg: Dict, model_cfg: Dict, device: Optional[torch.device] = None) -> DetectorTrainCtx:
    _require_no_external_weights(model_cfg)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PersonDetector(model_cfg).to(device)
    if (init_from := train_cfg.get("init_from")):
        _load_state_from_internal_ckpt(model, init_from)
    optim = build_optimizer(model, train_cfg.get("optimizer", {}))
    scheduler = build_scheduler(optim, train_cfg.get("scheduler", {}), int(train_cfg.get("epochs", 1)))
    loss_cfg = train_cfg.get("loss", {})
    loss_fn = DetectorLoss(
        center_weight=float(loss_cfg.get("center_weight", 1.0)),
        size_weight=float(loss_cfg.get("size_weight", 0.1)),
    ).to(device)
    input_hw = tuple(train_cfg.get("input_size", (256, 256)))
    return DetectorTrainCtx(
        device=device,
        model=model,
        optim=optim,
        scheduler=scheduler,
        loss_fn=loss_fn,
        output_dir=Path(train_cfg["output_dir"]),
        input_hw=input_hw,
        log_every_n_steps=int(train_cfg.get("log_every_n_steps", 50)),
        save_every_n_epochs=int(train_cfg.get("save_every_n_epochs", 5)),
        eval_every_n_epochs=int(train_cfg.get("eval_every_n_epochs", 2)),
    )


def _build_datasets(data_cfg: Dict, train_cfg: Dict, max_items: Optional[int]) -> tuple:
    input_hw = tuple(train_cfg.get("input_size", (256, 256)))
    stride = int(train_cfg.get("output_stride", 4))
    src = data_cfg["detector"]
    aug = DetectorAugConfig.from_yaml(data_cfg.get("augmentations", {}))
    train_ds = DetectorDataset(
        src["annotations"],
        input_size=input_hw,
        output_stride=stride,
        is_train=True,
        aug=aug,
        max_items=max_items,
    )
    val_ds = DetectorDataset(
        src["val_annotations"],
        input_size=input_hw,
        output_stride=stride,
        is_train=False,
        aug=aug,
        max_items=max_items,
    )
    return train_ds, val_ds


def _make_loader(ds, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        _TorchDatasetAdapter(ds),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate,
        drop_last=shuffle,
        pin_memory=False,
    )


def train_one_epoch(ctx: DetectorTrainCtx, loader: DataLoader, epoch: int) -> Dict[str, float]:
    ctx.model.train()
    running = {"loss": 0.0, "center_loss": 0.0, "size_loss": 0.0}
    n_steps = 0
    for step, batch in enumerate(loader):
        images = batch["image"].to(ctx.device)
        center = batch["center_heatmap"].to(ctx.device)
        size_t = batch["size_target"].to(ctx.device)
        size_m = batch["size_mask"].to(ctx.device)
        preds = ctx.model(images)
        losses = ctx.loss_fn(preds, center, size_t, size_m)
        ctx.optim.zero_grad(set_to_none=True)
        losses["loss"].backward()
        ctx.optim.step()
        for k in running:
            running[k] += float(losses[k].item())
        n_steps += 1
        if step % ctx.log_every_n_steps == 0:
            print(
                f"epoch {epoch} step {step} "
                f"loss={losses['loss'].item():.5f} "
                f"center={losses['center_loss'].item():.5f} "
                f"size={losses['size_loss'].item():.5f}"
            )
    ctx.scheduler.step()
    return {f"train_{k}": v / max(n_steps, 1) for k, v in running.items()}


@torch.no_grad()
def evaluate(ctx: DetectorTrainCtx, loader: DataLoader, stride: int) -> Dict[str, float]:
    ctx.model.eval()
    losses: list[float] = []
    ious: list[float] = []
    for batch in loader:
        images = batch["image"].to(ctx.device)
        center = batch["center_heatmap"].to(ctx.device)
        size_t = batch["size_target"].to(ctx.device)
        size_m = batch["size_mask"].to(ctx.device)
        preds = ctx.model(images)
        out = ctx.loss_fn(preds, center, size_t, size_m)
        losses.append(float(out["loss"].item()))

        gt = batch["bbox_xyxy_input"].to(ctx.device)
        pred_bbox = _decode_bbox(preds, stride, ctx.input_hw)
        ious.append(float(_iou(pred_bbox, gt).mean().item()))
    return {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "val_iou": float(np.mean(ious)) if ious else float("nan"),
    }


def save_checkpoint(ctx: DetectorTrainCtx, path: Path, extra: Optional[Dict] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": ctx.model.state_dict(), "output_stride": ctx.model.output_stride,
             "input_hw": list(ctx.input_hw)}
    if extra:
        state.update(extra)
    torch.save(state, path)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the single-person detector from scratch on AIST++ bbox labels. "
                    "See docs/project_decisions.md."
    )
    p.add_argument("--train", required=True, help="configs/train/train_detector.yaml")
    p.add_argument("--model", default=None)
    p.add_argument("--max-items", type=int, default=None, help="debug: cap dataset size")
    p.add_argument("--smoke-steps", type=int, default=0)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    train_cfg = load_yaml(args.train)
    data_cfg = load_yaml(train_cfg["data_config"])
    model_cfg = load_yaml(args.model or train_cfg["model_config"])

    seed_everything(42)

    train_ds, val_ds = _build_datasets(data_cfg, train_cfg, args.max_items)
    train_loader = _make_loader(
        train_ds, int(train_cfg.get("batch_size", 32)),
        int(train_cfg.get("num_workers", 4)), shuffle=True,
    )
    val_loader = _make_loader(
        val_ds, int(train_cfg.get("batch_size", 32)),
        int(train_cfg.get("num_workers", 4)), shuffle=False,
    )

    ctx = make_detector_ctx(train_cfg, model_cfg)
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    stride = int(train_cfg.get("output_stride", 4))
    epochs = int(train_cfg.get("epochs", 1))
    best_iou = -1.0
    step_budget = args.smoke_steps if args.smoke_steps > 0 else None
    steps_used = 0

    for epoch in range(1, epochs + 1):
        if step_budget is not None and steps_used >= step_budget:
            break
        stats = train_one_epoch(ctx, train_loader, epoch)
        if step_budget is not None:
            steps_used += len(train_loader)
        print({"epoch": epoch, **stats})

        if epoch % ctx.eval_every_n_epochs == 0:
            val_stats = evaluate(ctx, val_loader, stride)
            print({"epoch": epoch, **val_stats})
            if val_stats["val_iou"] > best_iou:
                best_iou = val_stats["val_iou"]
                save_checkpoint(ctx, ctx.output_dir / "best.pt", extra={"epoch": epoch, "val": val_stats})
                print(f"saved best -> {ctx.output_dir/'best.pt'} (val_iou={best_iou:.4f})")

        if epoch % ctx.save_every_n_epochs == 0:
            save_checkpoint(ctx, ctx.output_dir / f"epoch_{epoch:04d}.pt", extra={"epoch": epoch})

    save_checkpoint(ctx, ctx.output_dir / "last.pt", extra={"epoch": epochs})
    print(f"done. weights -> {ctx.output_dir}")


if __name__ == "__main__":
    main()
