"""Pose training CLI.

Trains pose/keypoint models from scratch on allowed supervised keypoint
sources. See docs/project_decisions.md sections 1 and 6: no pretrained
keypoint weights; allowed labels are AIST++ plus optional human-labeled
custom_dance frames.

Usage::

    python -m src.train.train_pose --train configs/train/train.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from src.utils.seed import seed_everything


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a pose model from scratch on allowed keypoint labels. See docs/project_decisions.md.")
    p.add_argument("--train", required=True, help="configs/train/train.yaml")
    p.add_argument("--model", default=None, help="override model config path")
    p.add_argument("--max-items-per-source", type=int, default=None, help="debug: cap per-source dataset size")
    p.add_argument("--epoch-size", type=int, default=None, help="override virtual epoch size")
    p.add_argument("--smoke-steps", type=int, default=0, help="debug: run only N steps total")
    p.add_argument("--batch-size", type=int, default=None, help="debug: override training config batch size")
    p.add_argument("--num-workers", type=int, default=None, help="debug: override training config dataloader workers")
    return p


def _require_no_external_weights(model_cfg: Dict) -> None:
    if model_cfg.get("pretrained", False):
        raise SystemExit(
            "Refusing to run: model_config.pretrained is true. "
            "docs/project_decisions.md section 1 forbids pretrained weights."
        )


_ALLOWED_TRAIN_SOURCES = {"aistpp", "custom_dance"}


def _require_allowed_train_sources(train_cfg: Dict, data_cfg: Dict) -> None:
    mix = train_cfg.get("dataset_mix") or {}
    bad = [k for k, w in mix.items() if float(w) > 0 and k not in _ALLOWED_TRAIN_SOURCES]
    if bad:
        raise SystemExit(
            f"Refusing to train: only AIST++ and human-labeled custom_dance frames "
            f"are allowed as supervised sources (docs/project_decisions.md section 6). "
            f"Got extra sources with non-zero weight: {bad}."
        )
    if not any(float(w) > 0 for w in mix.values()):
        raise SystemExit(
            "Refusing to train: dataset_mix must include at least one positive allowed source."
        )
    if float(mix.get("custom_dance", 0)) > 0:
        sub = (data_cfg.get("datasets") or {}).get("custom_dance") or {}
        missing = []
        if not sub.get("enabled", True):
            missing.append("datasets.custom_dance.enabled")
        for key in ("annotations", "val_annotations"):
            path = sub.get(key)
            if not path or not Path(path).exists():
                missing.append(f"datasets.custom_dance.{key}")
        if missing:
            raise SystemExit(
                "Refusing to train with custom_dance: manual labels must exist before "
                f"setting dataset_mix.custom_dance > 0. Missing: {missing}."
            )


def main() -> None:
    from src.utils.config import load_yaml

    args = _build_parser().parse_args()
    train_cfg = load_yaml(args.train)
    if args.batch_size is not None:
        train_cfg["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        train_cfg["num_workers"] = int(args.num_workers)
    data_cfg = load_yaml(train_cfg["data_config"])
    model_cfg = load_yaml(args.model or train_cfg["model_config"])
    _require_no_external_weights(model_cfg)
    _require_allowed_train_sources(train_cfg, data_cfg)

    from src.datasets.mixed_pose_dataset import build_mixed_from_configs
    from src.train.engine import (
        evaluate,
        make_loader,
        make_train_ctx,
        save_checkpoint,
        train_one_epoch,
    )

    seed_everything(42)

    mix = train_cfg.get("dataset_mix", {})
    train_ds = build_mixed_from_configs(
        data_cfg=data_cfg,
        mix=mix,
        is_train=True,
        epoch_size=args.epoch_size,
        max_items_per_source=args.max_items_per_source,
    )
    val_ds = build_mixed_from_configs(
        data_cfg=data_cfg,
        mix=mix,
        is_train=False,
        epoch_size=args.epoch_size,
        max_items_per_source=args.max_items_per_source,
    )
    train_loader = make_loader(train_ds, int(train_cfg.get("batch_size", 32)), int(train_cfg.get("num_workers", 4)), shuffle=True)
    val_loader = make_loader(val_ds, int(train_cfg.get("batch_size", 32)), int(train_cfg.get("num_workers", 4)), shuffle=False)

    ctx = make_train_ctx(train_cfg, model_cfg)
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    best_val = -float("inf")
    epochs = int(train_cfg.get("epochs", 1))

    step_budget = args.smoke_steps if args.smoke_steps > 0 else None
    steps_used = 0

    for epoch in range(1, epochs + 1):
        if step_budget is not None and steps_used >= step_budget:
            break
        train_stats = train_one_epoch(ctx, train_loader, epoch)
        if step_budget is not None:
            steps_used += len(train_loader)
        print({"epoch": epoch, **train_stats})

        if epoch % ctx.eval_every_n_epochs == 0:
            val_stats = evaluate(ctx, val_loader)
            print({"epoch": epoch, **val_stats})
            cur = val_stats.get("val_pck01") or -val_stats.get("val_loss", float("inf"))
            if cur > best_val:
                best_val = cur
                save_checkpoint(ctx, ctx.output_dir / "best.pt", extra={"epoch": epoch, "val": val_stats})
                print(f"saved best -> {ctx.output_dir/'best.pt'} (val_pck01={cur:.4f})")

        if epoch % ctx.save_every_n_epochs == 0:
            save_checkpoint(ctx, ctx.output_dir / f"epoch_{epoch:04d}.pt", extra={"epoch": epoch})

    save_checkpoint(ctx, ctx.output_dir / "last.pt", extra={"epoch": epochs})
    print(f"done. weights -> {ctx.output_dir}")


if __name__ == "__main__":
    main()
