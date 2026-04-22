"""Evaluate a trained pose checkpoint on a given validation split.

Usage::

    python -m src.train.eval_pose \
        --train configs/train/stage2_dance.yaml \
        --ckpt data/processed/stage_B/best.pt
"""
from __future__ import annotations

import argparse

import torch

from src.datasets.mixed_pose_dataset import build_mixed_from_configs
from src.train.engine import evaluate, make_loader, make_train_ctx
from src.utils.config import load_yaml


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a pose checkpoint.")
    p.add_argument("--train", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    train_cfg = load_yaml(args.train)
    data_cfg = load_yaml(train_cfg["data_config"])
    model_cfg = load_yaml(train_cfg["model_config"])

    val_ds = build_mixed_from_configs(data_cfg=data_cfg, mix=train_cfg.get("dataset_mix", {}), is_train=False)
    val_loader = make_loader(val_ds, args.batch_size, args.num_workers, shuffle=False)

    # Point init_from at the requested ckpt (must live inside data/processed/).
    train_cfg_local = dict(train_cfg)
    train_cfg_local["init_from"] = args.ckpt
    ctx = make_train_ctx(train_cfg_local, model_cfg)
    stats = evaluate(ctx, val_loader)
    print(stats)


if __name__ == "__main__":
    main()
