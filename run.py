"""CLI entry for the integrated dance-CV pipeline.

Usage:
    python run.py --benchmark BENCHMARK.mp4 --learner LEARNER.mp4 --out results/
"""
from __future__ import annotations

import argparse
import json

from src.pipeline.run_pipeline import run


def main() -> None:
    p = argparse.ArgumentParser(description="Integrated dance-CV pipeline")
    p.add_argument("--benchmark", required=True, help="Path to the reference video")
    p.add_argument("--learner",   required=True, help="Path to the learner video")
    p.add_argument("--out",       default="results/integrate_run", help="Output directory")
    p.add_argument("--config",    default="configs/integrate/pipeline.yaml")
    p.add_argument("--device",    default=None, help="cpu / cuda / mps; default = auto")
    args = p.parse_args()

    report = run(
        benchmark_video=args.benchmark,
        learner_video=args.learner,
        output_dir=args.out,
        config_path=args.config,
        device=args.device,
    )
    print(json.dumps({"overall_score": report["overall_score"],
                       "n_intervals": len(report["intervals"]),
                       "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
