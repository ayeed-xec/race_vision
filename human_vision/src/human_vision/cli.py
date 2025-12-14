from __future__ import annotations

import argparse
import sys

import numpy as np

from .models.specs_builtin import REGISTRY
from .models.store import ModelStore
from .models.sync import sync_repo_models
from .vision import Vision, VisionConfig


def _cmd_models(args: argparse.Namespace) -> int:
    store = ModelStore()
    if args.action == "list":
        for spec in REGISTRY.list_models():
            print(f"{spec.model_id}\t{spec.capability.value}\t{spec.backend_kind.value}")
        return 0
    if args.action == "ensure":
        spec = REGISTRY.get(args.model_id)
        path = store.ensure(spec)
        print(f"Ensured {spec.model_id} at {path}")
        return 0
    if args.action == "sync-repo":
        path = sync_repo_models(args.repo, branch=args.branch)
        print(f"Synced models to {path}")
        return 0
    return 1


def _cmd_benchmark(args: argparse.Namespace) -> int:
    frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    config = VisionConfig()
    with Vision(config) as vision:
        for _ in range(args.n):
            vision.analyze(frame)
    print("Benchmark completed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("human-vision")
    sub = parser.add_subparsers(dest="command")

    models = sub.add_parser("models")
    models_sub = models.add_subparsers(dest="action", required=True)
    models_sub.add_parser("list")
    ensure_p = models_sub.add_parser("ensure")
    ensure_p.add_argument("model_id")
    sync_p = models_sub.add_parser("sync-repo")
    sync_p.add_argument("--repo", required=True)
    sync_p.add_argument("--branch", default="main")

    bench = sub.add_parser("benchmark")
    bench.add_argument("--mode", default="threaded")
    bench.add_argument("--n", type=int, default=10)
    bench.add_argument("--height", type=int, default=480)
    bench.add_argument("--width", type=int, default=640)

    demo = sub.add_parser("demo")
    demo.add_argument("--webcam", type=int, default=0)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "models":
        return _cmd_models(args)
    if args.command == "benchmark":
        return _cmd_benchmark(args)
    if args.command == "demo":
        print("Demo requires OpenCV; this stub does not implement live UI.")
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
