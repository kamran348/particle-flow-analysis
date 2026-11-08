"""Command-line entry point: run from the project root using `python main.py`."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from particleflow.config import load_config
from particleflow.pipeline import run_pipeline
from particleflow.synthetic import create_demo_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Particle flow analysis")
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("demo", help="Generate a synthetic particle video")
    demo.add_argument("--output", default="data/raw/particles.avi")

    analyze = commands.add_parser("analyze", help="Run PIV video analysis")
    analyze.add_argument("--config", default="configs/default.toml")
    analyze.add_argument("--output", default="results/run1")
    analyze.add_argument("--track", action="store_true", help="Enable optional Trackpy tracking")

    args = parser.parse_args()
    if args.command == "demo":
        created = create_demo_video(Path(args.output))
        print(f"Created synthetic video: {created}")
    elif args.command == "analyze":
        config = load_config(args.config)
        if args.track:
            config = replace(config, tracking=replace(config.tracking, enabled=True))
        summary = run_pipeline(config, args.output)
        print(json.dumps(summary, indent=2))
        print(f"Saved outputs in: {args.output}")


if __name__ == "__main__":
    main()
