"""Command-line entry point: run from the project root using `python main.py`."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from particleflow.config import load_config
from particleflow.pipeline import export_tracking_flow, run_pipeline, run_tracking


def main() -> None:
    parser = argparse.ArgumentParser(description="Particle flow analysis")
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("demo", help="Generate a synthetic particle video")
    demo.add_argument("--output", default="data/raw/particles.avi")

    analyze = commands.add_parser("analyze", help="Analyze flow with PIV or consistent particle tracks")
    analyze.add_argument("--config", default="configs/default.toml")
    analyze.add_argument("--output", default="results/run1")
    analyze.add_argument("--track", action="store_true", help="Enable optional Trackpy tracking")

    track = commands.add_parser("track", help="Save particle IDs and trails over the video (PTV only)")
    track.add_argument("--config", default="configs/default.toml")
    track.add_argument("--output", default="results/tracking")
    track.add_argument("--step", type=int, default=1,
                       help="Track every Nth frame; defaults to 1, overriding config video.step")
    track.add_argument("--stop-frame", type=int, default=None,
                       help="Exclusive stop frame; 0 means the end (default: use config)")
    track.add_argument("--trail-length", type=int, default=20,
                       help="Trail duration in selected frames (default: 20)")
    track.add_argument("--flow", action="store_true",
                       help="Also save vector fields, speed maps and streamlines from these tracks")

    flow = commands.add_parser("flow", help="Create flow plots from an existing tracking run")
    flow.add_argument("--tracking-dir", required=True,
                      help="Folder containing trajectories.csv and tracking_summary.json")
    flow.add_argument("--output", default="results/tracking-flow")

    args = parser.parse_args()
    if args.command == "analyze":
        config = load_config(args.config)
        if args.track:
            config = replace(config, tracking=replace(config.tracking, enabled=True))
        summary = run_pipeline(config, args.output)
        print(json.dumps(summary, indent=2))
        print(f"Saved outputs in: {args.output}")
    elif args.command == "track":
        config = load_config(args.config)
        video = replace(config.video, step=args.step)
        if args.stop_frame is not None:
            video = replace(video, stop_frame=args.stop_frame)
        config = replace(config, video=video, tracking=replace(config.tracking, enabled=True))
        print("Detecting and linking particles, then rendering the tracking replay...")
        summary = run_tracking(config, args.output, trail_length=args.trail_length,
                               export_flow=args.flow)
        print(json.dumps(summary, indent=2))
        print(f"Open the replay: {Path(args.output) / 'tracking.avi'}")
        if args.flow:
            print(f"Saved flow plots and velocity fields in: {args.output}")
    elif args.command == "flow":
        summary = export_tracking_flow(args.tracking_dir, args.output)
        print(json.dumps(summary, indent=2))
        print(f"Saved flow plots and velocity fields in: {args.output}")


if __name__ == "__main__":
    main()
