from __future__ import annotations

import argparse
from pathlib import Path

from airscreen.app import AirScreenApp
from airscreen.config import AirScreenConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="airscreen",
        description="Run the experimental AirScreen gesture control runtime.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Initialize the app without opening the camera or controlling the pointer.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index passed to the frame source.",
    )
    parser.add_argument(
        "--debug-preview",
        action="store_true",
        help="Open a camera preview with fingertip overlays and pinch gesture state.",
    )
    parser.add_argument(
        "--enable-gaze",
        action="store_true",
        help="Add approximate gaze tracking to the debug preview.",
    )
    parser.add_argument(
        "--record-landmarks",
        type=Path,
        metavar="PATH",
        help="Write debug-preview hand and gaze landmarks to a JSON Lines file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AirScreenConfig(
        camera_index=args.camera_index,
        dry_run=args.dry_run,
        debug_preview=args.debug_preview,
        gaze_enabled=args.enable_gaze,
        landmark_record_path=args.record_landmarks,
    )
    return AirScreenApp(config).run()
