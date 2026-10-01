"""Run InfVSR on a video or a directory of videos."""

import argparse
from pathlib import Path

from diffsynth_wan.video import VIDEO_EXTENSIONS, video_source, write_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Video file or directory"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs"), help="Output directory"
    )
    parser.add_argument("--model_dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "diffsynth_wan" / "config.json",
    )
    parser.add_argument("--scale", type=int, default=4, help="Spatial upscaling factor")
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional input frame limit; default processes the entire video",
    )
    parser.add_argument(
        "--fps", type=float, default=0, help="Output FPS; 0 follows input"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no_scene_detection",
        action="store_true",
        help="Keep one reference and continuous caches for the entire video",
    )
    parser.add_argument(
        "--scene_threshold",
        type=float,
        default=27.0,
        help="Hard-cut content threshold (default: 27)",
    )
    parser.add_argument(
        "--reference_threshold",
        type=float,
        default=27.0,
        help="Reference-frame content threshold (default: 27)",
    )
    parser.add_argument(
        "--yuv444p",
        action="store_true",
        help="Save lossless H.264 yuv444p instead of default yuv420p",
    )
    args = parser.parse_args()
    if (
        args.scale < 1
        or (args.max_frames is not None and args.max_frames < 1)
        or args.fps < 0
        or args.scene_threshold <= 0
        or args.reference_threshold <= 0
    ):
        parser.error(
            "scale, max_frames and thresholds must be positive; fps must be nonnegative"
        )
    if args.input.is_file() and args.input.suffix.lower() in VIDEO_EXTENSIONS:
        videos = [args.input]
    elif args.input.is_dir():
        videos = sorted(
            p
            for p in args.input.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
    else:
        parser.error(f"Invalid video input: {args.input}")
    if not videos:
        parser.error("No supported videos found")
    destinations = [args.output / f"{p.stem}.mp4" for p in videos]
    if len(set(destinations)) != len(destinations):
        parser.error(
            "Input videos have duplicate stems; use separate output directories"
        )
    if any(path.exists() for path in destinations):
        parser.error("An output file already exists; choose a new output directory")
    names = ("InfVSR.ckpt", "Wan2.1_VAE.pth", "ram_swin_large_14m.pth", "DAPE.pth")
    paths = [args.model_dir / name for name in names]
    for path in [args.config, *paths]:
        if not path.is_file():
            parser.error(f"Missing file: {path}")
    from diffsynth_wan.pipeline import InfVSR

    model = InfVSR(paths[0], args.config, *paths[1:], device=args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    for source, destination in zip(videos, destinations):
        with video_source(
            source,
            args.max_frames,
            not args.no_scene_detection,
            args.scene_threshold,
            args.reference_threshold,
        ) as (chunks, fps):
            count = write_video(
                model(chunks, scale=args.scale),
                destination,
                args.fps or fps,
                yuv444p=args.yuv444p,
            )
        print(f"Saved {destination} ({count} frames)")


if __name__ == "__main__":
    main()
