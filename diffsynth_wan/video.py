"""Incremental video input and output."""

from contextlib import contextmanager
from itertools import chain, islice
import math

import imageio.v2 as imageio
import numpy as np
import torch
from torch.nn import functional as F

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".mpg", ".mpeg", ".m4v"}


def frames_to_tensor(frames):
    pixels = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).unsqueeze(0)
    return pixels.float().div_(255).mul_(2).sub_(1)


def frame_batches(frames, max_frames=None):
    """Group the first 9 and subsequent 12 pixel frames into DiT-sized chunks."""
    frames = iter(frames)
    consumed = 0
    batch_size = 9
    while max_frames is None or consumed < max_frames:
        count = (
            batch_size if max_frames is None else min(batch_size, max_frames - consumed)
        )
        batch = list(islice(frames, count))
        if not batch:
            return
        yield frames_to_tensor(batch)
        consumed += len(batch)
        batch_size = 12


@contextmanager
def video_source(
    path,
    max_frames=None,
    detect_scenes=True,
    scene_threshold=27.0,
    reference_threshold=27.0,
):
    reader = imageio.get_reader(str(path))
    try:
        metadata = reader.get_meta_data()
        fps = metadata.get("fps")
        if not fps:
            duration, count = metadata.get("duration"), metadata.get("nframes")
            fps = count / duration if count and duration and duration > 0 else 15.0
        fps = float(fps)
        if not math.isfinite(fps) or fps <= 0:
            fps = 15.0
        if detect_scenes:
            from .scenes import scene_batches

            chunks = scene_batches(
                reader, max_frames, scene_threshold, reference_threshold
            )
        else:
            chunks = frame_batches(reader, max_frames)
        yield chunks, fps
    finally:
        reader.close()


def resize_video(video, scale):
    b, c, t, h, w = video.shape
    frames = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    frames = F.interpolate(
        frames, scale_factor=scale, mode="bicubic", align_corners=False, antialias=True
    )
    return frames.reshape(b, t, c, *frames.shape[-2:]).permute(0, 2, 1, 3, 4)


def pad_chunk(video, first):
    length, height, width = video.shape[-3:]
    padding = (0, -width % 16, 0, -height % 16, 0, (int(first) - length) % 4)
    if any(padding):
        video = F.pad(video, padding, mode="replicate")
    return video, (length, height, width)


def write_video(chunks, path, fps, yuv444p=False):
    chunks = iter(chunks)
    first = next(chunks, None)
    if first is None:
        raise ValueError("No video frames to save")
    height, width = first.shape[-2:]
    pixel_format = "yuv444p" if height % 2 or width % 2 else "yuv420p"
    options = dict(
        fps=fps,
        quality=7,
        codec="libx264",
        pixelformat=pixel_format,
        macro_block_size=None,
    )
    if yuv444p:
        options = dict(
            fps=fps,
            codec="libx264",
            pixelformat="yuv444p",
            macro_block_size=None,
            ffmpeg_params=["-crf", "0"],
        )
    count = 0
    with imageio.get_writer(str(path), **options) as writer:
        for chunk in chain((first,), chunks):
            for frame in chunk[0].unbind(dim=1):
                image = ((frame.float() + 1) * 127.5).clamp_(0, 255)
                writer.append_data(
                    image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                )
                count += 1
    return count
