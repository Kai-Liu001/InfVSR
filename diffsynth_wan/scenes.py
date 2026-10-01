"""Online scene boundaries and shared visual references."""

from collections import deque
from dataclasses import dataclass
from itertools import islice

import numpy as np
import torch

from .video import frames_to_tensor


@dataclass
class VideoChunk:
    frames: torch.Tensor
    new_scene: bool
    update_reference: bool


def scene_batches(
    frames, max_frames=None, scene_threshold=27.0, reference_threshold=27.0
):
    from scenedetect.detectors import ContentDetector

    detector = ContentDetector(threshold=scene_threshold)
    delay = detector.event_buffer_length
    pending, cuts = deque(), deque()
    start = 0
    new_scene = True
    reference = None
    reference_index = 0

    def drain(limit, final=False):
        nonlocal start, new_scene, reference, reference_index
        while pending and start < limit:
            if cuts and cuts[0] == start:
                cuts.popleft()
                new_scene = True
            boundary = cuts[0] if cuts else None
            stop = min(limit, boundary) if boundary is not None else limit
            size = 9 if new_scene else 12
            available = stop - start
            if available < size and not (final or boundary == stop):
                break
            batch = [pending.popleft() for _ in range(min(size, available))]
            current = np.ascontiguousarray(batch[0][:, :, ::-1])
            update = new_scene
            if not update and start - reference_index >= 15:
                comparison = ContentDetector(
                    threshold=reference_threshold, min_scene_len=0
                )
                comparison.process_frame(0, reference)
                update = bool(comparison.process_frame(1, current))
            if update:
                reference, reference_index = current, start
            yield VideoChunk(frames_to_tensor(batch), new_scene, update)
            start += len(batch)
            new_scene = False

    count = 0
    for index, frame in enumerate(islice(frames, max_frames)):
        pending.append(frame)
        cuts.extend(
            detector.process_frame(index, np.ascontiguousarray(frame[:, :, ::-1]))
        )
        count = index + 1
        yield from drain(max(0, count - delay))
    if count:
        cuts.extend(detector.post_process(count - 1))
        yield from drain(count, final=True)
