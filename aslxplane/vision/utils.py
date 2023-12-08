from __future__ import annotations

import sys
from io import BytesIO
from copy import copy
from PIL import Image
import os
from pathlib import Path
from contextlib import contextmanager
from multiprocessing.pool import ThreadPool
from subprocess import Popen, PIPE

import cv2
import torch
from torch import Tensor
import zstandard as zstd
from tqdm import tqdm


def frame_to_jpeg_buffer(frame: Tensor) -> bytes:
    buffer = BytesIO()
    Image.fromarray(frame.numpy()).save(buffer, format="jpeg")
    buffer.seek(0)
    data = buffer.read()
    buffer.close()
    return data


def frame2bytes(frame: Tensor) -> bytes:
    buffer = BytesIO()
    torch.save(frame, buffer)
    buffer.seek(0)
    data = buffer.read()
    buffer.close()
    return data


def bytes2frame(data: bytes) -> Tensor:
    buffer = BytesIO(data)
    frame = torch.load(buffer)
    buffer.close()
    return frame


####################################################################################################


def frame2compressed_bytes(frame: Tensor) -> bytes:
    frame = frame.clone()
    with BytesIO() as buffer:
        torch.save(frame, buffer)
        buffer.seek(0)
        uncompressed = buffer.read()
    compressed = zstd.compress(copy(uncompressed))
    return compressed


def compressed_bytes2frame(buffer: bytes) -> Tensor:
    with BytesIO(zstd.decompress(buffer)) as buffer:
        return torch.load(buffer)


####################################################################################################


@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, "w") as fnull:
        oldout, olderr = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = fnull, fnull
            yield
        finally:
            sys.stdout, sys.stderr = oldout, olderr


def find_weather_file(video_file: Path) -> Path:
    name = video_file.with_suffix("").name
    prefix, id = name.split("_")
    return video_file.parent / f"{prefix}_weather_{id}.json"


def find_json_file(video_file: Path) -> Path:
    return video_file.parent / video_file.with_suffix(".json").name


def get_total_frame_length(video_file: Path) -> int:
    cap = cv2.VideoCapture(str(video_file))
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


####################################################################################################


def find_index_bisection(cum_lengths: list[int], n: int) -> int:
    left, right = 0, len(cum_lengths) - 1
    if n >= cum_lengths[right]:
        return right
    while right - left > 1:
        mid = (left + right) // 2
        if n >= cum_lengths[mid]:
            left = mid
        else:
            right = mid
    return left


def _process_video(fname: Path | str) -> int:
    fname = Path(fname).absolute()
    assert fname.suffix != ".mp4"
    if fname.with_suffix(".mp4").exists():
        return 0 # success
    p = Popen(
        ["ffmpeg", "-i", str(fname), "-c:v", "mjpeg", "-q:v", "2", str(fname.with_suffix(".mp4"))],
        stdout=PIPE,
        stderr=PIPE,
    )
    p.wait()
    return p.returncode


def process_videos(files: list[str | Path]) -> None:
    with ThreadPool(8) as pool:
        for _ in tqdm(pool.imap_unordered(_process_video, files), total=len(files)):
            pass
    return
