from __future__ import annotations

import sys
from pathlib import Path
from itertools import accumulate


import cv2
import torch
from torchvision import transforms as T
import json

from .utils import find_weather_file, find_json_file, get_total_frame_length


def read_nth_frame(video_file: Path | str, n: int):
    cap = cv2.VideoCapture(str(video_file.absolute()))
    if not cap.isOpened():
        raise Exception("Error opening video stream or file")
    assert n < cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ret, frame = cap.read()
    assert ret
    return torch.from_numpy(frame)


class RandomAccessXPlaneVideoDataset:
    def __init__(
        self,
        files: list[Path | str],
        transform: None | str | "Transform" = None,
        skip_start_frames: int = 60,
        skip_end_frames: int = 60,
        frame_skip_n: int = 10,
        output_full_data: bool = False,
    ):
        self.skip_start_frames, self.skip_end_frames = skip_start_frames, skip_end_frames
        self.frame_skip_n = frame_skip_n

        self.video_files = [Path(data_file).absolute() for data_file in files]
        idxs = list(range(len(self.video_files)))
        self.video_files = [self.video_files[idx] for idx in idxs]
        self.weather_files = [find_weather_file(video_file) for video_file in self.video_files]
        self.data_files = [find_json_file(video_file) for video_file in self.video_files]

        # take only the files for which all data exists
        mask = [
            weather_file.exists() and data_file.exists()
            for weather_file, data_file in zip(self.weather_files, self.data_files)
        ]
        self.video_files = [video_file for video_file, m in zip(self.video_files, mask) if m]
        self.data_files = [data_file for data_file, m in zip(self.data_files, mask) if m]
        self.weather_files = [
            weather_file for weather_file, m in zip(self.weather_files, mask) if m
        ]

        actual_lengths = [get_total_frame_length(video_file) for video_file in self.video_files]
        self.lengths = [
            (length - skip_start_frames - skip_end_frames) // frame_skip_n
            for length in actual_lengths
        ]
        self.cumlengths = [0] + list(accumulate(self.lengths))
        self.total_length = sum(self.lengths)
        if transform == "default":
            self.transform = T.Compose(
                [T.Lambda(lambda x: x.transpose(-3, -1).to(torch.float32)), T.Resize((224, 224))]
            )
        else:
            self.transform = transform
        self.output_full_data = output_full_data
        self.permutation = sum(
            [
                [
                    (i, min(skip_start_frames + frame_skip_n * j, actual_lengths[i] - 1))
                    for j in range(length)
                ]
                for (i, length) in enumerate(self.lengths)
            ],
            [],
        )

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        file_idx, frame_idx = self.permutation[index]
        frame = read_nth_frame(self.video_files[file_idx], frame_idx)
        if self.transform is not None:
            frame = self.transform(frame)
        data = json.loads(self.data_files[file_idx].read_text())[frame_idx]
        if self.output_full_data:
            weather = json.loads(self.weather_files[file_idx].read_text())
            return frame, data, weather
        else:
            return frame, data["state"]
