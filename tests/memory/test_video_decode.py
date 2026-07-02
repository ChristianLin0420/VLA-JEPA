import unittest
from unittest import mock

import av
import numpy as np

from starVLA.dataloader.gr00t_lerobot.video import (
    VideoDecodingError,
    get_frames_by_timestamps,
)


class _BrokenVideoReader:
    def __init__(self, error=None):
        self._c = object()
        self._error = error

    def seek(self, _timestamp, keyframes_only=True):
        del keyframes_only

    def __iter__(self):
        return self

    def __next__(self):
        if self._error is not None:
            raise self._error
        raise StopIteration


class TorchvisionAvDecodeTest(unittest.TestCase):
    def test_ffmpeg_error_is_wrapped_with_video_path(self):
        decode_error = av.error.InvalidDataError(
            1094995529, "Invalid data found when processing input"
        )
        reader = _BrokenVideoReader(error=decode_error)
        with mock.patch(
            "starVLA.dataloader.gr00t_lerobot.video.torchvision.io.VideoReader",
            return_value=reader,
        ), mock.patch(
            "starVLA.dataloader.gr00t_lerobot.video.torchvision.set_video_backend"
        ):
            with self.assertRaisesRegex(
                VideoDecodingError, "broken.mp4.*Invalid data"
            ):
                get_frames_by_timestamps(
                    "broken.mp4",
                    np.asarray([0.5]),
                    video_backend="torchvision_av",
                )

        self.assertIsNone(reader._c)

    def test_missing_frame_is_rejected_before_shape_can_change(self):
        reader = _BrokenVideoReader()
        with mock.patch(
            "starVLA.dataloader.gr00t_lerobot.video.torchvision.io.VideoReader",
            return_value=reader,
        ), mock.patch(
            "starVLA.dataloader.gr00t_lerobot.video.torchvision.set_video_backend"
        ):
            with self.assertRaisesRegex(
                VideoDecodingError, "requested 2 frame.*got 0"
            ):
                get_frames_by_timestamps(
                    "short.mp4",
                    np.asarray([0.5, 1.0]),
                    video_backend="torchvision_av",
                )


if __name__ == "__main__":
    unittest.main()
