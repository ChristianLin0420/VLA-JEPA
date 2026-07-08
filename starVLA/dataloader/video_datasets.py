import os
import random
import torch
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

from transformers import VJEPA2VideoProcessor

def random_crop_or_pad(video, target_h, target_w, pad_value=0):
    """
    video: np.ndarray [T, H, W, 3]
    return: np.ndarray [T, target_h, target_w, 3]
    """
    T, H, W, C = video.shape
    assert C == 3

    # 1️⃣ 随机 crop 起点（如果原图更大）
    top = random.randint(0, H - target_h) if H > target_h else 0
    left = random.randint(0, W - target_w) if W > target_w else 0

    cropped = video[
        :,
        top : top + min(H, target_h),
        left : left + min(W, target_w),
        :
    ]

    # 2️⃣ padding（如果原图更小）
    out = np.full(
        (T, target_h, target_w, 3),
        pad_value,
        dtype=video.dtype
    )

    h, w = cropped.shape[1:3]
    out[:, :h, :w, :] = cropped

    return out

def resize_video(video, target_h, target_w):
    """
    video: np.ndarray [T, H, W, 3]
    return: np.ndarray [T, target_h, target_w, 3]
    """
    T, H, W, C = video.shape
    assert C == 3

    out = np.empty((T, target_h, target_w, 3), dtype=video.dtype)

    for t in range(T):
        out[t] = cv2.resize(
            video[t],
            (target_w, target_h),  # 注意：cv2 是 (W, H)
            interpolation=cv2.INTER_AREA  # 下采样最稳
        )

    return out

def collate_fn(batch, n_views=2, resolution_size=224):
    examples = []
    for b in batch:
        video, instruction = b[0], b[1]
        example = {}
        example["image"] = [Image.fromarray(video[0]).resize((resolution_size, resolution_size))]
        example["video"] = np.stack([video, video.copy()], axis=0)  # [n_views, T, H, W, C]
        example["lang"] = instruction
        examples.append(example)

        #print(video.shape, video_batch["video"][0].shape)
        #print(video_batch["image"][0])
        #print(video_batch["lang"][0])
        #exit()
    return examples

class VideoFolderDataset(Dataset):
    def __init__(
        self,
        video_dir: str,
        text_file: str,
        n_frames: int,
        extensions=(".mp4", ".avi", ".webm"),
        crop_h_size=420,
        crop_w_size=240,
        max_retry: int = 10,
        expected_video_count: int | None = None,
        expected_label_count: int | None = None,
        frame_stride: int = 1,
    ):
        self.video_dir = video_dir
        self.n_frames = n_frames
        self.max_retry = max_retry
        self.crop_h_size = crop_h_size
        self.crop_w_size = crop_w_size
        self.frame_stride = max(1, int(frame_stride))

        # 只扫描文件名
        self.video_files = sorted([
            f for f in os.listdir(video_dir)
            if f.lower().endswith(extensions)
        ])
        if expected_video_count is not None and len(self.video_files) != expected_video_count:
            raise ValueError(
                f"video count mismatch for {video_dir}: found {len(self.video_files)}, "
                f"expected {expected_video_count}"
            )
        # SSV2 label files are headerless ``id;text`` records.  Letting pandas
        # infer a header silently drops the first labeled example.
        labels = pd.read_csv(
            text_file,
            sep=";",
            header=None,
            names=["id", "text"],
            dtype={"id": "int64", "text": "string"},
        )
        if labels[["id", "text"]].isnull().any().any():
            raise ValueError(f"SSV2 labels contain missing id/text values: {text_file}")
        if labels["id"].duplicated().any():
            duplicate_ids = labels.loc[labels["id"].duplicated(), "id"].head(5).tolist()
            raise ValueError(f"SSV2 labels contain duplicate ids {duplicate_ids}: {text_file}")

        self.label_count = len(labels)
        if expected_label_count is not None and self.label_count != expected_label_count:
            raise ValueError(
                f"SSV2 label count mismatch for {text_file}: "
                f"found {self.label_count}, expected {expected_label_count}"
            )
        self.id2text = dict(zip(labels["id"], labels["text"]))
        
        for each in self.video_files:
            file_idx = int(each.split(".")[0])
            if file_idx not in self.id2text:
                self.id2text[file_idx] = "Completing something that humans might want to do."

        if len(self.video_files) == 0:
            raise RuntimeError(f"No video files found in {video_dir}")

    def __len__(self):
        return len(self.video_files)
    
    def _load_video(self, idx):
        file_idx = int(self.video_files[idx].split(".")[0])
        video_path = os.path.join(self.video_dir, self.video_files[idx])

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("无法打开视频")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        span = (self.n_frames - 1) * self.frame_stride + 1
        if frame_count < span:
            raise ValueError(f"Video {video_path} has only {frame_count} frames, which is less than the required span of {span} frames.")

        start = random.randint(0, frame_count - span)

        # 3️⃣ 连续、递增、合法的 frame_ids
        frame_ids = np.arange(start, start + span, self.frame_stride, dtype=np.int64)

        frames = []
        for idx in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Unable to read frame at index {idx}")
            frames.append(frame)
        cap.release()
        #frames = random_crop_or_pad(
        #    np.array(frames),
        #    target_h=self.crop_h_size,
        #    target_w=self.crop_w_size,
        #    pad_value=0)
        frames = resize_video(
            np.array(frames),
            target_h=self.crop_h_size,
            target_w=self.crop_w_size)

        #print(frames.shape, video_path, file_idx, file_idx in self.id2text.keys())

        return [frames, self.id2text[file_idx]]

    def __getitem__(self, idx):
        for _ in range(self.max_retry):
            try:
                return self._load_video(idx)
            except Exception as e:
                idx = random.randint(0, len(self.video_files) - 1)

        return self._load_video(2)
