from typing import Optional, TypedDict

import numpy as np

import torch
from torch import Tensor
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import default_collate


class SignLanguageDataCollator:
    def __init__(
        self,
        includes_poses: bool = True,
        includes_videos: bool = False,
        pad_poses: bool = True,
        pad_videos: bool = False,
        min_length: Optional[int] = None,
        pose_dtype=torch.float32,
        video_dtype=torch.float16,
    ):
        self.includes_poses = includes_poses
        self.includes_videos = includes_videos
        self.pad_poses = pad_poses
        self.pad_videos = pad_videos
        self.min_length = min_length
        self.pose_dtype = pose_dtype
        self.video_dtype = video_dtype

    def _pad_and_process(self, items: list[Tensor], dtype: torch.dtype) -> Tensor:
        """
        Generic helper to pad, type-cast, and optionally permute time-series data.
        """
        # 1. Pad the sequence
        # batch_first=True results in (Batch, Time, Feat)
        if self.pad_poses:
            padded = pad_sequence(items, batch_first=True, padding_value=0.0)
        else:
            padded = torch.stack(items)

        if self.min_length is not None and padded.shape[1] < self.min_length:
            pad_amt = self.min_length - padded.shape[1]
            # F.pad arguments are (last_dim_left, last_dim_right, 2nd_last_left, 2nd_last_right, ...)
            # We want to pad the 2nd dimension (Time).
            # For a (B, T, C) tensor, we need to skip C (0, 0) and pad T (0, pad_amt).
            # Note: Logic changes slightly if input is 4D (Video), but usually Time is dim 1.
            padded = pad(padded, (0, 0, 0, pad_amt), "constant", 0)

        # 3. Cast Dtype
        padded = padded.to(dtype)
        return padded.contiguous()

    def __call__(self, batch: list[dict]) -> dict:
        poses = (
            [torch.as_tensor(b.pop("poses")) for b in batch]
            if self.includes_poses
            else None
        )
        videos = (
            [torch.as_tensor(b.pop("video")) for b in batch]
            if self.includes_videos
            else None
        )
        if poses is not None:
            lengths = [p.shape[0] for p in poses]
        elif videos is not None:
            lengths = [v.shape[0] for v in videos]
        else:
            # Fallback if no time-series data (rare)
            lengths = [0] * len(batch)

        # --- Lengths & Masks ---
        # Convert lengths to tensor
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)

        # Create Mask (B, T)
        # We find the max length of the current batch (or min_length if specified/larger)
        max_len = lengths_tensor.max().item()
        target_len = max(max_len, self.min_length) if self.min_length else max_len

        # Fast mask creation without loop
        # Creates a range [0, 1, ..., target_len-1] and compares with lengths
        mask = (
            torch.arange(target_len, device=lengths_tensor.device)[None, :]
            < lengths_tensor[:, None]
        )

        collated_batch = {}
        if poses is not None:
            # pose tensor of shape (B, T, ...)
            collated_batch["poses"] = self._pad_and_process(poses, dtype=self.pose_dtype)

        if videos is not None:
            # video tensor of shape (B, T, ...).
            collated_batch["video"] = self._pad_and_process(videos, dtype=self.video_dtype)

        # 3. Collate the rest (Labels, IDs, etc.)
        final_batch = default_collate(batch)

        # 4. Merge
        final_batch.update(collated_batch)
        final_batch["length"] = lengths_tensor
        final_batch["masks"] = mask  # (B, T) shape, easier for Transformers

        return final_batch


class PoseDataCollator:
    def __init__(
        self,
        body_regions: tuple[str, ...] = (
            "upper_pose",
            "left_hand",
            "right_hand",
            # "lips",
        ),
        flatten_poses: bool = False,
        data_dtype: torch.dtype = torch.float,
    ):
        self.body_regions = body_regions
        self.flatten_poses = flatten_poses
        self.data_dtype = data_dtype

    def _pad_poses(self, batch, sample_to_pose: callable = lambda s: s.pop("poses")):
        return pad_sequence(
            [torch.from_numpy(sample_to_pose(b)) for b in batch],
            batch_first=True,
            padding_value=0.0,
        ).to(self.data_dtype)

    def _create_masks(self, lengths: list[int]) -> torch.Tensor:
        masks = [torch.ones(length, dtype=torch.bool) for length in lengths]
        return pad_sequence(masks, batch_first=True)

    def __call__(self, batch: list[dict]) -> dict:
        if self.flatten_poses:
            lengths = [b['poses'].shape[0] for b in batch]
            poses = self._pad_poses(batch).permute(0, 2, 1).contiguous()
        else:
            poses = {
                k: self._pad_poses(batch, lambda s: s["poses"][k]).contiguous()
                for k in self.body_regions
            }
            lengths = [b.pop("poses")[self.body_regions[0]].shape[0] for b in batch]
        final_batch = default_collate(batch)
        final_batch["poses"] = poses
        final_batch["length"] = torch.tensor(lengths, dtype=torch.long)
        final_batch["masks"] = (
            self._create_masks(lengths).unsqueeze(1).contiguous()
        )
        return final_batch


class DenselyAnnotatedPoseDataCollator(PoseDataCollator):
    def __init__(
        self,
        segment_codec_names: list[str] | None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.segment_codec_names = segment_codec_names

    def _pad_target(self, batch, target_name: str):
        target = pad_sequence(
            [torch.from_numpy(b["targets"][target_name]) for b in batch],
            batch_first=True,
            padding_value=-1,
        )
        if target.ndim > 2:
            target = target.permute(0, 2, 1).contiguous()
        return target

    def __call__(self, batch):
        segments = [torch.from_numpy(b.pop("segments")).long() for b in batch]
        targets = {
            name: (
                [torch.from_numpy(b["targets"][name]).long() for b in batch]
                if name == 'segments'
                else self._pad_target(batch, name)
            )
            for name in batch[0]["targets"].keys()
        }
        [b.pop('targets') for b in batch]
        batch = super().__call__(batch)
        batch['segments'] = segments
        batch['targets'] = targets
        return batch
