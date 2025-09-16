import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import default_collate


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
