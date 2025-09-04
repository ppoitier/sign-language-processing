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
            poses = self._pad_poses(batch).permute(0, 2, 1).contiguous()
        else:
            poses = {
                k: self._pad_poses(batch, lambda s: s["poses"][k]).contiguous()
                for k in self.body_regions
            }
            [b.pop("poses") for b in batch]
        final_batch = default_collate(batch)
        final_batch["poses"] = poses
        final_batch["masks"] = (
            self._create_masks(final_batch["length"].tolist()).unsqueeze(1).contiguous()
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
        # targets = [{k: v for k, v in b.pop('targets').items()} for b in batch]
        # codec_data_per_name = {
        #     name: [b.pop(name) for b in batch] for name in self.segment_codec_names
        # }
        # segment_codecs = {
        #     name: {
        #         key: (
        #             # Pad sequences for any key that is NOT 'segments'.
        #             pad_sequence(
        #                 [
        #                     torch.from_numpy(d[key]).to(self.annots_dtype)
        #                     for d in codec_batch
        #                 ],
        #                 batch_first=True,
        #                 padding_value=-1,
        #             )
        #             if key != "segments"
        #             else
        #             # For 'segments', just convert to a list of tensors.
        #             [
        #                 torch.from_numpy(d[key]).to(self.annots_dtype)
        #                 for d in codec_batch
        #             ]
        #         )
        #         # Iterate over the keys of the *first item* for this codec.
        #         # This assumes all items in a batch share the same annotation structure.
        #         for key in codec_batch[0].keys()
        #     }
        #     for name, codec_batch in codec_data_per_name.items()
        # }
        # segment_codecs = {
        #     name: {
        #         "frames": self._pad_per_frame_annots(batch, name),
        #         "segments": [torch.from_numpy(b.pop(name)["segments"]).to(self.annots_dtype) for b in batch],
        #     }
        #     for name in self.segment_codec_names
        # }
        batch = super().__call__(batch)
        batch['segments'] = segments
        batch['targets'] = targets
        return batch
