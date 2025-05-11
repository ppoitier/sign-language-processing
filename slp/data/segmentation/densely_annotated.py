from collections import Counter

import torch
import webdataset as wds
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, default_collate
from tqdm import tqdm
import numpy as np

from slp.codecs.segmentation.base import SegmentationCodec
from slp.utils.windows import (
    convert_segmentation_instances_to_windows,
    filter_empty_segmentation_windows,
)

torch.set_float32_matmul_precision("medium")


def _get_wds_mapping_fn(
    segment_codecs: dict[str, SegmentationCodec] | None = None,
    body_regions=("upper_pose", "left_hand", "right_hand", "lips"),
):
    def _map_wds_to_sample(wds_sample):
        segments = wds_sample["segments.npy"]
        instance = {
            "id": wds_sample["__key__"],
            "poses": {
                region: wds_sample[f"pose.{region}.npy"] for region in body_regions
            },
            "segments": segments,
        }
        length = wds_sample[f"pose.{body_regions[0]}.npy"].shape[0]
        if segment_codecs is not None:
            for key, codec in segment_codecs.items():
                instance[key] = {
                    'segments': codec.encode_segments_to_segment_targets(segments),
                    'frames': codec.encode_segments_to_frame_targets(segments, length)
                }
        return instance

    return _map_wds_to_sample


def _compute_label_occurrences(samples: list[dict], target_name: str):
    label_counter = Counter()
    for sample in samples:
        frame_labels = sample[target_name]['frames']
        if frame_labels.ndim > 1:
            frame_labels = frame_labels[:, 0]
        label_counter += Counter(frame_labels)
    occurrences = np.zeros(len(label_counter))
    for label, count in label_counter.items():
        occurrences[int(label)] = count
    return occurrences


class Collator:
    def __init__(self, segment_codec_names: list[str] | None = None):
        self.segment_codec_names = segment_codec_names

    def __call__(self, batch):
        segments = [b.pop("segments") for b in batch]
        poses = pad_sequence(
            [torch.from_numpy(b["poses"]) for b in batch], batch_first=True
        ).float()
        lengths = [b.pop("poses").shape[0] for b in batch]
        segment_codecs = {}
        if self.segment_codec_names is not None:
            segment_codecs = {
                name: {
                    'segments': [torch.from_numpy(b[name]['segments']) for b in batch],
                    'frames': pad_sequence(
                        [torch.from_numpy(b.pop(name)['frames']) for b in batch],
                        batch_first=True,
                        padding_value=-1,
                    ),
                }
                for name in self.segment_codec_names
            }
        final_batch = default_collate(batch)
        final_batch.update(segment_codecs)
        final_batch["segments"] = [
            torch.from_numpy(s).reshape(-1, 3).long() for s in segments
        ]
        final_batch["poses"] = poses
        final_batch["lengths"] = lengths
        final_batch["masks"] = pad_sequence(
            [torch.ones(length, dtype=torch.uint8) for length in lengths],
            batch_first=True,
        )
        return final_batch


class DenselyAnnotatedSLDataset(Dataset):
    def __init__(
        self,
        url: str,
        segment_codecs: dict[str, SegmentationCodec] | None = None,
        pose_transforms=None,
        verbose: bool = False,
        use_windows: bool = False,
        window_size: int = 1500,
        window_stride: int = 1200,
        max_empty_windows: int | None = None,
    ):
        super().__init__()
        self.samples: list[dict] = []
        self.pose_transforms = pose_transforms
        web_dataset = wds.DataPipeline(
            wds.SimpleShardList(url),
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.decode(),
            wds.map(_get_wds_mapping_fn(segment_codecs)),
        )
        print("Loading instances from shards:", url)
        for sample in tqdm(web_dataset, disable=not verbose, unit="samples"):
            self.samples.append(sample)
        if use_windows:
            n_instances = len(self.samples)
            self.samples = convert_segmentation_instances_to_windows(
                self.samples, window_size, window_stride
            )
            n_windows = len(self.samples)
            if verbose:
                print(f"Converted {n_instances} instances to {n_windows} windows.")
            if max_empty_windows is not None:
                self.samples = filter_empty_segmentation_windows(
                    self.samples, max_empty_windows
                )
                if verbose:
                    print(
                        f"Removed {n_windows - len(self.samples)} empty instances. There are {len(self.samples)} final windows."
                    )

    def get_label_occurrences(self, target_name: str):
        return _compute_label_occurrences(self.samples, target_name)

    def get_label_frequencies(self, target_name: str):
        occurrences = self.get_label_occurrences(target_name)
        return occurrences / occurrences.sum()

    def get_label_weights(self, target_name: str):
        return 1 / self.get_label_frequencies(target_name)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = {**self.samples[index]}
        if self.pose_transforms is not None:
            sample["poses"] = self.pose_transforms(sample["poses"])
        return sample
