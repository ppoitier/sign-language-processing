from collections import Counter

import numpy as np
import webdataset as wds
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from slp.codecs.annotations.base import AnnotationCodec
from slp.codecs.annotations.loading import load_annotation_codec
from slp.config.templates.data import SegmentationDatasetConfig
from slp.data.dataloader import DenselyAnnotatedPoseDataCollator
from slp.transforms.pose_pipelines import get_pose_pipeline
from slp.utils.windows import (
    convert_segmentation_instances_to_windows,
    filter_empty_segmentation_windows,
)


def _get_wds_mapping_fn(
    annot_codecs: dict[str, AnnotationCodec] | None = None,
    body_regions=("upper_pose", "left_hand", "right_hand", "lips"),
):
    def _map_wds_to_sample(wds_sample):
        segments = wds_sample["segments.npy"]
        length = wds_sample[f"pose.{body_regions[0]}.npy"].shape[0]
        instance = {
            "id": wds_sample["__key__"],
            "poses": {
                region: wds_sample[f"pose.{region}.npy"] for region in body_regions
            },
            "segments": segments,
            "targets": {},
            "length": length,
        }
        if annot_codecs is not None:
            for key, codec in annot_codecs.items():
                instance["targets"] = {
                    **instance["targets"],
                    **codec.encode(segments, n_frames=length),
                }
        return instance

    return _map_wds_to_sample


def _compute_label_occurrences(samples: list[dict]):
    label_counter = Counter()
    for sample in samples:
        frame_labels = sample['targets']["frame_labels"]
        if frame_labels.ndim > 1:
            frame_labels = frame_labels[:, 0]
        label_counter += Counter(frame_labels)
    occurrences = np.zeros(len(label_counter))
    for label, count in label_counter.items():
        occurrences[int(label)] = count
    return occurrences


class DenselyAnnotatedSLDataset(Dataset):
    def __init__(
        self,
        url: str,
        annot_codecs: dict[str, AnnotationCodec] | None = None,
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
            wds.map(_get_wds_mapping_fn(annot_codecs)),
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
                print(f"Converted {n_instances} full sequences to {n_windows} windows.")
            if max_empty_windows is not None:
                self.samples = filter_empty_segmentation_windows(
                    self.samples, max_empty_windows
                )
                if verbose:
                    print(
                        f"Removed {n_windows - len(self.samples)} empty sequences. There are {len(self.samples)} final windows."
                    )

    def get_label_occurrences(self):
        return _compute_label_occurrences(self.samples)

    def get_label_frequencies(self):
        occurrences = self.get_label_occurrences()
        return occurrences / occurrences.sum()

    def get_label_weights(self):
        return 1 / self.get_label_frequencies()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = {**self.samples[index]}
        if self.pose_transforms is not None:
            sample["poses"] = self.pose_transforms(sample["poses"])
        return sample


def load_dataset(
    config: SegmentationDatasetConfig,
) -> tuple[DenselyAnnotatedSLDataset, DataLoader | None]:
    use_windows, window_size, window_stride, max_empty_windows = False, 1500, 1200, None
    pose_transforms = None
    annot_codecs = {}
    if config.preprocessing is not None:
        use_windows = config.preprocessing.use_windows
        window_size = config.preprocessing.window_size
        window_stride = config.preprocessing.window_stride
        max_empty_windows = config.preprocessing.max_empty_windows
        pose_transforms = get_pose_pipeline(
            config.preprocessing.pose_transforms_pipeline
        )
        annot_codecs = {
            codec_config.name: load_annotation_codec(codec_config)
            for codec_config in config.preprocessing.segment_codecs
        }
    dataset = DenselyAnnotatedSLDataset(
        url=config.shards_url,
        pose_transforms=pose_transforms,
        verbose=config.verbose,
        annot_codecs=annot_codecs,
        use_windows=use_windows,
        window_size=window_size,
        window_stride=window_stride,
        max_empty_windows=max_empty_windows,
    )
    if config.loader is None:
        return dataset, None
    collator = None
    if config.preprocessing.input_type == "poses":
        collator = DenselyAnnotatedPoseDataCollator(
            segment_codec_names=list(annot_codecs.keys()),
            flatten_poses=config.loader.flatten_pose,
        )
    loader = DataLoader(
        dataset,
        batch_size=config.loader.batch_size,
        shuffle=config.loader.shuffle,
        num_workers=config.loader.num_workers,
        persistent_workers=True if config.loader.num_workers > 0 else False,
        collate_fn=collator,
        pin_memory=config.loader.pin_memory,
    )
    return dataset, loader
