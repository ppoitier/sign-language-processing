from collections import Counter

import numpy as np
import orjson
import webdataset as wds
from torch.utils.data import Dataset, DataLoader
from torchcodec.decoders import VideoDecoder

from slp.config.templates.data import RecognitionDatasetConfig
from slp.data.dataloader import PoseDataCollator
from slp.transforms.pose_pipelines import get_pose_pipeline


def _read_json(filepath: str):
    with open(filepath, 'rb') as f:
        return orjson.loads(f.read())


def _get_wds_mapping_fn(
    body_regions=(
            "upper_pose",
            "left_hand",
            "right_hand",
            # "lips",
    ),
    label_mapping=None,
):
    def _map_wds_to_sample(wds_sample):
        label = str(wds_sample['label.txt'])
        return {
            'id': wds_sample['__key__'],
            'poses': {
                body_region: wds_sample[f'pose.{body_region}.npy']
                for body_region in body_regions
            },
            'label': label,
            'label_id': int(wds_sample['label.idx']) if label_mapping is None else label_mapping[label],
        }
    return _map_wds_to_sample


def _compute_label_occurrences(samples: list[dict]):
    label_counter = Counter([int(s['label']) for s in samples])
    occurrences = np.zeros(max(label_counter.keys())+1)
    for label, count in label_counter.items():
        occurrences[label] = count
    return occurrences


class IsolatedSignsRecognitionDataset(Dataset):
    def __init__(
            self,
            url: str,
            include_poses: bool = True,
            include_videos: bool = False,
            video_dir: str | None = None,
            video_ext: str = 'mp4',
            gpu_video_decoding: bool = False,
            pose_transforms=None,
            video_transform=None,
            split_filepath: str | None = None,
            label_mapping_filepath: str | None = None,
    ):
        super().__init__()
        self.include_poses = include_poses
        self.include_videos = include_videos

        self.video_dir = video_dir
        self.video_ext = video_ext
        self.video_gpu_decoding = gpu_video_decoding

        self.pose_transforms = pose_transforms
        self.video_transform = video_transform

        external_label_mapping = None
        if label_mapping_filepath is not None:
            print(f"Loading external label mapping: [{label_mapping_filepath}]...")
            external_label_mapping = _read_json(label_mapping_filepath)

        split = None
        if split_filepath is not None:
            print(f"Keep only samples in split: [{split_filepath}]...")
            split = set(_read_json(split_filepath))

        web_dataset = wds.DataPipeline(
            wds.SimpleShardList(url),
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.decode(),
            wds.select(lambda sample: (sample['__key__'] in split) if split is not None else True),
            wds.map(_get_wds_mapping_fn(label_mapping=external_label_mapping)),
        )
        print("Loading instances from shards:", url)
        self.samples = list(web_dataset)

        self.label_to_idx = {label: label_id for label, label_id in set((sample['label'], sample['label_id']) for sample in self.samples)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        print(f"Dataset loaded: {len(self.samples)} instances.")


    def get_label_occurrences(self):
        return _compute_label_occurrences(self.samples)

    def get_label_frequencies(self):
        occurrences = self.get_label_occurrences()
        return occurrences / occurrences.sum()

    def get_label_weights(self):
        return 1 / self.get_label_frequencies()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = {**self.samples[idx]}

        if self.pose_transforms is not None:
            sample['poses'] = self.pose_transforms(sample['poses'])

        if self.include_videos:
            decoder = VideoDecoder(f"{self.video_dir}/{sample['id']}.{self.video_ext}", device='cuda' if self.video_gpu_decoding else 'cpu')
            sample['video'] = decoder.get_frames_in_range(start=0, stop=decoder.metadata.num_frames, step=1).data

        if self.video_transform is not None:
            sample['video'] = self.video_transform(sample['video'])

        return sample


def load_pose_dataset(config: RecognitionDatasetConfig) -> tuple[IsolatedSignsRecognitionDataset, DataLoader | None]:
    pose_transforms = None
    if config.preprocessing is not None:
        pose_transforms = get_pose_pipeline(
            config.preprocessing.pose_transforms_pipeline
        )
    dataset = IsolatedSignsRecognitionDataset(
        url=config.shards_url,
        pose_transforms=pose_transforms,
        split_filepath=config.split_filepath,
        label_mapping_filepath=config.label_mapping_filepath,
        include_videos=config.include_videos,
        video_dir=config.video_dir,
    )
    if config.loader is None:
        return dataset, None
    loader_config = config.loader
    loader = DataLoader(
        dataset,
        batch_size=loader_config.batch_size,
        shuffle=loader_config.shuffle,
        num_workers=loader_config.num_workers,
        persistent_workers=True if loader_config.num_workers > 0 else False,
        pin_memory=loader_config.pin_memory,
        collate_fn=PoseDataCollator(flatten_poses=loader_config.flatten_pose),
    )
    return dataset, loader
