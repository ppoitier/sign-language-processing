from collections import Counter
import numpy as np
from torch.utils.data import Dataset, DataLoader
import webdataset as wds

from slp.config.templates.data import RecognitionDatasetConfig
from slp.transforms.pose_pipelines import get_pose_pipeline
from slp.data.dataloader import PoseDataCollator


def _get_wds_mapping_fn(
    body_regions=(
            "upper_pose",
            "left_hand",
            "right_hand",
            # "lips",
    ),
):
    def _map_wds_to_sample(wds_sample):
        return {
            'id': wds_sample['__key__'],
            'poses': {
                body_region: wds_sample[f'pose.{body_region}.npy']
                for body_region in body_regions
            },
            'label': int(wds_sample['label.idx']),
        }
    return _map_wds_to_sample


def _compute_label_occurrences(samples: list[dict]):
    label_counter = Counter([s['label'] for s in samples])
    occurrences = np.zeros(len(label_counter))
    for label, count in label_counter.items():
        occurrences[int(label)] = count
    return occurrences


class IsolatedSignsRecognitionDataset(Dataset):
    def __init__(self, url: str, pose_transforms=None, verbose=False):
        super().__init__()
        self.pose_transforms = pose_transforms
        self.verbose = verbose
        web_dataset = wds.DataPipeline(
            wds.SimpleShardList(url),
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.decode(),
            wds.map(_get_wds_mapping_fn()),
        )
        print("Loading instances from shards:", url)
        self.samples = list(web_dataset)

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
        verbose=config.verbose,
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
