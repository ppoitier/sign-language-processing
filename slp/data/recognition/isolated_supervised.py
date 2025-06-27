from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset, default_collate
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
import webdataset as wds


def _map_wds_to_sample(wds_sample):
    return {
        'id': wds_sample['__key__'],
        'poses': {
            body_region: wds_sample[f'pose.{body_region}.npy']
            for body_region in ('upper_pose', 'left_hand', 'right_hand', 'lips')
        },
        'label': int(wds_sample['label.idx']),
    }


def _compute_label_occurrences(samples: list[dict]):
    label_counter = Counter([s['label'] for s in samples])
    occurrences = np.zeros(len(label_counter))
    for label, count in label_counter.items():
        occurrences[int(label)] = count
    return occurrences


class ISLRCollator:
    def __init__(
            self,
            body_regions = ('upper_pose', 'left_hand', 'right_hand', 'lips'),
            min_length: int | None = None,
            flatten_poses: bool = False,
    ):
        self.body_regions = body_regions
        self.min_length = min_length
        self.flatten_poses = flatten_poses

    def __call__(self, batch):
        if self.flatten_poses:
            lengths = [b["poses"].shape[0] for b in batch]
            poses = pad_sequence([torch.from_numpy(b['poses']) for b in batch], batch_first=True).float()
        else:
            lengths = [b["poses"][self.body_regions[0]].shape[0] for b in batch]
            poses = {k: pad_sequence([torch.from_numpy(b['poses'][k]) for b in batch], batch_first=True).float() for k in self.body_regions}
        masks = pad_sequence([torch.ones(l, dtype=torch.uint8) for l in lengths], batch_first=True, padding_value=0)
        [b.pop('poses') for b in batch]
        batch = default_collate(batch)
        batch['lengths'] = lengths
        batch['masks'] = masks
        batch['poses'] = poses
        if self.min_length is not None:
            padding_size = self.min_length - batch['masks'].shape[1]
            batch['masks'] = pad(batch['masks'], (0, padding_size), value=0)
            if self.flatten_poses:
                batch['poses'] = pad(batch['poses'], (0, 0, 0, padding_size), value=0.0)
            else:
                batch['poses'] = {k: pad(v, (v.ndim - 2) * (0, 0) + (0, padding_size), value=0.0) for k, v in batch['poses'].items()}
        return batch

class IsolatedSignsRecognition(Dataset):
    def __init__(self, url: str, pose_transforms=None, verbose=False):
        super().__init__()
        self.pose_transforms = pose_transforms
        self.verbose = verbose
        web_dataset = wds.DataPipeline(
            wds.SimpleShardList(url),
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.decode(),
            wds.map(_map_wds_to_sample),
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
