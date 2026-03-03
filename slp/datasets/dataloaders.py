from torch.utils.data import DataLoader
from sldl import SignLanguageDataset, SignLanguageCollator

from slp.core.config.dataset import DataLoaderConfig


def load_dataloader(dataset: SignLanguageDataset, config: DataLoaderConfig):
    # noinspection PyTypeChecker
    return DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.n_workers,
        pin_memory=config.pin_memory,
        collate_fn=SignLanguageCollator(targets=dataset.targets),
    )
