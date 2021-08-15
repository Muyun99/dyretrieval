import numpy as np
import random
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from functools import partial
from torch.utils.data import DataLoader
from mmcv.utils import Registry, build_from_cfg

PIPELINES = Registry('pipeline')
DATASETS = Registry('dataset')

def build_dataset(cfg):
    dataset = build_from_cfg(cfg, DATASETS)
    return dataset


def build_dataloader(dataset,
                     batch_size,
                     num_workers,
                     shuffle=True,
                     seed=None,
                     drop_last=False,
                     pin_memory=True,
                     sampler=None,
                     **kwargs):
    rank, world_size = get_dist_info()
    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=batch_size),
        pin_memory=pin_memory,
        shuffle=shuffle,
        worker_init_fn=init_fn,
        drop_last=drop_last,
        **kwargs)

    return data_loader

def worker_init_fn(worker_id, num_workers, rank, seed):
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)