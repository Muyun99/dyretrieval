import copy
import random
import torch
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler
import argparse

from mmcv import Config, DictAction
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tools.torch_utils import *

from torch.cuda.amp import GradScaler, autocast
def parse_args():
    parser = argparse.ArgumentParser(description='Train a models')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--tag', help='the tag')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--device', help='device used for training')
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    args = parser.parse_args()

    return args

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, dataset, batch_size, num_instances):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, pid in enumerate(self.dataset.df['label']):
            self.index_dic[pid].append(index)

        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length

def sampler_test():
    # config/resnet/resnet18_b16x8_xunfei_face.py --tag efficientnet_b0_fold0_baseline --options "model=efficientnet_b0" "data.train.ann_file=train_fold0.csv" "data.val.ann_file=valid_fold0.csv"
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.config = args.config
    cfg.tag = args.tag

    if args.options is not None:
        cfg.merge_from_dict(args.options)

    set_seed(cfg)
    set_cudnn(cfg)
    set_work_dir(cfg)
    make_log_dir(cfg)
    save_config(cfg)
    set_gpu(cfg)
    log_func = lambda string='': print_log(string, cfg)

    ###################################################################################
    # Dataset, DataLoader
    ###################################################################################
    log_func('[i] train dataset is {}'.format(cfg.data.train.ann_file))
    log_func('[i] valid dataset is {}'.format(cfg.data.val.ann_file))
    train_dataset = build_dataset(cfg.data.train)
    valid_dataset = build_dataset(cfg.data.val)

    triplet_sampler = RandomIdentitySampler(dataset=train_dataset, batch_size=cfg.batch_size, num_instances=4)

    train_dataloader = build_dataloader(dataset=train_dataset, batch_size=cfg.batch_size,
                                        num_workers=cfg.num_workers, shuffle=False, pin_memory=False)
    train_sampler_dataloader = build_dataloader(dataset=train_dataset, batch_size=cfg.batch_size, sampler=triplet_sampler,
                                        num_workers=cfg.num_workers, shuffle=False, pin_memory=False)
    valid_dataloader = build_dataloader(dataset=valid_dataset, batch_size=cfg.batch_size,
                                        num_workers=cfg.num_workers, shuffle=False, pin_memory=False)
    for index, data in enumerate(train_dataloader):
        images, labels = data['img'], data['gt_label']
        print(f'shape is {images.shape}')
        print(labels)
        break

    for index, data in enumerate(train_sampler_dataloader):
        images, labels = data['img'], data['gt_label']
        print(f'shape is {images.shape}')
        print(labels)
        break

if __name__ == '__main__':
    sampler_test()