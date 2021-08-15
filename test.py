import pandas as pd
import numpy as np
import copy

import ttach as tta
from mmcv import Config
from tqdm import tqdm

from dyret import build_dataset, build_dataloader, build_model
from tools.torch_utils import *
from train import parse_args
from torch.cuda.amp import autocast
from torchsummary import summary
from sklearn.preprocessing import normalize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_test_features():
    test_features = []
    test_submit = []
    test_feats_fold = []
    test_path = test_dataset.df['filename'].tolist()
    test_path = [i.split('/')[-1] for i in test_path]
    test_path = np.array(test_path)

    with torch.no_grad():
        for data in test_dataloader:
            images = data['img']
            images = images.to(device)
            feat = model(images)
            test_features.append(feat.data.cpu().numpy())

    test_features = np.vstack(test_features)
    test_features = normalize(test_features)

    test_feats_fold.append(test_features)
    test_features = np.stack(test_feats_fold).mean(0)

    # test_features_list = list(test_features)
    # test_path_list = list(test_path)

    list_distance = []
    for path, feature in zip(test_path[:], test_features[:]):
        distance = np.dot(feature, test_features.T)
        list_distance.append(distance)
        pred = [x.split('/')[-1] for x in test_path[np.where(distance > 0.9)[0]]]
        if len(pred) <= 1:
            ids = distance.argsort()[::-1]
            pred = [x.split('/')[-1] for x in test_path[ids[:2]]]

        test_submit.append([
            path.split('/')[-1],
            pred
        ])

    array_distance = np.stack(list_distance)
    np.save(os.path.join(cfg.work_dir, 'distance.npy', ), array_distance)
    np.save(os.path.join(cfg.work_dir, 'feature.npy', ), test_features)


    test_submit = pd.DataFrame(test_submit, columns=['name', 'label'])
    test_submit['label'] = test_submit['label'].apply(lambda x: ' '.join(x))
    test_submit.to_csv('submit.csv', index=False)

if __name__ == '__main__':
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
    set_gpu(cfg)

    state_dict = torch.load(cfg.model_path)
    cfg.num_classes = state_dict['margin.weight'].shape[0]
    log_func = lambda string='': print_log(string, cfg)
    test_dataset = build_dataset(cfg.data.test)
    test_dataloader = build_dataloader(dataset=test_dataset, batch_size=cfg.batch_size,
                                       num_workers=cfg.num_workers, shuffle=False, pin_memory=False)

    model = build_model(cfg)
    model = model.cuda()
    log_func(f'[i] Architecture is {cfg.model}')
    log_func(f'[i] Total Params: %.2fM' % (calculate_parameters(model)))
    model.load_state_dict(state_dict)
    log_func(f'[i] Loading weight from: {cfg.model_path}')

    get_test_features()
