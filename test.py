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
    model.eval()
    test_features = []
    test_submit = []
    test_submit_extend = []
    test_path = test_dataset.df['filename'].tolist()
    test_path = [i.split('/')[-1] for i in test_path]
    test_path = np.array(test_path)

    with torch.no_grad():
        for data in test_dataloader:
            images = data['img']
            images = images.to(device)
            if cfg.fp16 is True:
                with autocast():
                    feat = model(images)
            else:
                feat = model(images)
            test_features.append(feat.data.cpu().numpy())

    test_features = np.vstack(test_features)
    test_features = normalize(test_features)

    list_original_distance = []
    for path, feature in zip(test_path[:], test_features[:]):
        distance = np.dot(feature, test_features.T)
        list_original_distance.append(distance)
        pred = [x.split('/')[-1] for x in test_path[np.where(distance > 0.9)[0]]]
        if len(pred) <= 1:
            ids = distance.argsort()[::-1]
            pred = [x.split('/')[-1] for x in test_path[ids[:2]]]

        test_submit.append([
            path.split('/')[-1],
            pred
        ])

    list_extend_distance = []
    for path, feature in zip(test_path[:], test_features[:]):
        original_distance = np.dot(feature, test_features.T)
        feature_qe = np.multiply(original_distance[np.argsort(original_distance)[::-1][:2]].reshape(2, -1), test_features[np.argsort(original_distance)[::-1][:2]]).mean(0)
        extend_distance = np.dot(feature_qe, test_features.T)
        list_extend_distance.append(extend_distance)

        pred = [x.split('/')[-1] for x in test_path[np.where(extend_distance > 0.9)[0]]]
        if len(pred) <= 1:
            ids = extend_distance.argsort()[::-1]
            pred = [x.split('/')[-1] for x in test_path[ids[:2]]]

        test_submit_extend.append([
            path.split('/')[-1],
            pred
        ])

    array_original_distance = np.stack(list_original_distance)
    array_extend_distance = np.stack(list_extend_distance)
    np.save(os.path.join(cfg.work_dir, 'distance.npy', ), array_original_distance)
    np.save(os.path.join(cfg.work_dir, 'feature.npy', ), test_features)
    np.save(os.path.join(cfg.work_dir, 'extend_distance.npy', ), array_extend_distance)

    pd_test_submit = pd.DataFrame(test_submit, columns=['name', 'label'])
    pd_test_submit['label'] = pd_test_submit['label'].apply(lambda x: ' '.join(x))
    pd_test_submit.to_csv('submit.csv', index=False)

    pd_test_submit_extend = pd.DataFrame(test_submit_extend, columns=['name', 'label'])
    pd_test_submit_extend['label'] = pd_test_submit_extend['label'].apply(lambda x: ' '.join(x))
    pd_test_submit_extend.to_csv('submit_extend.csv', index=False)

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
    # cfg.num_classes = state_dict['margin.weight'].shape[0]
    cfg.num_classes = state_dict['classifier.weight'].shape[0]
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
