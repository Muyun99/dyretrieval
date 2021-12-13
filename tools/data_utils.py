from sklearn.model_selection import train_test_split, StratifiedKFold
from mmcv.utils import Config
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import copy
from tqdm import tqdm
import mmcv


def split_train_val():
    train_df = pd.read_csv(cfg.path_train_csv)
    train_df.columns = ['filename', 'label']

    if cfg.num_KFold > 1:
        train_df['fold'] = train_df['label'] % cfg.num_KFold

        for fold_idx in range(cfg.num_KFold):
            fold_train = copy.deepcopy(train_df[train_df['fold'] != fold_idx])
            fold_train['label'] = (pd.factorize(fold_train['label'])[0])
            fold_train.drop(columns=['fold'], inplace=True)

            fold_valid = copy.deepcopy(train_df[train_df['fold'] == fold_idx])
            fold_valid.drop(columns=['fold'], inplace=True)

            demo_train = fold_train[:100]
            demo_test = fold_train[:100]
            demo_train.to_csv(os.path.join(cfg.path_save_trainval_csv, f'train_fold_demo.csv'), index=False)
            demo_test.to_csv(os.path.join(cfg.path_save_trainval_csv, f'valid_fold_demo.csv'), index=False)

            fold_train.to_csv(os.path.join(cfg.path_save_trainval_csv, f'train_fold{fold_idx}.csv'), index=False)
            fold_valid.to_csv(os.path.join(cfg.path_save_trainval_csv, f'valid_fold{fold_idx}.csv'), index=False)
            print(f'train_fold{fold_idx}: {fold_train.shape[0]}, valid_fold{fold_idx}: {fold_valid.shape[0]}')
    else:
        train_data, valid_data = train_test_split(
            train_df, shuffle=True, test_size=cfg.size_valid, random_state=cfg.seed_random)
        train_data.to_csv(os.path.join(cfg.path_save_trainval_csv, f'train.csv'), index=False)
        valid_data.to_csv(os.path.join(cfg.path_save_trainval_csv, f'valid.csv'), index=False)
        print(f'train:{train_data.shape[0]}, valid:{valid_data.shape[0]}')

def generate_train_csv():
    train_df = pd.read_csv(cfg.path_raw_train_csv)
    train_df['filename'] = train_df['name'].apply(
        lambda item: os.path.join(cfg.path_train_img, item.split('/')[-1])
    )
    train_df['label'] = pd.factorize(train_df['label'])[0]
    train_df = train_df.sort_values(by='label')
    train_df = pd.concat([train_df['filename'], train_df['label']], axis=1)
    train_df.to_csv(cfg.path_train_csv, index=False)


def generate_test_csv():
    test_imgs = glob(os.path.join(cfg.path_test_img, '*.jpg'))
    test_dict = dict()
    for img in test_imgs:
        test_dict[img] = None
    test_df = pd.DataFrame(test_imgs)
    test_df.columns = ['filename']
    test_df = test_df.sort_values(by='filename')
    test_df.to_csv(cfg.path_test_csv, index=False)


def example():
    df = pd.read_csv(os.path.join(cfg.path_save_trainval_csv, 'train_fold0.csv'))
    print(len(df['label'].unique()))
    print(df['label'].max())

def get_id_distribution():
    df = pd.read_csv(os.path.join(cfg.path_save_trainval_csv, 'train_fold0.csv'))
    print(df['label'].describe())
    df.plot.hist(bins=len(df['label'].unique()), alpha=0.5)
    plt.show()

def generate_img():
    train_imgs = np.load(cfg.path_raw_train_npy)
    print(train_imgs.shape)
    
    for idx in tqdm(range(train_imgs.shape[0])):
        mmcv.imwrite(train_imgs[idx, :, :, :], os.path.join(cfg.path_train_img, f'{idx}.png'))



if __name__ == '__main__':
    cfg = Config.fromfile('config_data_utils_test.py')
    generate_img()

    # generate_train_csv()
    # generate_test_csv()
    # split_train_val()
    # example()
    # get_id_distribution()
