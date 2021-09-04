from mmcv import Config
from tools.torch_utils import *
from train import parse_args
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    path_test_csv = os.path.join(cfg.data.test.data_prefix, cfg.data.test.ann_file)
    df_test = pd.read_csv(path_test_csv)
    test_path = df_test['filename']

    tag_b0_list = ['resnet18_b16x8_xunfei_face_tag_efficientnet_b0_fold0',
                   'resnet18_b16x8_xunfei_face_tag_efficientnet_b0_fold1',
                   'resnet18_b16x8_xunfei_face_tag_efficientnet_b0_fold2',
                   'resnet18_b16x8_xunfei_face_tag_efficientnet_b0_fold3',
                   'resnet18_b16x8_xunfei_face_tag_efficientnet_b0_fold4']
    tag_b3_list = ['resnet18_b16x8_xunfei_face_tag_efficientnet_b3_fold0',
                   'resnet18_b16x8_xunfei_face_tag_efficientnet_b3_fold1',
                   'resnet18_b16x8_xunfei_face_tag_efficientnet_b3_fold2',
                   'resnet18_b16x8_xunfei_face_tag_efficientnet_b3_fold3',
                   'resnet18_b16x8_xunfei_face_tag_efficientnet_b3_fold4']
    tag_21k_list = ['resnet18_b16x8_xunfei_face_tag_tf_efficientnetv2_s_in21k_fold0',
                    'resnet18_b16x8_xunfei_face_tag_tf_efficientnetv2_s_in21k_fold1',
                    'resnet18_b16x8_xunfei_face_tag_tf_efficientnetv2_s_in21k_fold2',
                    'resnet18_b16x8_xunfei_face_tag_tf_efficientnetv2_s_in21k_fold3',
                    'resnet18_b16x8_xunfei_face_tag_tf_efficientnetv2_s_in21k_fold4']

    tag_test = ['resnet18_b16x8_xunfei_face_tag_new_baseline_efficientnet_b0_fold_0',
                'resnet18_b16x8_xunfei_face_warmup_tag_new_baseline_efficientnet_b0_fold_0',
                'resnet18_b16x8_xunfei_face_warmup_REA_tag_new_baseline_efficientnet_b0_fold_0']
    tag_new_baseline = ['resnet18_b16x8_xunfei_face_warmup_REA_Label_smooth_tag_new_baseline_efficientnet_b0_fold0',
                        'resnet18_b16x8_xunfei_face_warmup_REA_Label_smooth_tag_new_baseline_efficientnet_b0_fold1',
                        'resnet18_b16x8_xunfei_face_warmup_REA_Label_smooth_tag_new_baseline_efficientnet_b0_fold2']
    tag_b3_new_baseline = ['resnet18_b16x8_xunfei_face_warmup_REA_Label_smooth_tag_new_baseline_efficientnet_b3_fold0_120epoch',
                           'resnet18_b16x8_xunfei_face_warmup_REA_Label_smooth_tag_new_baseline_efficientnet_b3_fold1_120epoch',
                           'resnet18_b16x8_xunfei_face_warmup_REA_Label_smooth_tag_new_baseline_efficientnet_b3_fold2_120epoch',
                           'resnet18_b16x8_xunfei_face_warmup_REA_Label_smooth_tag_new_baseline_efficientnet_b3_fold3_120epoch',
                           'resnet18_b16x8_xunfei_face_warmup_REA_Label_smooth_tag_new_baseline_efficientnet_b3_fold4_120epoch']
    tag_b0_new_baseline = [
        'resnet18_b16x8_xunfei_face_warmup_REA_Label_smooth_tag_new_baseline_efficientnet_b0_fold0_120epoch',
        'resnet18_b16x8_xunfei_face_warmup_REA_Label_smooth_tag_new_baseline_efficientnet_b0_fold1_120epoch',
        'resnet18_b16x8_xunfei_face_warmup_REA_Label_smooth_tag_new_baseline_efficientnet_b0_fold2_120epoch',
        'resnet18_b16x8_xunfei_face_warmup_REA_Label_smooth_tag_new_baseline_efficientnet_b0_fold3_120epoch',
        'resnet18_b16x8_xunfei_face_warmup_REA_Label_smooth_tag_new_baseline_efficientnet_b0_fold4_120epoch']

    tag_b3_bigger = [
        'resnet18_b16x8_xunfei_face_warmup_REA_Label_smooth_bigger_size_tag_new_baseline_efficientnet_b3_fold0_120epoch',
        'resnet18_b16x8_xunfei_face_warmup_REA_Label_smooth_bigger_size_tag_new_baseline_efficientnet_b3_fold1_120epoch',
        'resnet18_b16x8_xunfei_face_warmup_REA_Label_smooth_bigger_size_tag_new_baseline_efficientnet_b3_fold2_120epoch',
        'resnet18_b16x8_xunfei_face_warmup_REA_Label_smooth_bigger_size_tag_new_baseline_efficientnet_b3_fold3_120epoch',
        'resnet18_b16x8_xunfei_face_warmup_REA_Label_smooth_bigger_size_tag_new_baseline_efficientnet_b3_fold4_120epoch',
    ]

    tag_b0_bigger = [
        'resnet18_b16x8_xunfei_face_warmup_REA_Label_smooth_bigger_size_tag_new_baseline_efficientnet_b0_fold0_120epoch',
        'resnet18_b16x8_xunfei_face_warmup_REA_Label_smooth_bigger_size_tag_new_baseline_efficientnet_b0_fold1_120epoch',
        'resnet18_b16x8_xunfei_face_warmup_REA_Label_smooth_bigger_size_tag_new_baseline_efficientnet_b0_fold2_120epoch'
    ]

    # tag_list = tag_new_baseline + tag_test + tag_b0_list + tag_b3_list + tag_21k_list
    # tag_list = tag_new_baseline + tag_test
    tag_list = tag_b3_new_baseline + tag_b0_new_baseline + tag_b3_bigger + tag_b0_bigger

    list_distances = []
    list_features = []
    for tag in tag_list:
        feature = np.load(os.path.join('work_dirs', tag, f'feature.npy'))
        distance = np.load(os.path.join('work_dirs', tag, f'extend_distance.npy'))

        list_features.append(feature)
        list_distances.append(distance)

    ensemble_distance = np.stack(list_distances).mean(0)
    ensemble_test_submit = []

    for path, vector_distance in zip(test_path[:], ensemble_distance[:]):
        pred = [x.split('/')[-1] for x in test_path[np.where(vector_distance > 0.925)[0]]]
        if len(pred) <= 1:
            ids = vector_distance.argsort()[::-1]
            pred = [x.split('/')[-1] for x in test_path[ids[:2]]]

        ensemble_test_submit.append([
            path.split('/')[-1],
            pred
        ])

    test_submit = pd.DataFrame(ensemble_test_submit, columns=['name', 'label'])
    test_submit['label'] = test_submit['label'].apply(lambda x: ' '.join(x))
    test_submit.to_csv('ensemble_submit.csv', index=False)