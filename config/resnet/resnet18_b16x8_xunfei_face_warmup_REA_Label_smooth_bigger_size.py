_base_ = [
    '../_base_/models/resnet18.py', '../_base_/datasets/xunfei_retrieval.py',
    '../_base_/schedulers/default_scheduler.py', '../_base_/default_runtime.py'
]

batch_size = 16

# warm up
warmup_epochs = 5
lr_scheduler = dict(type='MultiStepLR', milestones=[20, 25], gamma=0.1)
max_epoch = 30


# RandomErasing
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

train_pipeline = [
    dict(type='Resize', size=(875, 606)),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomErasing', erase_prob=0.5, min_area_ratio=0.02, max_area_ratio=0.4, aspect_range=(3 / 10, 10 / 3)),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

valid_pipeline = [
    dict(type='Resize', size=(875, 606)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='Resize', size=(875, 606)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
dataset_type = 'competition_dataset'
root = '/home/muyun99/data/dataset/competition_data/xunfei_eco_image_retrivel'
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix=root,
        pipeline=train_pipeline,
        ann_file='train_fold0.csv',
        test_mode=False),
    val=dict(
        type=dataset_type,
        data_prefix=root,
        pipeline=valid_pipeline,
        ann_file='valid_fold0.csv',
        test_mode=False),
    test=dict(
        type=dataset_type,
        data_prefix=root,
        pipeline=test_pipeline,
        ann_file='test.csv',
        test_mode=True))


# LabelSmooth
loss1 = dict(type='CrossEntropyLabelSmooth', epsilon=0.1)

resume_path = None