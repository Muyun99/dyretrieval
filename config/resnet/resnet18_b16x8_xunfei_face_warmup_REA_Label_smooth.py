_base_ = [
    '../_base_/models/resnet18.py', '../_base_/datasets/xunfei_retrieval.py',
    '../_base_/schedulers/default_scheduler.py', '../_base_/default_runtime.py'
]

# warm up
warmup_epochs = 5


# RandomErasing
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
train_pipeline = [
    dict(type='Resize', size=(400, 300)),
    dict(type='RandomCrop', size=(384, 287)),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomErasing', erase_prob=0.5, min_area_ratio=0.02, max_area_ratio=0.4, aspect_range=(3 / 10, 10 / 3)),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

# LabelSmooth
loss1 = dict(type='CrossEntropyLabelSmooth', epsilon=0.1)