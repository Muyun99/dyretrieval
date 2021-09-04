# get_optimizer
# optimizer = dict(type='SGD', lr=0.0003, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='Adam', lr=0.00035, weight_decay=0.0001)

# TODO: grad_clip
# optimizer_config = dict(grad_clip=None)

# get_scheduler
warmup_epochs = 0
lr_scheduler = dict(type='MultiStepLR', milestones=[40, 70], gamma=0.1)
max_epoch = 120