# get_optimizer
# optimizer = dict(type='SGD', lr=0.0003, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='Adam', lr=0.0003, weight_decay=0.0001)

# TODO: grad_clip
# optimizer_config = dict(grad_clip=None)

# get_scheduler
lr_scheduler = dict(type='MultiStepLR', milestones=[30, 40], gamma=0.85)

max_epoch = 50