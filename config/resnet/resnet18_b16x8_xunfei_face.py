_base_ = [
    '../_base_/models/resnet18.py', '../_base_/datasets/xunfei_retrieval.py',
    '../_base_/schedulers/default_scheduler.py', '../_base_/default_runtime.py'
]

# 查看是不是sampler的问题
# dataset_sampler_type = None

