from .triplet_sampler import RandomIdentitySampler

def build_sampler(cfg, dataset):
    name_sampler = cfg.dataset_sampler_type
    sampler = None

    if name_sampler == 'Triplet_sampler':
        sampler = RandomIdentitySampler(dataset=dataset, batch_size=cfg.batch_size, num_instances=cfg.num_instances)

    if sampler is None:
        raise Exception('scheduler is wrong')
    return sampler