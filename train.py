import argparse

from mmcv import Config, DictAction
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from dyret import build_model, build_optimizer, build_loss, build_scheduler, build_dataset, build_dataloader, build_sampler
from tools.torch_utils import *

from torch.cuda.amp import GradScaler, autocast

def parse_args():
    parser = argparse.ArgumentParser(description='Train a models')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--tag', help='the tag')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--device', help='device used for training')
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    args = parser.parse_args()

    return args

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
    save_config(cfg)
    set_gpu(cfg)
    log_func = lambda string='': print_log(string, cfg)

    ###################################################################################
    # Dataset, DataLoader
    ###################################################################################
    log_func('[i] train dataset is {}'.format(cfg.data.train.ann_file))
    log_func('[i] valid dataset is {}'.format(cfg.data.val.ann_file))
    train_dataset = build_dataset(cfg.data.train)
    valid_dataset = build_dataset(cfg.data.val)
    if cfg.dataset_sampler_type is None:
        train_dataloader = build_dataloader(dataset=train_dataset, batch_size=cfg.batch_size,
                                            num_workers=cfg.num_workers, shuffle=True, pin_memory=False)
    else:
        train_sampler = build_sampler(cfg, train_dataset)
        train_dataloader = build_dataloader(dataset=train_dataset, batch_size=cfg.batch_size, sampler=train_sampler,
                                            num_workers=cfg.num_workers, shuffle=False, pin_memory=False)

    valid_dataloader = build_dataloader(dataset=valid_dataset, batch_size=cfg.batch_size,
                                        num_workers=cfg.num_workers, shuffle=False, pin_memory=False)

    ###################################################################################
    # Network
    ###################################################################################
    cfg.num_classes = train_dataset.num_classes
    model = build_model(cfg)
    model = model.cuda()
    model.train()
    log_func('[i] Architecture is {}'.format(cfg.model))
    log_func('[i] Total Params: %.2fM' % (calculate_parameters(model)))

    if cfg.resume_path is not None:
        state_dict = torch.load(cfg.resume_path)
        model.load_state_dict(state_dict)
        log_func(f'[i] Loading weight from: {cfg.model_path}')
    if cfg.parallel:
        model = nn.DataParallel(model)


    load_model_func = lambda: load_model(model, cfg.model_path, parallel=cfg.parallel)
    save_model_func = lambda: save_model(model, cfg.model_path, parallel=cfg.parallel)

    ###################################################################################
    # Loss, Optimizer, LR_scheduler
    ###################################################################################
<<<<<<< HEAD
    id_loss = build_loss(cfg)

    def criterion(score, feature, target):
        # return id_loss(score, target) + triplet_loss(feature, target)[0]
        return id_loss(score, target)
=======
    # id_loss 取值 CrossEntropyLoss、CrossEntropyLabelSmooth
    id_loss = build_loss(cfg.loss1, num_classes=cfg.num_classes)

    # triplet_loss 取值 TripletLoss
    triplet_loss = build_loss(cfg.loss2)

    def criterion(score, feature, target):
        return id_loss(score, target) + triplet_loss(feature, target)[0]
>>>>>>> 5b282ba08b0dbfa80cc6e1cfce0ba624310b8cfd

    optimizer = build_optimizer(cfg=cfg, model=model)
    scheduler = build_scheduler(cfg=cfg, optimizer=optimizer)

    ###################################################################################
    # Train
    ###################################################################################
    train_timer = Timer()
    eval_timer = Timer()
    best_f1_score = -1
    best_threahold = -1
    val_iteration = len(train_dataloader)
    log_iteration = val_iteration
    max_iteration = cfg.max_epoch * val_iteration
    log_func(f'[i] val_iteration : {val_iteration}')
    log_func(f'[i] max_iteration : {max_iteration}')

    writer = SummaryWriter(cfg.tensorboard_dir)
    train_iterator = Iterator(train_dataloader)
    train_meter = Average_Meter(['loss'])
    iteration = 0


    if cfg.fp16 is True:
        scaler = GradScaler()
    else:
        scaler = None

    for iteration in range(max_iteration):
        data = train_iterator.get()
        images, labels = data['img'], data['gt_label']
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)

        optimizer.zero_grad()
        if cfg.fp16 is True:
            with autocast():
<<<<<<< HEAD
                cls_score, global_features = model(images, labels)
=======
                cls_score, global_features = model(images)
>>>>>>> 5b282ba08b0dbfa80cc6e1cfce0ba624310b8cfd
                loss = criterion(score=cls_score, feature=global_features, target=labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
<<<<<<< HEAD
            cls_score, global_features = model(images, labels)
=======
            cls_score, global_features = model(images)
>>>>>>> 5b282ba08b0dbfa80cc6e1cfce0ba624310b8cfd
            loss = criterion(score=cls_score, feature=global_features, target=labels)
            loss.backward()
            optimizer.step()

        train_meter.add({
            'loss': loss.item(),
        })

        #################################################################################################
        # For Log
        #################################################################################################
        if (iteration + 1) % log_iteration == 0:
            loss = train_meter.get(clear=True)
            learning_rate = float(get_learning_rate_from_optimizer(optimizer))
            time = train_timer.tok(clear=True)

            log_func(f'[i] iteration = {iteration + 1} \
                time = {time} sec \
                lr = {learning_rate} \
                loss = {loss:.4f}')

            writer.add_scalar('Train/loss', loss, iteration)
            writer.add_scalar('Train/learning_rate', learning_rate, iteration)

        #################################################################################################
        # Evaluation
        #################################################################################################
        if (iteration + 1) % val_iteration == 0:
            eval_timer.tik()
            model.eval()

            threahold, f1_score = valid_dataset.evaluate(cfg=cfg, model=model, valid_dataloader=valid_dataloader)

            model.train()
            time = eval_timer.tok(clear=True)

            if best_f1_score < f1_score:
                best_f1_score = f1_score
                best_threahold = threahold
                save_model_func()
                best_flag = 1
            else:
                best_flag = 0


            log_func(f'[i] iteration = {iteration + 1} \
                time = {time} sec \
                valid_F1 = {f1_score*100 :.4f} \
                best_valid_F1 = {best_f1_score*100 :.4f}'
                )
            if best_flag == 1:
                log_func('[i] save models')



            writer.add_scalar('Evaluation/valid_F1', f1_score, iteration)
            writer.add_scalar('Evaluation/best_valid_F1', best_f1_score, iteration)
            writer.add_scalar('Evaluation/best_threahold', best_threahold, iteration)

        #################################################################################################
        # For Step()
        #################################################################################################
        if (iteration + 1) % log_iteration == 0:
            scheduler.step()
        iteration += 1
    writer.close()
