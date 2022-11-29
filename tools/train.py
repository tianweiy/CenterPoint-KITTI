import argparse
import datetime
import glob
import os
from pathlib import Path
from numpy.core.records import array
from test import repeat_eval_ckpt, find_next_folder_name

import torch
import torch.distributed as dist
import torch.nn as nn
from tensorboardX import SummaryWriter
from memory_profiler import profile

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model
import shutil

def create_paths(args):
    """[Those paths are used for saving the checkpoints and tensorboard files. In order to distinguish betweehn the]

    Args:
        args ([type]): [description]

    Returns:
        [type]: [description]
    """
    log_name = "batch" + str(args.batch_size) + "_epochs" + str(args.epochs) + "_set" + str(args.set_size) +"_bipfn" + str(args.bifpn) + str("_WithSkip" if args.bifpn_skip else "_NoSkip")
    log_name += "_lr" + "{:.5f}".format(cfg.OPTIMIZATION.LR / cfg.OPTIMIZATION.DIV_FACTOR)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt' / log_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clear and os.path.exists(ckpt_dir):
        shutil.rmtree(str(ckpt_dir))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    subdirs = os.listdir(ckpt_dir)
    dir_name = find_next_folder_name(subdirs)
    config_ckpt = ckpt_dir
    ckpt_dir = config_ckpt / dir_name  if not args.testmode else config_ckpt / "test"
    
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tb_path = output_dir / 'tensorboard' / log_name

    if args.clear and os.path.exists(tb_path):
        shutil.rmtree(str(tb_path))
    tb_path.mkdir(parents=True, exist_ok=True)
    subdirs = os.listdir(tb_path)

    tb_path = tb_path / find_next_folder_name(subdirs) if not args.testmode else tb_path / "test"
    tb_path.mkdir(parents=True, exist_ok=True)
    
    return output_dir, ckpt_dir, config_ckpt, tb_path

    

def parse_config():
    """[Argumenrs given throught the cmd \ terminal]

    Returns:
        [EasyDict]: [args, dictionary of the arguments.]
        [EasyDict]: [cfg, configuration values from the configuration file under cfgs\cfgs]
    """    
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=4, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=100, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--set_size', type=float, default=None, help='Set percentage of dataset usage for training')
    parser.add_argument('--bifpn', type=int, nargs='*', default=[], help='<Required> Set number of bifpn blocks')
    parser.add_argument('--bifpn_skip', dest='bifpn_skip', action='store_true', help='Use skip connections with BiFPN blocks')
    parser.add_argument('--testmode', dest='testmode', action='store_true', help="Don't create another folder")
    parser.add_argument('--eval', type=bool, default=False, help='If to do evalutation at the end')
    parser.add_argument('--clear', dest='clear', action='store_true')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    return args, cfg

# @profile
def main():
    torch.cuda.empty_cache()
    args, cfg = parse_config()
 
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666)
        
    # -----------------------Create relevant names for folders, so we could diffrintate between different runs.---------------------------
    output_dir, ckpt_dir, config_ckpt, tb_path = create_paths(args)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))
    
    tb_log = SummaryWriter(log_dir=str(tb_path)) if cfg.LOCAL_RANK == 0 else None
    
    # -----------------------create dataloader & network & optimizer---------------------------
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        set_size_percentage=args.set_size,
        bifpn=args.bifpn,
        bifpn_skip=args.bifpn_skip,
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    if args.ckpt is not None:
        cfg.OPTIMIZATION.DIV_FACTOR = 20
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist, logger=logger)

    
    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
        
    else:
        ckpt_list = glob.glob(str(config_ckpt / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            it, start_epoch = model.load_params_with_optimizer(
                ckpt_list[-1], to_cpu=dist, optimizer=optimizer, logger=logger
            )
            last_epoch = start_epoch + 1

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # Validation data
    test_set, test_loader, test_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=logger,
        set_size_percentage=args.set_size,
        bifpn=args.bifpn,
        training=False
    )

    # -----------------------start training--------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    train_model(
        model,
        optimizer,
        train_loader,
        test_loader,
        model_func=model_fn_decorator(), # The method used for forward pass.
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        test_sampler=test_sampler,
        config=cfg
    )

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    # Original work evaluates all the ckpts at the end of the training.
    # We decided to that indivudialy with using the train.py
    if args.eval:

        test_set, test_loader, test_sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.batch_size,
            dist=dist_train, workers=args.workers, logger=logger,
            set_size_percentage=args.set_size,
            bifpn=args.bifpn,
            training=False
        )
        logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                    (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

        eval_output_dir = output_dir / 'eval' / 'eval_with_train'
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        args.start_epoch = max(args.epochs - 10, 0)  # Only evaluate the last 10 epochs

        repeat_eval_ckpt(
            model.module if dist_train else model,
            test_loader, args, eval_output_dir, logger, ckpt_dir,
            dist_test=dist_train
        )
        logger.info('**********************End evaluation %s/%s(%s)**********************' %
                    (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()
