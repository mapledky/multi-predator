import os, torch, time, shutil, json,glob, argparse, shutil
import numpy as np
from easydict import EasyDict as edict

from datasets.dataloader import get_dataloader, get_datasets
from models.architectures import KPFCNN
from lib.utils import setup_seed, load_config
from lib.tester import get_trainer
from lib.loss import MetricLoss
from configs.models import architectures

from torch import optim
from torch import nn

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
setup_seed(0)

"""
python code/OverlapPredator/main.py code/OverlapPredator/configs/train/front.yaml
python -m torch.distributed.launch --nproc_per_node 4 code/OverlapPredator/main.py code/OverlapPredator/configs/train/front.yaml
"""

if __name__ == '__main__':
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help= 'Path to the config file.')
    parser.add_argument("--local_rank", default=0)
    args = parser.parse_args()
    config = load_config(args.config)
    local_rank = args.local_rank
    torch.cuda.set_device(f'cuda:{local_rank}')

    config['snapshot_dir'] = 'code/OverlapPredator/snapshot/%s' % config['exp_dir']
    config['tboard_dir'] = 'code/OverlapPredator/snapshot/%s/tensorboard' % config['exp_dir']
    config['save_dir'] = 'code/OverlapPredator/snapshot/%s/checkpoints' % config['exp_dir']
    config = edict(config)

    os.makedirs(config.snapshot_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.tboard_dir, exist_ok=True)
    json.dump(
        config,
        open(os.path.join(config.snapshot_dir, 'config.json'), 'w'),
        indent=4,
    )
    if config.gpu_mode:
        config.device = torch.device('cuda')
    else:
        config.device = torch.device('cpu')
    
    if config.distributed:
        dist.init_process_group(backend='nccl')
        device = torch.device("cuda",  int(local_rank))
        config.device = device
        config.local_rank = local_rank
    # backup the files
    os.system(f'cp -r code/OverlapPredator/models {config.snapshot_dir}')
    os.system(f'cp -r code/OverlapPredator/datasets {config.snapshot_dir}')
    os.system(f'cp -r code/OverlapPredator/lib {config.snapshot_dir}')
    shutil.copy2('code/OverlapPredator/main.py',config.snapshot_dir)
    
    
    # model initialization
    config.architecture = architectures[config.dataset]
    config.model = KPFCNN(config)   

    # create optimizer 
    if config.optimizer == 'SGD':
        config.optimizer = optim.SGD(
            config.model.parameters(), 
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            )
    elif config.optimizer == 'ADAM':
        config.optimizer = optim.Adam(
            config.model.parameters(), 
            lr=config.lr,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
        )
    
    # create learning rate scheduler
    config.scheduler = optim.lr_scheduler.ExponentialLR(
        config.optimizer,
        gamma=config.scheduler_gamma,
    )
    
    # create dataset and dataloader
    train_set, val_set, benchmark_set = get_datasets(config)
    if config.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
    else:
        train_sampler = None
        val_sampler = None
    if config.distributed:
        config.train_loader, neighborhood_limits = get_dataloader(dataset=train_set,
                                            batch_size=config.batch_size,
                                            num_workers=config.num_workers,
                                            sampler=train_sampler
                                            )
        config.val_loader, _ = get_dataloader(dataset=val_set,
                                            batch_size=config.batch_size,
                                            num_workers=1,
                                            neighborhood_limits=neighborhood_limits,
                                            sampler=val_sampler
                                            )
    else:
        config.train_loader, neighborhood_limits = get_dataloader(dataset=train_set,
                                            batch_size=config.batch_size,
                                            shuffle=True,
                                            num_workers=config.num_workers,
                                            )
        config.val_loader, _ = get_dataloader(dataset=val_set,
                                            batch_size=config.batch_size,
                                            shuffle=False,
                                            num_workers=1,
                                            neighborhood_limits=neighborhood_limits,
                                            )
    config.test_loader, _ = get_dataloader(dataset=benchmark_set,
                                        batch_size=config.batch_size,
                                        shuffle=False,
                                        num_workers=1,
                                        neighborhood_limits=neighborhood_limits)
    
    # create evaluation metrics
    config.desc_loss = MetricLoss(config)
    trainer = get_trainer(config)
    if(config.mode=='train'):
        trainer.train()
    elif(config.mode =='val'):
        trainer.eval()
    else:
        trainer.test()        