import os, sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from model.model_builder import init_model
from model import *
from model.DeeplabV2 import *
from init_config import *
from easydict import EasyDict as edict
import sys
from trainer.self_dist_trainer import DistTrainer
import copy
import numpy as np
import random
import argparse
import wandb


##!! remove key when making local
os.environ["WANDB_API_KEY"]="8f4eb3e6949541646d5dfc27a84f62fecd62413c"

## uncomment following lines for debugging
# os.environ["WANDB_SILENT"] = "true"
# os.environ["WANDB_MODE"] = "dryrun"


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', dest='config_file',
                        help='configuration filename',
                        default="config/damnet_config_upsize_dist.yml", 
                        type=str)
    return parser.parse_args()

def freeze_model(model):
    for module in model.modules():
        for i in module.parameters():
            i.requires_grad = False

def main():
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    args = parse_args()
    if args.config_file is None:
        raise Exception('no configuration file')

    config, writer = init_config(args.config_file, sys.argv)
    
    wandb.login(timeout=300)
    wandb.init(
                entity="nik1806",
                project='DASS-retraining', 
                name="Run: self-distillation training",
                config=config,
                )
    # wandb.run.log_code(".") # to save additional code

    config.num_classes = 19

    teacher_model = init_model(config)
    freeze_model(teacher_model) # freeze teacher model

    student_model = ResPair_Deeplab(num_classes=config.num_classes, cfg=config).cuda()
    params = torch.load(config.dist_weight)
    student_model.load_state_dict(params['state_dict'], strict=False)

    trainer = DistTrainer(teacher_model, student_model, config, writer)

    trainer.train()
# -

if __name__ == "__main__":
    main()
