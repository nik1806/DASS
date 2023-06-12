import os, sys
import GPUtil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from dataset.dataset import *  # init_dataset
from model.model_builder import init_model
from model import *
from init_config import *
from easydict import EasyDict as edict
import sys
from trainer.source_only_trainer import Trainer
import trainer
import time, datetime
import copy
import numpy as np
import random
import wandb

##!! remove key when making public
os.environ["WANDB_API_KEY"]="8f4eb3e6949541646d5dfc27a84f62fecd62413c"
# os.environ["WANDB_DISABLE_SERVICE"]="True"

## uncomment following lines for debugging
# os.environ["WANDB_SILENT"] = "true"
# os.environ["WANDB_MODE"] = "dryrun"

def main():
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    cudnn.enabled = True
    cudnn.benchmark = True
#    torch.backends.cudnn.deterministic = True

    config, writer = init_config("config/so_config.yml", sys.argv)

    wandb.login(timeout=300)
    wandb.init(
                entity="nik1806",
                project='DASS-retraining', 
                name="Run: Source only training - Synthia seq (image)",
                config=config,
                )

    if config.source=='synthia':
        config.num_classes=16
    elif config.source=='synthia_seq': ##!!
        config.num_classes=12
    else:
        config.num_classes=19

    model = init_model(config)
    wandb.watch(model, log='all')

    trainer = Trainer(model, config, writer)

    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    start = datetime.datetime(2020, 1, 22, 23, 00, 0)
    print("wait")
    while datetime.datetime.now() < start:
        time.sleep(1)
    main()
