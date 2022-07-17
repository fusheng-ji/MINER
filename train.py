import torch
from torch import nn
import torch.nn.functional as F
from einops import reduce, repeat
from PIL import Image
Image.MAX_IMAGE_PIXELS = None # enable reading large image
import numpy as np
import copy
import os
import warnings
wangrings.filterwarnings("ignore")

from opt import get_opts

# datasets
from dataset import MinerDataset
from torch.utils.data import DataLoader

# models
from models import E_2d, E_3d, PE, MLPCollection, MLPCollection_Gabor
from utils import methods, einops_f

# metrics
from metrics import mse, psnr, iou

# optimizer
from torch.optim import Adam, RAdam
from torch.optim.lr_scheduler import CosineAnnealingLR

from pytorch_lightning import LightningMoudle, Trainer
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar

# log
import wandb

class MINER(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.first_val = True
        self.automatic_optimization = False

        if hparams.task == 'image':
            n_in = 2 # uv
            n_out = 3 # rgb
        elif hparams.task == 'mesh':
            n_in = 3 # xyz
            n_out = 1 # occupancy

        if hparams.use_pe:
            if hparams.task== 'image': E = E_2d
            elif hparams.task == 'mesh': E = E_3d
            P = torch.cat([E * 2 ** i for i in range(hparams.n_freq)], 1)
            self.pe = PE(P)
            n_in = self.pe.out_dim

        # create two copies of the same network
        """
        the network used in training
        """
        if hparams.arch == 'mlp':
            self.optim = RAdam
            net = MLPCollection
        elif hparams.arch == 'gabor':
            self.optim = Adam
            net = MLPCollection_Gabor
        self.mlpcollection_train = net(n_blocks=hparams.n_blocks,
                                  n_in=n_in, n_out=n_out,
                                  n_layers=haparams.n_layers,
                                  n_hidden=hparams.n_hidden,
                                  final_act=hparams.final_act,
                                  a=hparams.a)
        """
        the network used in validation, updated by the trained network
        """
        self.mlpcollection_val = copy.deepcopy(self.mlpcollection_train)
        for p in self.mlpcollection_val.parameters():
            p.requires_grad = False

        self.register_buffer('training_blocks',
                             torch.ones(hparams.n_blocks, dtype=torch.bool))

    def call(self, model, x, b_chunks):
        kwargs = {'to_cpu': not self.mlpcollection_val.training}
        if hparams.use_pe: kwargs['pe'] = self.pe
        out = model(x, b_chunks, **kwargs)
        if hparams.level <= hparams.n_scales-2