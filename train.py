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

from pytorch_lightning import LightningMoudle, Trainer
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar

calss MINER(LightningModule):
    def __init__(self, hparms):

