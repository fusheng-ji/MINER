import torch
from torch.utils.data import Dataset
from kornia.utils.grid import create_meshgrid, create_meshgrid3d
from einops import repeat
from utils import einops_f
import numpy as np

class MinerDataset(Dataset):
    def __init__(self, out, hparams, active_blocks=None):
        """
        out:  output resized to the current level
                subtracted by the upsampled reconstruction of the previous level
                in finer levels
        active_block:   torch.tensor None to return all blocks,
                        otherwise specify the blocks to take
        """
        self.patch_size = np.prod(hparams.patch_size)

        # split into patches      
        out = einops_f(out, hparams.methods['reshape'][3], hparams)
        self.out = torch.tensor(out) # (n, p, c) number patches channels
        if hparams.task == 'image':
            input = create_meshgrid(hparams.p2, hparams.p1)
        elif hparams.task == 'mesh':
            input = create_meshgrid3d(hparams.p3, hparams.p2, hparams.p1)

        self.input = einops_f(input, hparams.methods['reshape'][7]) 
        
        if active_blocks is not None:
            self.input = repeat(self.inp, '1 p c -> n p c', n = len(self.out))
            self.input = self.input[active_blocks]
            self.out = self.out[active_blocks]

        
    def __len__(self):
       return self.size 
    
    def __getitem__(self, idx: int):
        # each batch contains all blocks with randomly selected cells
        # the cells in each block are of the same position
        return {"input": self.input[:, idx], "out": self.out[:, idx]}
