import torch
from torch import nn
from torch.distributions.gamma import Gamma
import numpy as np
from einops import rearrange


@torch.jit.script
def gaussian_activation(x, a):
    return torch.exp(-x**2/(2 * a**2))

@torch.jit.script
def scaled_sin_activation(x, a):
    return torch.sin(x * a)
# encoding matrices for positional encoding
E_2d = torch.eye(2)
# defined in mip-nerf360 https://arxiv.org/pdf/2111.12077.pdf page11

E_3d = torch.FloatTensor([
    0.8506508, 0, 0.5257311,
    0.809017, 0.5, 0.309017,
    0.5257311, 0.8506508, 0,
    1, 0, 0,
    0.809017, 0.5, -0.309017,
    0.8506508, 0, -0.5257311,
    0.309017, 0.809017, -0.5,
    0, 0.5257311, -0.8506508,
    0.5, 0.309017, -0.809017,
    0, 1, 0,
    -0.5257311, 0.8506508, 0,
    -0.309017, 0.809017, -0.5,
    0, 0.5257311, 0.8506508,
    -0.309017, 0.809017, 0.5,
    0.309017, 0.809017, 0.5,
    0.5, 0.309017, 0.809017,
    0.5, -0.309017, 0.809017,
    0, 0, 1,
    -0.5, 0.309017, 0.809017,
    -0.809017, 0.5, 0.309017,
    -0.809017, 0.5, -0.309017]).view(21, 3).T

class PE(nn.Module):
    """
    positional encoding
    """
    def __init__(self, p):
        """
        P: (d, F) encoding matrix
        """
        super().__init__()
        self.register_buffer("P", P)

    @property
    def out_dim(self):
        return self.P.shape[1]*2
    def forward(self, x):
        """
        x: (n_blocks, B, d)
        """
        x_ = x @ self.P # (n_blocks, B, F)
        return torch.cat([torch.sin(x_), torch.cos(x_)], -1) # (n_blocks, B, 2 * F)

class MLPCollection(nn.Module):
    """
    MLPCollection consists of all the MLPs of a certain scale.
    All MLPs are inferred at the same time without using for loop
    """
    def __init__(self, n_blocks, n_in, n_out,
                 n_layers, n_hidden, final_act,
                 a=0.1):
        super.__init__()

        self.n_layers = n_layers
        self.final_act = final_act
        for i in range(n_layers):
            if i == 0: # first layer
                wi = nn.Parameter(torch.empty(n_blocks, n_in, n_hidden))
                bi = nn.Parameter(torch.empty(n_blocks, 1, n_hidden))
                ai = nn.Parameter(a * torch.ones(n_blocks, 1, 1))
            elif i < n_layers - 1: # middle layers
                wi = nn.Parameter(torch.empty(n_blocks, n_hidden, n_hidden))
                bi = nn.Parameter(torch.empty(n_blocks, 1, n_hidden))
                ai = nn.Parameter(a * torch.ones(n_blocks, 1, 1))
            else: # last layer
                wi = nn.Parameter(torch.empty(n_blocks, n_hidden, n_out))
                bi = nn.Parameter(torch.empty(n_blocks, 1, n_out))
                if final_act == 'sigmoid':
                    ai = nn.Sigmoid()
                elif final_act == 'sin':
                    ai = nn.Parameter(a * torch.ones(n_blocks, 1, 1))

        # layer initialization
        if i == 0:
            """
            nn.init.uniform_(tensor, a, b): 
            Fills the input Tensor with values drawn from the uniform distribution U(a,b).
            """
            nn.init.uniform_(wi, -1/(n_in**0.5), 1/(n_in**0.5))
            nn.init.uniform_(bi, -1/(n_in ** 0.5), 1/(n_in ** 0.5))
        else:
            nn.init.uniform_(wi, -1/(n_hidden ** 0.5), 1/(n_hidden ** 0.5))
            nn.init.uniform_(bi, -1/(n_hidden ** 0.5), 1/(n_hidden ** 0.5))
        """
        setattr(): sets the value of the specified attribute of the specified object
        getattr(): returns the value of the specified attribute from the specified object
        """
        setattr(self, f'w{i}', wi)
        setattr(self, f'b{i}', bi)
        setattr(self, f'a{i}', ai)

    def forward(self, x, b_chunks=16384, to_cpu=False, **kwargs):
        """
        Inputs:
            x: (n_blocks, B, n_in)
            b_chunks: int, @x is split into chunks of at most @b_chunks blocks

        Outputs:
            (n_blocks, B, n_out)
        """
        out = []
        for c in range(0, len(x), b_chunks):
            x_ = x[c: c+b_chunks]
            if 'pe' in kwargs:
                x_ = kwargs['pe'](x_)
            for i in range(self.n_layers):
                wi = getattr(self, f'w{i}')[c: c+b_chunks]
                bi = getattr(self, f'b{i}')[c: c+b_chunks]
                ai = getattr(self, f'a{i}')
                if i < self.n_layers-1:
                    x_ = gaussian_activation(x_@wi+bi, ai[c: c+b_chunks]) # a@b: matrix multiplication
                else:
                    if self.final_act == 'sigmoid':
                        x_ = ai(x_@wi+bi)
                    elif self.final_act == 'sin':
                        x_ = scaled_sin_activation(x_@wi+bi, ai[c: c+b_chunks])
            if to_cpu:
                x_ = x_.cpu()
            out += [x_]
        return torch.cat(out)

# https://github.com/boschresearch/multiplicative-filter-networks/blob/main/mfn/mfn.py
class MLPCollection_Gabor(nn.Module):
    """
    We now present two instantiations of th MFN,
    using sinusoids or a Gabor wavelet as the filter g;
    we call these two networks the FOURIERNET and GABORNET respectively.

    A well-known deficiency of the pure Fourier bases is that they have global support,
    and thus may have difficulty representing more local features.
    A common alternative to these bases is the use of Gabor filter
    to capture both a frequency and spatial locality component.
    Specifically, we consider a Gabor filter of the form
        torch.sin(xi@fwi+fbi) * torch.exp(-D/2*rearrange(fgammai, 'n h -> n 1 h'))
    The output of a Gabor Network is given by a linear combination of Gabor bases
    """
    def __init__(self, n_blocks, n_in, n_out,
                 n_layers, n_hidden, final_act,
                 a=0.1, weight_scale=256.0, alpha=6.0, beta=1.0):
        super().__init__()

        self.n_layers = n_layers
        self.final_act = final_act
        weight_scale /= (n_layers-1)**0.5
        for i in range(n_layers):
            if i < n_layers-1:
                fwi = nn.Parameter(torch.empty(n_blocks, n_in, n_hidden))
                fbi = nn.Parameter(torch.empty(n_blocks, 1, n_hidden))
                fmui = nn.Parameter(2*(torch.rand(n_blocks, 1, n_hidden, n_in)-0.5))
                fgammai = nn.Parameter(Gamma(alpha/(n_layers-1), beta).sample((n_blocks, n_hidden)))
                nn.init.uniform_(fwi, -1/(n_in**0.5), 1/(n_in**0.5))
                nn.init.uniform_(fbi, -np.pi, np.pi)
                fwi.data *= weight_scale*rearrange(fgammai, 'n d -> n 1 d')**0.5

                setattr(self, f'fw{i}', fwi)
                setattr(self, f'fb{i}', fbi)
                setattr(self, f'fmu{i}', fmui)
                setattr(self, f'fgamma{i}', fgammai)
            if i > 0:
                outdim = n_hidden if i < n_layers-1 else n_out
                wi = nn.Parameter(torch.empty(n_blocks, n_hidden, outdim))
                bi = nn.Parameter(torch.empty(n_blocks, 1, outdim))
                nn.init.uniform_(wi, -1/(n_hidden**0.5), 1/(n_hidden**0.5))
                nn.init.uniform_(bi, -1/(n_hidden**0.5), 1/(n_hidden**0.5))

                setattr(self, f'w{i}', wi)
                setattr(self, f'b{i}', bi)
                if i == n_layers-1:
                    if final_act == 'sigmoid':
                        ai = nn.Sigmoid()
                    elif final_act == 'sin':
                        ai = nn.Parameter(a*torch.ones(n_blocks, 1, 1))
                    setattr(self, f'a{i}', ai)

    def forward(self, x, b_chunks=16384, to_cpu=False, **kwargs):
        """
        Inputs:
            x: (n_blocks, B, n_in)
            b_chunks: int, @x is split into chunks of at most @b_chunks blocks

        Outputs:
            (n_blocks, B, n_out)
        """
        out = []
        for c in range(0, len(x), b_chunks):
            x_ = xi = x[c: c+b_chunks]
            for i in range(self.n_layers):
                if i < self.n_layers-1:
                    fwi = getattr(self, f'fw{i}')[c: c+b_chunks]
                    fbi = getattr(self, f'fb{i}')[c: c+b_chunks]
                    fmui = getattr(self, f'fmu{i}')[c: c+b_chunks]
                    fgammai = getattr(self, f'fgamma{i}')[c: c+b_chunks]
                    D = torch.norm(rearrange(xi, 'n b d -> n b 1 d') - fmui, dim=-1)**2
                if i > 0:
                    wi = getattr(self, f'w{i}')[c: c+b_chunks]
                    bi = getattr(self, f'b{i}')[c: c+b_chunks]
                if i==0:
                    x_ = torch.sin(xi@fwi+fbi) * \
                         torch.exp(- 0.5 * D * rearrange(fgammai, 'n h -> n 1 h'))
                elif i < self.n_layers-1:
                    x_ = (x_@wi+bi) * \
                         torch.sin(xi@fwi+fbi) * \
                         torch.exp(- 0.5 * D * rearrange(fgammai, 'n h -> n 1 h'))
                else: # last layer
                    ai = getattr(self, f'a{i}')
                    if self.final_act == 'sigmoid':
                        x_ = ai(x_@wi+bi)
                    elif self.final_act == 'sin':
                        x_ = scaled_sin_activation(x_@wi+bi, ai[c: c+b_chunks])
            if to_cpu: x_ = x_.cpu()
            out += [x_]
        return torch.cat(out)