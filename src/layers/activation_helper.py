import torch
from torch import nn
from torch.nn import functional as F


def gelu(x, inplace=False):
    return F.gelu(x)


def tanh(x, inplace: bool = False):
    return x.tanh_() if inplace else x.tanh()


def sigmoid(x, inplace: bool = False):
    return x.sigmoid_() if inplace else x.sigmoid()


__activation_functions__ = dict(
    relu=F.relu,
    relu6=F.relu6,
    gelu=gelu,
    sigmoid=sigmoid,
    tanh=tanh
)


def get_activation(act_type):
    assert act_type in __activation_functions__.keys()
    return __activation_functions__[act_type]


class Activation(nn.Module):
    def __init__(self, activation, inplace=True):
        super().__init__()
        self.act_fn = get_activation(activation)
        self.act_ = activation
        self.inplace = inplace
    
    def forward(self, x):
        return self.act_fn(x, self.inplace)

    def __repr__(self):
        return f'Activation(activation={self.act_}, inplace={self.inplace})'
