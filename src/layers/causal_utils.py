"""
implementation from https://github.com/pytorch/pytorch/issues/1333#issuecomment-400338207
"""
import torch
from torch import nn
from torch.nn import functional as F
from src.layers.scaled_std_conv import ScaledStdConv1d, ScaledStdConvTranspose1d


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        if isinstance(kernel_size, tuple):
            self.__padding = (kernel_size[0] - 1) * dilation
        else:

            self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result


class CausalConv1d_v2(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        if isinstance(kernel_size, tuple):
            self.__padding = (kernel_size[0] - 1) * dilation
        else:

            self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d_v2, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)
        # self.replication_pad = nn.ReplicationPad1d((self.__padding, 0))

    def forward(self, x):
        # result = super(CausalConv1d, self).forward(input)
        # if self.__padding != 0:
        #     return result[:, :, :-self.__padding]
        # return result
        if self.__padding != 0:
            x = torch.nn.functional.pad(x, (self.__padding, 0))
        # x = self.replication_pad(x)
        x = super().forward(x)
        return x


class ScaledStdCausalConv1d(ScaledStdConv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True, padding_mode='zeros',
                 gamma=1.0, eps=1e-5, gain_init=1.0):
        if isinstance(kernel_size, tuple):
            self.__padding = (kernel_size[0] - 1) * dilation
        else:

            self.__padding = (kernel_size - 1) * dilation
        super(ScaledStdCausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            gamma=gamma, eps=eps, gain_init=gain_init
        )
        # self.replication_pad = nn.ReplicationPad1d((self.__padding, 0))

    def forward(self, x):
        # x = self.replication_pad(x)
        if self.__padding != 0:
            x = torch.nn.functional.pad(x, (self.__padding, 0))
        x = super().forward(x)
        return x


class ScaledStdCausalConvTranspose1d(ScaledStdCausalConv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=False, padding_mode='zeros',
                 gamma=1.0, eps=1e-5, gain_init=1.0):
        if isinstance(kernel_size, tuple):
            self.__padding = (kernel_size[0] - 1) * dilation
        else:
            self.__padding = (kernel_size - 1) * dilation
        super(ScaledStdCausalConvTranspose1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.gamma = gamma

        # self.replication_pad = nn.ReplicationPad1d((self.__padding, 0))

    def copy_params_from_conv(self, weight, gain):

        # Causal Convolution maintains size of the input
        # one can just invert the filter and apply regular causal convolution as is
        # inverting filter -> flip it, and transpose on the weights axis
        # self.weight = nn.Parameter(torch.flip(weight.data, [0, 1]).transpose(0, 1))

        self.weight = nn.Parameter(weight.data)
        self.gain = nn.Parameter(gain)
        # recalculate scale
        self.scale = self.gamma * weight[0].numel() ** -0.5

    def get_weight(self):
        # std, mean = torch.std_mean(self.weight, dim=[1, 2], keepdim=True, unbiased=False)
        std = torch.std(self.weight, dim=[1, 2], keepdim=True, unbiased=False)
        mean = torch.mean(self.weight, dim=[1, 2], keepdim=True)
        weight = self.scale * (self.weight - mean) / (std + self.eps)

        return self.gain * weight

    def forward(self, x):
        return F.conv1d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, x, output_size=None):
        # Causal Convolution maintains size of the input
        # one can just invert the filter and apply regular causal convolution as is
        # inverting filter -> flip it, and transpose on the weights axis
        # couldn't do it earlier: mismatch between pretrained self.gain shape
        # so doing it here, before conv is actually applied

        if self.__padding != 0:
            x = torch.nn.functional.pad(x, (self.__padding, 0))
        weight = self.get_weight()
        return F.conv1d(x, weight.transpose(0, 1), self.bias,
                        self.stride, self.padding, self.dilation, self.groups)
