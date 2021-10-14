import torch
from torch import nn
from torch.nn import functional as F


class ScaledStdConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 gamma=1.0, eps=1e-5, gain_init=1.0):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.gain = nn.Parameter(torch.full((self.out_channels, 1, 1), gain_init))
        self.scale = gamma * self.weight[0].numel() ** -0.5
        self.eps = eps

    def get_weight(self):
        # std, mean = torch.std_mean(self.weight, dim=[1, 2], keepdim=True, unbiased=False)
        std = torch.std(self.weight, dim=[1, 2], keepdim=True, unbiased=False)
        mean = torch.mean(self.weight, dim=[1, 2], keepdim=True)
        weight = self.scale * (self.weight - mean) / (std + self.eps)
        return self.gain * weight

    def forward(self, x):
        return F.conv1d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)


class ScaledStdConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros',
                 gamma=1.0, eps=1e-5, gain_init=1.0):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.gain = nn.Parameter(torch.full((self.out_channels, 1, 1), gain_init))
        self.scale = gamma * self.weight[0].numel() ** -0.5
        self.eps = eps
        self.gamma = gamma

    def copy_params_from_conv(self, weight, gain):
        # override weight and gain from corresponding ScaledStdConv1d layer
        self.weight = nn.Parameter(weight)
        self.gain = nn.Parameter(gain)

        # recalculate scale
        self.scale = self.gamma * self.weight[0].numel() ** -0.5

    def get_weight(self):
        # std, mean = torch.std_mean(self.weight, dim=[1, 2], keepdim=True, unbiased=False)
        std = torch.std(self.weight, dim=[1, 2], keepdim=True, unbiased=False)
        mean = torch.mean(self.weight, dim=[1, 2], keepdim=True)
        weight = self.scale * (self.weight - mean) / (std + self.eps)
        return self.gain * weight

    def forward(self, x, output_size=None):
        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size, self.dilation)
        return F.conv_transpose1d(x, self.get_weight(), self.bias, self.stride,
                                  self.padding, output_padding, self.groups,
                                  self.dilation)
