import torch
from torch import nn


class StatisticsPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-5

    def forward(self, x):
        """
        x is a tensor with frame-wise features
        of shape (batch_size, num_filters, seq_length)
        """
        mean = x.mean(dim=2)
        std = x.std(dim=2)
        gnoise = self._get_gauss_noise(mean.size(), device=mean.device)
        gnoise = gnoise
        mean += gnoise
        std = std + self.eps
        pooled_stats = torch.cat((mean, std), dim=1)
        return pooled_stats

    def _get_gauss_noise(self, shape_of_tensor, device="cpu"):
        """Returns a tensor of epsilon Gaussian noise.

        Arguments
        ---------
        shape_of_tensor : tensor
            It represents the size of tensor for generating Gaussian noise.
        """
        gnoise = torch.randn(shape_of_tensor, device=device)
        gnoise -= torch.min(gnoise)
        gnoise /= torch.max(gnoise)
        gnoise = self.eps * ((1 - 9) * gnoise + 9)
        return gnoise
