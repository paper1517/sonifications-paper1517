import torch
from torch import nn
from torch.nn import functional as F


class ContrastiveAccuracy(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveAccuracy, self).__init__()
        self.temperature = temperature

    def forward(self, anchor, pos):
        batch_size = anchor.size(0)
        device = anchor.device
        assert pos.size(0) == batch_size

        # based on line
        # labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
        # from https://github.com/google-research/simclr/blob/3ad6700c1b139ee18e43f73546b7263a710de699/tf2/objective.py#L73
        labels_contrastive = F.one_hot(torch.arange(batch_size, device=device), num_classes=batch_size*2)

        # based on line
        # logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
        # https://github.com/google-research/simclr/blob/3ad6700c1b139ee18e43f73546b7263a710de699/tf2/objective.py#L80
        logits_contrastive = torch.div(torch.matmul(anchor, pos.t()), self.temperature)

        correct = torch.argmax(labels_contrastive, 1).eq(torch.argmax(logits_contrastive, 1)).sum() / batch_size
        return correct
