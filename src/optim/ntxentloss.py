"""
NTXentLoss implementation
Eduardo Fonseca's extremely fast implementation, repackaged into a nice nn.Module
Original: https://github.com/edufonseca/uclser20/blob/main/src/utils_train_eval.py Copyrights @ Eduardo Fonseca

Hacked together by authors of paper1517
"""
import torch
from torch import nn


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, anchor, pos):
        batch_size = anchor.size(0)
        device = anchor.device
        assert pos.size(0) == batch_size
        embedding = torch.cat([anchor, pos])

        # cosine similarity matrix
        logits = torch.div(torch.matmul(embedding, embedding.t()), self.temperature)
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        mask = mask.repeat(2, 2)

        logits_mask = (torch.ones_like(mask) - torch.eye(batch_size * 2).to(device))
        mask = mask * logits_mask

        exp_logits_den = torch.log((torch.exp(logits) * logits_mask).sum(1, keepdim=True) + 1e-10)

        exp_logits_pos = torch.log(torch.exp(logits) + 1e-10)
        log_prob = exp_logits_pos - exp_logits_den

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos
        loss = loss.view(2, batch_size).mean()
        return loss
