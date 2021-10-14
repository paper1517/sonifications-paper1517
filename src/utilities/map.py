import torch
import numpy as np
from sklearn.metrics import average_precision_score


def calculate_mAP(preds, gts, mixup=False, mode="macro"):
    preds = torch.cat(preds, 0).numpy()
    gts = torch.cat(gts, 0).numpy()
    if mixup:
        gts[gts >= 0.5] = 1
        gts[gts < 0.5] = 0
    map_value = average_precision_score(gts, preds, average=mode)
    return map_value
