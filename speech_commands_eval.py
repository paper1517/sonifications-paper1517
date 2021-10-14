import pandas as pd
import torch
import glob
from torch import nn
import json
import pickle
import torch
import os
import sys
from src.models.contrastive_model import Model
from src.utilities.config_parser import get_config
import argparse
from src.models.contrastive_model import get_pretrained_weights_for_transfer
from src.data.raw_transforms import get_raw_transforms_v2, simple_supervised_transforms, PadToSize, PeakNormalization, Compose
from src.data.raw_dataset import RawWaveformDataset
from torch.utils.data import DataLoader
from src.models.model_helper import get_feature_extractor
from src.data.utils import _collate_fn_raw, _collate_fn_raw_multiclass
import tqdm
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score
from torch.nn import functional as F
torch.multiprocessing.set_sharing_strategy('file_system')


class BaselineModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        num_classes = cfg['num_classes']
        self.features = get_feature_extractor(cfg)
        self.fc = nn.Linear(cfg['proj_out_dim'], num_classes)

    def forward(self, x):
        x, _ = self.features(x)
        x = F.relu(x, inplace=True)
        x = self.fc(x)
        return x


class FinetunedModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        pretrained_hparams_path = cfg['pretrained_hparams_path']
        pretrained_ckpt_path = cfg['pretrained_ckpt_path']
        num_classes = cfg['num_classes']
        self.features, output_dims = get_pretrained_weights_for_transfer(pretrained_hparams_path,
                                                                         pretrained_ckpt_path)
        self.fc = nn.Linear(output_dims, num_classes)

    def forward(self, x):
        x, _ = self.features(x)
        x = F.relu(x, inplace=True)
        x = self.fc(x)
        return x


def calculate_mAP(preds, gts, mode="weighted"):
    preds = torch.cat(preds, 0).numpy()
    gts = torch.cat(gts, 0).numpy()
    map_value = average_precision_score(gts, preds, average=mode)
    return map_value


parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", type=str)
parser.add_argument("--contrastive_pretrained", action="store_true")
parser.add_argument('--results_file', type=str)


def get_val_acc(x):
    x = x.split("/")[-1]
    x = x.replace(".pth","")
    x = x.split("val_acc=")[-1]
    return float(x)


if __name__ == '__main__':
    args = parser.parse_args()
    hparams_path = os.path.join(args.exp_dir, "hparams.pickle")
    ckpts = sorted(glob.glob(os.path.join(args.exp_dir, "ckpts", "*")), key=get_val_acc)
    print(ckpts)
    # metrics_path = os.path.join(args.exp_dir, "sorted_metrics.csv")
    # filename = pd.read_csv(metrics_path).iloc[0]['filename']
    # print(filename)
    # filename = filename.split("/")[-1]
    ckpt_path = ckpts[-1]
    print(ckpt_path)
    res_file = os.path.join(args.exp_dir, "results.txt")
    if os.path.exists(res_file):
        print("exists...")
        exit()
    checkpoint = torch.load(ckpt_path)
    with open(hparams_path, "rb") as fp:
        hparams = pickle.load(fp)
    if args.contrastive_pretrained:
        model = FinetunedModel(hparams.cfg['model'])
    else:
        print("making baseline model")
        model = BaselineModel(hparams.cfg['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda().eval()
    # val_tfs = get_raw_transforms_v2(False, 8000*10, center_crop_val=False)
    ac = hparams.cfg['audio_config']
    print(ac)
    val_clip_size = int(ac['val_clip_size'] * ac['sample_rate'])
    val_tfs = Compose([
        PadToSize(val_clip_size, 'wrap'),
        PeakNormalization(sr=ac['sample_rate'])
    ])
    # val_tfs = simple_supervised_transforms(False, val_clip_size,
    #                                        sample_rate=ac['sample_rate'])
    val_set = RawWaveformDataset("/media/user/nvme/datasets/speech_commands_8000/meta/test.csv",
                                 "/media/user/nvme/datasets/speech_commands_8000/meta/lbl_map.json",
                                 hparams.cfg['audio_config'], mode='multiclass',
                                 transform=val_tfs, is_val=True
                                 )

    with open("/media/user/nvme/datasets/speech_commands_8000/meta/lbl_map.json", "r") as fd:
        lbl_map = json.load(fd)

    inv_map = {v: k for k, v in lbl_map.items()}

    val_loader = DataLoader(val_set, sampler=None, num_workers=4,
                            collate_fn=_collate_fn_raw_multiclass,
                            shuffle=False, batch_size=1,
                            pin_memory=False)

    val_preds = []
    val_gts = []
    for batch in tqdm.tqdm(val_loader):
        x, _, y = batch
        x = x.cuda()
        with torch.no_grad():
            pred = model(x)
            pred = torch.argmax(pred, 1).detach().item()

        val_preds.append(pred)
        val_gts.append(y.detach().cpu().float().item())

    # macro_mAP = calculate_mAP(val_preds, val_gts, mode="macro")
    # weighted_mAP = calculate_mAP(val_preds, val_gts, mode="weighted")
    # print("macro_mAP: {} | weighted_mAP: {}".format(macro_mAP, weighted_mAP))
    acc = accuracy_score(np.asarray(val_gts), np.asarray(val_preds))
    print("Accuracy: {}".format(acc))
    fname = ckpt_path.split("/")[-3]
    ckpt_ext = "/".join(ckpt_path.split("/")[-3:])
    with open(res_file, "w") as fd:
        fd.writelines("model, acc, ckpt_ext\n")
        fd.writelines("{},{},{}\n".format(fname, acc, ckpt_ext))

