import os
import librosa
import tqdm
import json
import numpy as np
import pickle
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, minmax_scale, maxabs_scale
from src.data.raw_transforms import get_raw_transforms_v2, simple_supervised_transforms
from src.data.raw_dataset import RawWaveformDataset
from torch.utils.data import DataLoader
from src.data.utils import _collate_fn_raw, _collate_fn_raw_multiclass
from src.models.contrastive_model import get_pretrained_weights_for_transfer
from src.models.model_helper import get_feature_extractor
from src.utilities import fourier_analysis


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


def process_vis(deconv_output, input_audio, scaling_type="peak_norm"):
    o = deconv_output.flatten()
    if scaling_type == "minmax":
        scaled = minmax_scale(o.reshape(-1, 1), feature_range=(input_audio.min(), input_audio.max())).flatten()
    elif scaling_type == "peak_norm":
        # essentially max abs scaling to get deconv outputs in -1,1 range
        # followed by scaling by max abs value in input_audio
        scaled = maxabs_scale(o.reshape(-1, 1)).flatten()
        scaled *= np.max(np.abs(input_audio))
    else:
        raise ValueError(f"invalid value {scaling_type} for scaling_type. Should be one of ['minmax', 'peak_norm']")
    return scaled


def save_figure(data, info, path, dpi):
    plt.plot(data)
    plt.title(info)
    plt.savefig(path, dpi=dpi)


def infer_model(net, inp):
    with torch.no_grad():
        pred, output_features, switch_indices = net(inp, True)
    return output_features, switch_indices


def get_supervised_predictions(net, inp, mode='multilabel'):
    with torch.no_grad():
        pred = net(inp)
        if mode == "multilabel":
            y = torch.sigmoid(pred)
        elif mode == "multiclass":
            y = torch.softmax(pred, 1)
        return y


def infer_deconv_model(deconv, inp, output_features, switch_indices, tgt_layer, top_n=3):
    inp_waveform = inp.detach().cpu().squeeze().numpy()
    with torch.no_grad():
        output, idxs = deconv.visualize_layer(inp, output_features, switch_indices, tgt_layer, top_n=top_n)
    processed_output = {}
    for k, v in output.items():
        processed_output[k] = process_vis(v, inp_waveform)
    return processed_output, inp_waveform, idxs, output


def prep_input(inp_array):
    arr = torch.from_numpy(np.ascontiguousarray(inp_array, dtype=np.float64))
    arr = arr.unsqueeze(0).unsqueeze(0).float().cuda()
    return arr


def prep_contrastive_model_and_decoder(exp_dir, last_epoch=100, to_cuda=torch.cuda.is_available()):
    from src.models.contrastive_model import Model
    from src.models.cnn12_decoder import DeconvolutionalDecoder

    hparams_path = os.path.join(exp_dir, "hparams.pickle")
    ckpt = glob.glob(os.path.join(exp_dir, "ckpts", "epoch={:03d}*".format(last_epoch)))
    ckpt_path = ckpt[-1]
    print(f"Loading {ckpt_path}")
    checkpoint = torch.load(ckpt_path)
    with open(hparams_path, "rb") as fp:
        hparams = pickle.load(fp)
    if hparams.cfg['model']['arch'] == "cnn14_like":
        print("replacing arch=cnn14_like with cnn12")
        hparams.cfg['model']['arch'] = "cnn12"
    model = Model(hparams.cfg['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    if to_cuda:
        model = model.cuda()
    model = model.eval()
    net = model.feature_extractor

    deconv = DeconvolutionalDecoder(net)
    if to_cuda:
        deconv = deconv.cuda()
    deconv = deconv.eval()
    output = {
        'full_model': model,
        "feature_extractor": net,
        "deconv_decoder": deconv,
        "hparams": hparams
    }
    return output


def get_supervised_pretrained_model(hparams_path, ckpt_path, load_ckpt=True):
    with open(hparams_path, "rb") as fp:
        hparams = pickle.load(fp)
    if hparams.cfg['model']['arch'] == "cnn14_like":
        print("replacing arch=cnn14_like with cnn12")
        hparams.cfg['model']['arch'] = "cnn12"
    print(f"Loading {ckpt_path}")
    checkpoint = torch.load(ckpt_path)
    model = BaselineModel(hparams.cfg['model'])
    if load_ckpt:
        model.load_state_dict(checkpoint['model_state_dict'])
    return model, hparams.cfg['model']['fc1_dim']


def prep_finetuned_model_and_decoder(exp_dir, last_epoch=50, to_cuda=torch.cuda.is_available()):
    from src.models.contrastive_model import Model
    from src.models.cnn12_decoder import DeconvolutionalDecoder
    hparams_path = os.path.join(exp_dir, "hparams.pickle")
    ckpt = glob.glob(os.path.join(exp_dir, "ckpts", "epoch={:03d}*".format(last_epoch)))
    ckpt_path = ckpt[-1]
    print(f"Loading {ckpt_path}")
    checkpoint = torch.load(ckpt_path)
    with open(hparams_path, "rb") as fp:
        hparams = pickle.load(fp)
    if hparams.cfg['model']['arch'] == "cnn14_like":
        print("replacing arch=cnn14_like with cnn12")
        hparams.cfg['model']['arch'] = "cnn12"
    model = BaselineModel(hparams.cfg['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    if to_cuda:
        model = model.cuda()
    model = model.eval()
    net = model.features

    deconv = DeconvolutionalDecoder(net)
    if to_cuda:
        deconv = deconv.cuda()
    deconv = deconv.eval()
    output = {
        'full_model': model,
        "feature_extractor": net,
        "deconv_decoder": deconv,
        "hparams": hparams
    }
    return output


def get_audioset_eval_samples(meta_dir, audio_config, crop_size=5, cuda=torch.cuda.is_available()):
    eval_df = os.path.join(meta_dir, "sonification_eval.csv")
    lbl_map_path = os.path.join(meta_dir, "lbl_map.json")
    with open(lbl_map_path, "r") as fd:
        lbl_map = json.load(fd)
    inv_lbl_map = {}
    for k, v in lbl_map.items():
        inv_lbl_map[v] = k
    crop_size = int(crop_size * audio_config['sample_rate'])
    val_tfs = simple_supervised_transforms(False, crop_size,
                                           sample_rate=audio_config['sample_rate'])
    val_set = RawWaveformDataset(eval_df,
                                 lbl_map_path,
                                 audio_config, mode='multilabel',
                                 transform=val_tfs, is_val=True, delimiter=";"
                                 )
    val_loader = DataLoader(val_set, sampler=None, num_workers=1,
                            collate_fn=_collate_fn_raw,
                            shuffle=False, batch_size=1,
                            pin_memory=False)
    batches = []
    gt_tensors = []
    gt_string = []
    for batch in tqdm.tqdm(val_loader):
        x, _, y = batch
        if cuda:
            x = x.cuda()
        batches.append(x)
        y = y.detach().cpu().float()
        # print(y)
        y_np = y.numpy().flatten()
        idxs = np.where(y_np == 1)[0]
        lbl_strs = [inv_lbl_map[idx] for idx in idxs]
        gt_tensors.append(y)
        gt_string.append(lbl_strs)

    output = {
        'input_tensors': batches,
        "gt_tensors": gt_tensors,
        "labels": gt_string,
        "lbl_map": lbl_map,
        "inv_lbl_map": inv_lbl_map
    }
    return output


def get_fsdkaggle2018_eval_samples(meta_dir, audio_config, crop_size=5, cuda=torch.cuda.is_available()):
    eval_df = os.path.join(meta_dir, "sonification_eval.csv")
    lbl_map_path = os.path.join(meta_dir, "lbl_map.json")
    with open(lbl_map_path, "r") as fd:
        lbl_map = json.load(fd)
    inv_lbl_map = {}
    for k, v in lbl_map.items():
        inv_lbl_map[v] = k
    crop_size = int(crop_size * audio_config['sample_rate'])
    val_tfs = simple_supervised_transforms(False, crop_size,
                                           sample_rate=audio_config['sample_rate'])
    val_set = RawWaveformDataset(eval_df,
                                 lbl_map_path,
                                 audio_config,
                                 transform=val_tfs, is_val=True,
                                 )
    val_loader = DataLoader(val_set, sampler=None, num_workers=1,
                            collate_fn=_collate_fn_raw_multiclass,
                            shuffle=False, batch_size=1,
                            pin_memory=False)
    batches = []
    gt_tensors = []
    gt_string = []
    for batch in tqdm.tqdm(val_loader):
        x, _, y = batch
        if cuda:
            x = x.cuda()
        batches.append(x)
        y = y.detach().cpu().float()
        # print(y)
        y_np = y.numpy().flatten()
        idxs = np.where(y_np == 1)[0]
        lbl_strs = [inv_lbl_map[idx] for idx in idxs]
        gt_tensors.append(y)
        gt_string.append(lbl_strs)

    output = {
        'input_tensors': batches,
        "gt_tensors": gt_tensors,
        "labels": gt_string,
        "lbl_map": lbl_map,
        "inv_lbl_map": inv_lbl_map
    }
    return output


def make_dataloader(meta_dir, audio_config, crop_size=5, csv_name=None, mode="multilabel", delim=";"):
    lbl_map_path = os.path.join(meta_dir, "lbl_map.json")
    if csv_name:
        eval_csv_path = os.path.join(meta_dir, csv_name)
    else:
        eval_csv_path = os.path.join(meta_dir, "sonification_eval.csv")

    crop_size = int(crop_size * audio_config['sample_rate'])
    val_tfs = simple_supervised_transforms(False, crop_size,
                                           sample_rate=audio_config['sample_rate'])
    val_set = RawWaveformDataset(eval_csv_path,
                                 lbl_map_path,
                                 audio_config, mode=mode,
                                 transform=val_tfs, is_val=True, delimiter=delim
                                 )
    collater = _collate_fn_raw if mode == "multilabel" else _collate_fn_raw_multiclass
    val_loader = DataLoader(val_set, sampler=None, num_workers=1,
                            collate_fn=collater,
                            shuffle=False, batch_size=1,
                            pin_memory=False)
    with open(lbl_map_path, "r") as fd:
        lbl_map = json.load(fd)
    inv_lbl_map = {v: k for k, v in lbl_map.items()}

    return val_loader, val_set, lbl_map, inv_lbl_map


def get_spec(x):
    _, _, data_spec, _ = fourier_analysis.perform_stft(x, noverlap_ms=0.01, nfft=512, boundary="zeros")
    spec = np.abs(data_spec) ** 2
    spec = librosa.amplitude_to_db(spec)
    return spec


def model_helper(exp_dir, is_contrastive, epoch_index=None):
    if is_contrastive:
        if epoch_index is None:
            res = prep_contrastive_model_and_decoder(exp_dir)
        else:
            res = prep_contrastive_model_and_decoder(exp_dir, last_epoch=epoch_index)
    else:
        if epoch_index is None:
            res = prep_finetuned_model_and_decoder(exp_dir, last_epoch=50)
        else:
            res = prep_finetuned_model_and_decoder(exp_dir, last_epoch=epoch_index)
    model, net, deconv, hparams = res['full_model'], res['feature_extractor'], res['deconv_decoder'], res['hparams']
    return model, net, deconv, hparams
