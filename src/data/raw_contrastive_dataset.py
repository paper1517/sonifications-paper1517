import os
import math
import time
import io
import lmdb
import tqdm
import glob
import numpy as np
import librosa
import torch
import json
import random
import pandas as pd
from torch.utils.data import Dataset
from src.data.raw_waveform_parser import RawAudioParser
from src.data.raw_transforms import NonOverlappingRandomCrop
from src.data.utils import load_audio, load_audio_bytes
from src.data.audioset_labels import audioset_lbl_map


class RawContrastiveDataset(Dataset):
    def __init__(self, manifest_path, audio_config, transform,
                 is_val=False, **kwargs):
        super(RawContrastiveDataset, self).__init__()
        assert audio_config is not None
        assert transform is not None
        assert os.path.exists(manifest_path)
        self.transform = transform
        self.is_val = is_val
        self.sr = None
        self.normalize = None
        self.min_duration = None
        self.view_size = None
        self.parse_audio_config(audio_config)
        df = pd.read_csv(manifest_path)
        self.files = df['files'].values.tolist()
        self.labels = df['labels'].values.tolist()
        self.data_parser = RawAudioParser(normalize_waveform=self.normalize)
        self.length = len(self.files)
        self.view_cropper = NonOverlappingRandomCrop(self.view_size, self.sr, self.min_separation_ms)

    def parse_audio_config(self, audio_config):
        self.sr = int(audio_config.get("sample_rate", "8000"))
        self.normalize = bool(audio_config.get("normalize", False))
        self.min_duration = float(audio_config.get("min_duration", 10.))
        self.min_separation_ms = float(audio_config.get("min_separation_ms", 300))
        if self.is_val:
            self.view_size = float(audio_config.get("val_clip_size", 2.5))
        else:
            self.view_size = float(audio_config.get("random_clip_size", 2.5))
        self.view_size = int(self.view_size * self.sr)

    def __getitem__(self, index):
        # read and parse the file
        audio = load_audio(self.files[index], self.sr, self.min_duration)
        # print("__getitem__ audio shape", audio.shape)
        x, _ = self.data_parser(audio)
        # get views of x
        x_i, x_j = self.view_cropper(x)
        # apply transforms on each
        x_i = self.transform(x_i)
        x_j = self.transform(x_j)
        # return views, along with index
        labels = self.__parse_labels__(self.labels[index])
        return x_i, x_j, index, labels

    def __parse_labels__(self, lbls: str) -> torch.Tensor:
        label_tensor = torch.zeros(len(audioset_lbl_map)).float()
        for lbl in lbls.split(";"):
            label_tensor[audioset_lbl_map[lbl]] = 1

        return label_tensor

    def __len__(self):
        return self.length
