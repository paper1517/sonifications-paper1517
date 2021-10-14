import torch
import os
from torch import nn


def get_feature_extractor(opt):
    pretrained = opt.get("pretrained", "")
    pretrained_fc = opt.get("pretrained_fc", None)
    if os.path.isfile(pretrained) and pretrained_fc > 2 and type(pretrained_fc) == int:
        pretrained_flag = True
        fc1_dim = pretrained_fc
        ckpt = torch.load(pretrained)
        print("pretrained model {} with {} classes found.".format(pretrained, pretrained_fc))
    else:
        pretrained_flag = False
        fc1_dim = opt['fc1_dim']

    if opt['arch'] == "cnn12":
        coeff = opt.get('scale_coefficient', 1.0)
        act = opt.get("activation", "relu")
        aggregation = opt.get("aggregation", "avgpool")
        from src.models.cnn12 import Encoder
        model = Encoder(fc1_dim=fc1_dim, filter_coeff=coeff, activation=act,
                        aggregation_pool=aggregation)
    elif opt['arch'] == "cnn12_causal":
        coeff = opt.get('scale_coefficient', 1.0)
        act = opt.get("activation", "relu")
        aggregation = opt.get("aggregation", "avgpool")
        from src.models.cnn12_causal import Encoder
        model = Encoder(fc1_dim=fc1_dim, filter_coeff=coeff, activation=act,
                        aggregation_pool=aggregation)
    else:
        raise ValueError("Unsupported value {} for opt['arch']".format(opt['arch']))
    if pretrained_flag:
        if 'standard' in opt['arch']:
            fc_in = model.classifier.fc10.in_features
            print("pretrained loading: ", model.load_state_dict(ckpt))
            model.classifier.fc10 = nn.Linear(fc_in, opt['fc1_dim'])
        if "wav2letter_classifier" in opt['arch']:
            fc_in = model.classifier.fc10.in_features
            print("pretrained loading: ", model.load_state_dict(ckpt))
            model.classifier.fc10 = nn.Linear(fc_in, opt['fc1_dim'])
    return model
