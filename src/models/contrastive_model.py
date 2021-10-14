import torch
import pickle
from torch import nn
from torch.nn import functional as F
from src.models.model_helper import get_feature_extractor


class Model(nn.Module):
    def __init__(self, opt, add_linear_head=False):
        super(Model, self).__init__()
        self.feature_extractor = get_feature_extractor(opt)

        # the self.feature model has a linear fc layer already
        # SimCLRv2 passes ResNet's AveragePool layer output to the projection head
        # https://github.com/google-research/simclr/blob/3ad6700c1b139ee18e43f73546b7263a710de699/tf2/resnet.py#L490
        # thus we use less layers here
        # PS: apply ReLU after output of feature

        self.fc2 = nn.Linear(opt['fc1_dim'], opt['projection_dim'])
        self.fc3 = nn.Linear(opt['projection_dim'], opt['proj_out_dim'],
                             bias=False)
        if add_linear_head:
            print("Attention: Adding Supervised Linear Head")
            self.linear_head = nn.Linear(opt['fc1_dim'], opt['num_classes'])
        else:
            self.linear_head = None

    def forward(self, x):
        features, _ = self.feature_extractor(x)
        x = F.relu(features, inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = F.normalize(self.fc3(x), dim=1)
        if self.linear_head:
            linear_output = self.linear_head(features.detach())
            return x, linear_output
        else:
            return x, features


def get_pretrained_weights_for_transfer(hparams_path, ckpt_path, load_ckpt=True):
    print("loading pretrained weights...")
    print("\t hparams -> {}".format(hparams_path))
    with open(hparams_path, "rb") as fp:
        hparams = pickle.load(fp)
    print("\t checkpoint -> {}".format(ckpt_path))
    checkpoint_archive = torch.load(ckpt_path)
    # reconstruct model
    model = Model(hparams.cfg['model'])
    if load_ckpt:
        model.load_state_dict(checkpoint_archive['model_state_dict'])

    output_feature_dims = hparams.cfg['model']['fc1_dim']

    # as per SimCLRv2, fine-tuning is done from the first layer of the MLP head
    # since in our model, the first layer of the projection head is in self.feature itself
    # we simply return self.feature
    return model.feature_extractor, output_feature_dims
