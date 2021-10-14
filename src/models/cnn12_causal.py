"""
Standard 1D-Causal Convolutional encoder
"""

import torch
from torch import nn
from collections import OrderedDict
from src.models.fp32_group_norm import Fp32GroupNorm
from src.layers.scaled_std_conv import ScaledStdConv1d
from src.layers.gamma_act import GammaAct, act_with_gamma
from src.layers import activation_helper
from src.layers.statistics_pooling import StatisticsPooling
from src.layers.causal_utils import CausalConv1d_v2


class Encoder(nn.Module):
    def __init__(self, fc1_dim=1000,
                 activation="relu", norm=None, filter_coeff=1.,
                 aggregation_pool="average"):
        super(Encoder, self).__init__()
        self.norm = norm
        # self.activation_fn = activation_helper.Activation(activation)
        self.activation = activation

        self.kernel_sizes = [7, 7, 5, 5, 5, 3, 3, 3, 3, 3, 3]
        self.strides = [2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1]
        self.filter_coeff = filter_coeff
        self.fc1_dim = fc1_dim
        self.aggregation_pool = aggregation_pool

        self.features = nn.Sequential(
            OrderedDict([
                ("conv1", CausalConv1d_v2(1, int(64 * self.filter_coeff),
                                    kernel_size=(self.kernel_sizes[0],), stride=(1,))),
                # ("bn1", Fp32GroupNorm(1, int(64 * self.filter_coeff))),
                # ("bn1", nn.BatchNorm1d(int(128 * self.filter_coeff))),
                ("act1", activation_helper.Activation(activation)),
                ("mp1", nn.MaxPool1d(self.strides[0], self.strides[0], return_indices=True)),

                ("conv2", CausalConv1d_v2(int(self.filter_coeff * 64), int(self.filter_coeff * 64),

                                    kernel_size=(self.kernel_sizes[1],), stride=(1,))),
                # ("bn2", Fp32GroupNorm(1, int(128 * self.filter_coeff))),
                # ("bn2", nn.BatchNorm1d(int(256 * self.filter_coeff))),
                ("act2", activation_helper.Activation(activation)),
                ("mp2", nn.MaxPool1d(self.strides[1], self.strides[1], return_indices=True)),

                ("conv3", CausalConv1d_v2(int(self.filter_coeff * 64), int(self.filter_coeff * 128),

                                    kernel_size=(self.kernel_sizes[2],), stride=(1,))),
                # ("bn3", Fp32GroupNorm(1, int(128 * self.filter_coeff))),
                # ("bn3", nn.BatchNorm1d(int(512 * self.filter_coeff))),
                ("act3", activation_helper.Activation(activation)),
                ("mp3", nn.MaxPool1d(self.strides[2], self.strides[2], return_indices=True)),

                ("conv4", CausalConv1d_v2(int(self.filter_coeff * 128), int(self.filter_coeff * 128),

                                    kernel_size=(self.kernel_sizes[3],), stride=(1,))),
                # ("bn4", Fp32GroupNorm(1, int(256 * self.filter_coeff))),
                # ("bn4", nn.BatchNorm1d(int(512 * self.filter_coeff))),
                ("act4", activation_helper.Activation(activation)),
                ("mp4", nn.MaxPool1d(self.strides[3], self.strides[3], return_indices=True)),

                ("conv5", CausalConv1d_v2(int(self.filter_coeff * 128), int(self.filter_coeff * 256),

                                    kernel_size=(self.kernel_sizes[4],), stride=(1,))),
                ("act5", activation_helper.Activation(activation)),
                ("mp5", nn.MaxPool1d(self.strides[4], self.strides[4], return_indices=True)),

                ("conv6", CausalConv1d_v2(int(self.filter_coeff * 256), int(self.filter_coeff * 256),

                                    kernel_size=(self.kernel_sizes[5],), stride=(1,))),
                ("act6", activation_helper.Activation(activation)),
                ("mp6", nn.MaxPool1d(self.strides[5], self.strides[5], return_indices=True)),

                ("conv7", CausalConv1d_v2(int(self.filter_coeff * 256), int(self.filter_coeff * 512),
                                    kernel_size=(self.kernel_sizes[6],), stride=(1,))),
                ("act7", activation_helper.Activation(activation)),
                ("mp7", nn.MaxPool1d(self.strides[6], self.strides[6], return_indices=True)),

                ("conv8", CausalConv1d_v2(int(self.filter_coeff * 512), int(self.filter_coeff * 512),

                                    kernel_size=(self.kernel_sizes[7],), stride=(1,))),
                ("act8", activation_helper.Activation(activation)),
                ("mp8", nn.MaxPool1d(self.strides[7], self.strides[7], return_indices=True)),

                ("conv9", CausalConv1d_v2(int(self.filter_coeff * 512), int(self.filter_coeff * 1024),

                                    kernel_size=(self.kernel_sizes[8],), stride=(1,))),
                ("act9", activation_helper.Activation(activation)),
                # ("mp9", nn.MaxPool1d(self.strides[8], self.strides[8], return_indices=True)),

                ("conv10", CausalConv1d_v2(int(self.filter_coeff * 1024), int(self.filter_coeff * 1024),
                                    kernel_size=(self.kernel_sizes[9],), stride=(1,))),
                ("act10", activation_helper.Activation(activation)),
                # ("mp10", nn.MaxPool1d(self.strides[9], self.strides[9], return_indices=True)),

                ("conv11", CausalConv1d_v2(int(self.filter_coeff * 1024), int(self.filter_coeff * 2048),
                                    kernel_size=(self.kernel_sizes[10],), stride=(1,))),
                ("act11", activation_helper.Activation(activation)),
            ])
        )
        num_output_filters = int(self.filter_coeff * 2048)
        if self.aggregation_pool == "statistics":
            self.agg = StatisticsPooling()
            num_output_filters = num_output_filters * 2

        self.fc1 = nn.Sequential(
            OrderedDict([
                ("drop10", nn.Dropout(0.4)),
                ("fc10", nn.Linear(num_output_filters, self.fc1_dim)),
            ])
        )
        for m in self.features.modules():
            if isinstance(m, CausalConv1d_v2):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0., .01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # self.output_features = [0] * len(self.features)
        self.sizes = dict()

    def forward(self, x, save_features=False):
        output_features = {}
        switch_indices = {}
        for name, layer in self.features.named_children():
            if isinstance(layer, nn.MaxPool1d):
                x, indices = layer(x)
                if save_features:
                    switch_indices[name] = indices
            else:
                x = layer(x)
            if save_features:
                output_features[name] = x
            # print(name, x.shape)

        # print(x.shape)
        if self.aggregation_pool == "statistics":
            x = self.agg(x)
        else:
            x = x.mean(2)
        # x = x.flatten()
        x = self.fc1(x)

        # the following is required for compatibility
        # this way, no changes are needed in any other files
        if save_features:
            return x, output_features, switch_indices
        else:
            return x, output_features
