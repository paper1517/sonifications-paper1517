import torch
import numpy as np
import torch
from torch import nn
from collections import OrderedDict
from src.models.fp32_group_norm import Fp32GroupNorm
from src.layers.scaled_std_conv import ScaledStdConv1d
from src.layers.causal_utils import ScaledStdCausalConv1d
from src.layers.scaled_std_conv import ScaledStdConvTranspose1d
from src.layers.causal_utils import ScaledStdCausalConvTranspose1d
from src.layers.gamma_act import GammaAct, act_with_gamma
from src.layers import activation_helper


class DeconvolutionalDecoder(nn.Module):
    def __init__(self, encoder, causal=False):
        super(DeconvolutionalDecoder, self).__init__()
        activation = encoder.activation
        self.causal = causal
        encoder_state_dict = encoder.state_dict()

        self.kernel_sizes = encoder.kernel_sizes
        self.strides = encoder.strides
        self.filter_coeff = encoder.filter_coeff

        self.deconv_act11 = activation_helper.Activation(activation)
        self.deconv_conv11 = nn.ConvTranspose1d(int(self.filter_coeff * 2048), int(self.filter_coeff * 1024),
                                                kernel_size=(self.kernel_sizes[10],),
                                                stride=(1,), bias=False)
        self.deconv_conv11.weight = nn.Parameter(encoder_state_dict['features.conv11.weight'])

        self.deconv_act10 = activation_helper.Activation(activation)
        self.deconv_conv10 = nn.ConvTranspose1d(int(self.filter_coeff * 1024), int(self.filter_coeff * 1024),
                                       kernel_size=(self.kernel_sizes[9],),
                                       stride=(1,), bias=False)
        self.deconv_conv10.weight = nn.Parameter(encoder_state_dict['features.conv10.weight'])

        self.deconv_act9 = activation_helper.Activation(activation)
        self.deconv_conv9 = nn.ConvTranspose1d(int(self.filter_coeff * 1024), int(self.filter_coeff * 512),
                                        kernel_size=(self.kernel_sizes[8],),
                                       stride=(1,), bias=False)
        self.deconv_conv9.weight = nn.Parameter(encoder_state_dict['features.conv9.weight'])

        self.deconv_mp8 = nn.MaxUnpool1d(self.strides[7], self.strides[7])
        self.deconv_act8 = activation_helper.Activation(activation)
        self.deconv_conv8 = nn.ConvTranspose1d(int(self.filter_coeff * 512), int(self.filter_coeff * 512),
                                        kernel_size=(self.kernel_sizes[7],),
                                       stride=(1,), bias=False)
        self.deconv_conv8.weight = nn.Parameter(encoder_state_dict['features.conv8.weight'])

        self.deconv_mp7 = nn.MaxUnpool1d(self.strides[6], self.strides[6])
        self.deconv_act7 = activation_helper.Activation(activation)
        self.deconv_conv7 = nn.ConvTranspose1d(int(self.filter_coeff * 512), int(self.filter_coeff * 256),
                                               kernel_size=(self.kernel_sizes[6],),
                                               stride=(1,), bias=False)
        self.deconv_conv7.weight = nn.Parameter(encoder_state_dict['features.conv7.weight'])

        self.deconv_mp6 = nn.MaxUnpool1d(self.strides[5], self.strides[5])
        self.deconv_act6 = activation_helper.Activation(activation)
        self.deconv_conv6 = nn.ConvTranspose1d(int(self.filter_coeff * 256), int(self.filter_coeff * 256),
                                                      kernel_size=(self.kernel_sizes[5],),
                                                     stride=(1,), bias=False)
        self.deconv_conv6.weight = nn.Parameter(encoder_state_dict['features.conv6.weight'])

        self.deconv_mp5 = nn.MaxUnpool1d(self.strides[4], self.strides[4])
        self.deconv_act5 = activation_helper.Activation(activation)
        self.deconv_conv5 = nn.ConvTranspose1d(int(self.filter_coeff * 256), int(self.filter_coeff * 128),
                                                      kernel_size=(self.kernel_sizes[4],),
                                                     stride=(1,), bias=False)
        self.deconv_conv5.weight = nn.Parameter(encoder_state_dict['features.conv5.weight'])

        self.deconv_mp4 = nn.MaxUnpool1d(self.strides[3], self.strides[3])
        self.deconv_act4 = activation_helper.Activation(activation)
        self.deconv_conv4 = nn.ConvTranspose1d(int(self.filter_coeff * 128), int(self.filter_coeff * 128),
                                                      kernel_size=(self.kernel_sizes[3],),
                                                     stride=(1,), bias=False)
        self.deconv_conv4.weight = nn.Parameter(encoder_state_dict['features.conv4.weight'])

        self.deconv_mp3 = nn.MaxUnpool1d(self.strides[2], self.strides[2])
        self.deconv_act3 = activation_helper.Activation(activation)
        self.deconv_conv3 = nn.ConvTranspose1d(int(self.filter_coeff * 128), int(self.filter_coeff * 64),
                                                      kernel_size=(self.kernel_sizes[2],),
                                                     stride=(1,), bias=False)
        self.deconv_conv3.weight = nn.Parameter(encoder_state_dict['features.conv3.weight'])

        self.deconv_mp2 = nn.MaxUnpool1d(self.strides[1], self.strides[1])
        self.deconv_act2 = activation_helper.Activation(activation)
        self.deconv_conv2 = nn.ConvTranspose1d(int(self.filter_coeff * 64), int(self.filter_coeff * 64),
                                                      kernel_size=(self.kernel_sizes[1],),
                                                     stride=(1,), bias=False)
        self.deconv_conv2.weight = nn.Parameter(encoder_state_dict['features.conv2.weight'])

        self.deconv_mp1 = nn.MaxUnpool1d(self.strides[0], self.strides[0])
        self.deconv_act1 = activation_helper.Activation(activation)
        self.deconv_conv1 = nn.ConvTranspose1d(int(self.filter_coeff * 64), 1,
                                                      kernel_size=(self.kernel_sizes[0],),
                                                     stride=(1,), bias=False)
        self.deconv_conv1.weight = nn.Parameter(encoder_state_dict['features.conv1.weight'])

    def deconvolve(self, batch, input, encoder_features, switches, target_layer):
        if target_layer < 1 or target_layer > 11:
            raise ValueError("Incorrect target_layer value: {}".format(target_layer))
        x = input
        if target_layer >= 11:
            # print("Applying deconv7")
            x = self.deconv_act11(x)
            x = self.deconv_conv11(x, output_size=[encoder_features['conv10'].shape[-1]])
            # print("deconv11:", x.shape)
        if target_layer >= 10:
            # print("Applying deconv7")
            x = self.deconv_act10(x)
            x = self.deconv_conv10(x, output_size=[encoder_features['conv9'].shape[-1]])
            # print("deconv10:", x.shape)
        if target_layer >= 9:
            # print("Applying deconv7")
            x = self.deconv_act9(x)
            x = self.deconv_conv9(x, output_size=[encoder_features['mp8'].shape[-1]])
            # print("deconv9:", x.shape)
        if target_layer >= 8:
            # print("Applying deconv7")
            x = self.deconv_mp8(x, switches['mp8'], output_size=[encoder_features['conv8'].shape[-1]])
            x = self.deconv_act8(x)
            x = self.deconv_conv8(x, output_size=[encoder_features['mp7'].shape[-1]])
            # print("deconv8:", x.shape)
        if target_layer >= 7:
            # print("Applying deconv7")
            x = self.deconv_mp7(x, switches['mp7'], output_size=[encoder_features['conv7'].shape[-1]])
            x = self.deconv_act7(x)
            x = self.deconv_conv7(x, output_size=[encoder_features['mp6'].shape[-1]])
            # print("deconv7:", x.shape)
        if target_layer >= 6:
            # print("Applying deconv6")
            x = self.deconv_mp6(x, switches['mp6'], output_size=[encoder_features['conv6'].shape[-1]])
            x = self.deconv_act6(x)
            x = self.deconv_conv6(x, output_size=[encoder_features['mp5'].shape[-1]])
        if target_layer >= 5:
            # print("Applying deconv5")
            x = self.deconv_mp5(x, switches['mp5'], output_size=[encoder_features['conv5'].shape[-1]])
            x = self.deconv_act5(x)
            x = self.deconv_conv5(x, output_size=[encoder_features['mp4'].shape[-1]])
        if target_layer >= 4:
            # print("Applying deconv4")
            x = self.deconv_mp4(x, switches['mp4'], output_size=[encoder_features['conv4'].shape[-1]])
            x = self.deconv_act4(x)
            x = self.deconv_conv4(x, output_size=[encoder_features['mp3'].shape[-1]])
        if target_layer >= 3:
            # print("Applying deconv3")
            x = self.deconv_mp3(x, switches['mp3'], output_size=[encoder_features['conv3'].shape[-1]])
            x = self.deconv_act3(x)
            x = self.deconv_conv3(x, output_size=[encoder_features['mp2'].shape[-1]])
        if target_layer >= 2:
            # print("Applying deconv2")
            x = self.deconv_mp2(x, switches['mp2'], output_size=[encoder_features['conv2'].shape[-1]])
            x = self.deconv_act2(x)
            x = self.deconv_conv2(x, output_size=[encoder_features['mp1'].shape[-1]])
        if target_layer >= 1:
            # print("Applying deconv1")
            x = self.deconv_mp1(x, switches['mp1'], output_size=[encoder_features['conv1'].shape[-1]])
            x = self.deconv_act1(x)
            x = self.deconv_conv1(x, output_size=[batch.shape[-1]])
        return x

    def visualize_layer(self, batch, encoder_features, switches, target_layer, top_n=9):
        # the order of things is as follows:
        # 1. Get target layers feature maps
        # 2. select top n most active activation maps. let these be idxs = {i_0, ---- i_n-1}
        # 3. for i in [i_0, - - - - -, i_n-1]:
        #       -> set all activation maps apart from i to zero
        #       -> deconv consecutively until you reach the input space
        # 4. return all these visualizations
        # pass
        # currently only support 1 inference at a time
        assert len(batch) == 1

        conv_maps = encoder_features["conv{}".format(target_layer)]
        feature_maps = encoder_features["act{}".format(target_layer)]
        if target_layer < 9:
            feature_maps = encoder_features['mp{}'.format(target_layer)]
        assert conv_maps.size(0) == feature_maps.size(0) == 1

        # idxs = torch.argsort(feature_maps.sum(dim=2), 1, descending=True)[0, :top_n].detach().cpu().tolist()
        # results = {}
        # for idx in range(feature_maps.size(1)):
        #     inp = torch.zeros_like(feature_maps)
        #     inp[:, idx, :] = feature_maps[:, idx, :]
        #     vis = self.deconvolve(batch, inp, encoder_features, switches, target_layer)
        #     results[idx] = vis.detach().cpu().numpy()
        # return results, idxs
        inp = feature_maps.clone()
        vis = self.deconvolve(batch, inp, encoder_features, switches, target_layer)
        vis = vis.detach().cpu().numpy()
        return vis

    def visualize_layer_v2(self, batch, encoder_features, switches, target_layer, top_n=9):
        # the order of things is as follows:
        # 1. Get target layers feature maps
        # 2. select top n most active activation maps. let these be idxs = {i_0, ---- i_n-1}
        # 3. for i in [i_0, - - - - -, i_n-1]:
        #       -> set all activation maps apart from i to zero
        #       -> deconv consecutively until you reach the input space
        # 4. return all these visualizations
        # pass
        # currently only support 1 inference at a time
        assert len(batch) == 1

        conv_maps = encoder_features["conv{}".format(target_layer)]
        feature_maps = encoder_features["act{}".format(target_layer)]
        if target_layer < 9:
            feature_maps = encoder_features['mp{}'.format(target_layer)]
        assert conv_maps.size(0) == feature_maps.size(0) == 1

        # idxs = torch.argsort(feature_maps.sum(dim=2), 1, descending=True)[0, :top_n].detach().cpu().tolist()
        # results = {}
        results = []
        for idx in range(feature_maps.size(1)):
            inp = torch.zeros_like(feature_maps)
            inp[:, idx, :] = feature_maps[:, idx, :]
            vis = self.deconvolve(batch, inp, encoder_features, switches, target_layer)
            vis = vis.detach().cpu().squeeze().numpy()
            results.append(vis)
        # result = np.asarray(results).mean(0)
        return results

    def visualize_specific_map(self, batch, encoder_features, switches, target_layer, target_filter, top_n=9):
        assert len(batch) == 1

        conv_maps = encoder_features["conv{}".format(target_layer)]
        feature_maps = encoder_features["act{}".format(target_layer)]
        if target_layer < 9:
            feature_maps = encoder_features['mp{}'.format(target_layer)]
        assert conv_maps.size(0) == feature_maps.size(0) == 1
        inp = torch.zeros_like(feature_maps)
        inp[:, target_filter, :] = feature_maps[:, target_filter, :]
        vis = self.deconvolve(batch, inp, encoder_features, switches, target_layer)
        vis = vis.detach().cpu().numpy()
        return vis

    def visualize_top_k_maps(self, batch, encoder_features, switches, target_layer, top_k):
        assert len(batch) == 1

        conv_maps = encoder_features["conv{}".format(target_layer)]
        feature_maps = encoder_features["act{}".format(target_layer)]
        if target_layer < 9:
            feature_maps = encoder_features['mp{}'.format(target_layer)]
        assert conv_maps.size(0) == feature_maps.size(0) == 1
        idxs = torch.argsort(feature_maps.abs().mean(dim=2), 1, descending=True)[0, :top_k].detach().cpu().tolist()
        results = []
        for idx in idxs:
            inp = torch.zeros_like(feature_maps)
            inp[:, idx, :] = feature_maps[:, idx, :]
            vis = self.deconvolve(batch, inp, encoder_features, switches, target_layer)
            vis = vis.detach().cpu().squeeze().numpy()
            results.append(vis)
        return results
