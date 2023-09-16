import functools

import torch
from torch import nn
from torch.nn import functional as F

from math import ceil
from models.networks import BaseNetwork
from models.modules.snlayer import SNConv2d, SNLinear

class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, activation, kernel_size=3, stride=1):
        super(ConvBlock, self).__init__()
        padding = int(ceil((kernel_size-1)/2))
        conv_layer = SNConv2d(in_dim, out_dim, kernel_size=kernel_size,
                     stride=stride, padding=padding)
        conv_layer.weight.data.normal_(mean=0.0, std=1.0)
        conv_layer.bias.data.fill_(0.0)
        self.block = nn.Sequential(
            conv_layer,
            activation)

    def forward(self, x):
        out = self.block(x)
        return out

class ResnetEBMBlock(nn.Module):
    """Define a mobile-version Resnet block"""

    def __init__(self, in_dim, out_dim, activation):
        super(ResnetEBMBlock, self).__init__()
        self.activation = activation
        self.conv_block = self.build_conv_block(in_dim, out_dim)
        self.pooling_layer = nn.AvgPool2d(kernel_size=2, stride=2)
        if in_dim != out_dim:
            shortcut_conv = SNConv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
            shortcut_conv.weight.data.normal_(mean=0.0, std=1.0)
            shortcut_conv.bias.data.fill_(0.0)
            self.shortcut =  nn.Sequential(
                shortcut_conv,
            )
        else:
            self.shortcut = nn.Sequential()

    def build_conv_block(self, in_dim, out_dim):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """

        conv_layer_1 = SNConv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        conv_layer_1.weight.data.normal_(mean=0.0, std=1.0)
        conv_layer_1.bias.data.fill_(0.0)
        conv_layer_2 = SNConv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, lower_bound=True)
        conv_layer_2.weight.data.normal_(mean=0.0, std=1e-10)
        conv_layer_2.bias.data.fill_(0.0)

        conv_block = [conv_layer_1,
                      self.activation,
                      conv_layer_2]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""

        out = self.shortcut(x) + self.conv_block(x)
        out = self.pooling_layer(out)
        out = self.activation(out)
        return out

class ResnetEBM(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, input_nc, n_blocks=7):
        assert (n_blocks >= 0)
        super(ResnetEBM, self).__init__()
        self.n_blocks=n_blocks
        input_nc = input_nc//2
        self.fromRGB = ConvBlock(2*3, input_nc, activation=nn.LeakyReLU(), kernel_size=3, stride=1)

        input_nc = input_nc
        output_nc = 2 * input_nc
        setattr(self, 'primal_%d'%0,ResnetEBMBlock(input_nc, output_nc, activation=nn.LeakyReLU()))
        for i in range(1, n_blocks):
            if i == 1:
                input_nc = output_nc
                output_nc = 2 * output_nc
            else:
                input_nc = output_nc
                output_nc = output_nc
            setattr(self, 'primal_%d'%i,ResnetEBMBlock(input_nc, output_nc, activation=nn.LeakyReLU()))
        self.linear_layer = SNLinear(output_nc, 1)
        self.activation = nn.ReLU(True)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, t_input, s_input):
        t_input = self.avg_pool(t_input)
        s_input = self.avg_pool(s_input)
        input = torch.cat([t_input, s_input], 1)
        output = self.fromRGB(input)
        for i in range(self.n_blocks):
            output = getattr(self, 'primal_%d'%i)(output)
        output = self.activation(output)
        output = torch.mean(output.reshape([output.shape[0], output.shape[1], -1]), dim=2)
        output = self.linear_layer(output)
        return output
