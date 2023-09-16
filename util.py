import torch
from torch.optim import lr_scheduler
import math
def save_network(models, model_names, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for model_name, model in zip(model_names, models):
        save_filename = '%s_net_%s.pth' % (epoch, model_name)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(model.state_dict(), save_path)

def get_scheduler(optimizers, nepochs, nepochs_decay):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    """
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 - nepochs) / float(nepochs_decay + 1)
        return lr_l
    scheduler_list = []
    for optimizer in optimizers:
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        scheduler_list.append(scheduler)
    return scheduler_list

def update_learning_rate(schedulers, optimizers):
    for scheduler in schedulers:
        scheduler.step()
    lr = optimizers[0].param_groups[0]['lr']
    print('learning rate = %.7f' % lr)

"""
borrowed from
https://github.com/godisboy/SN-GAN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import conv
from torch.nn.modules import Linear
from torch.nn.modules.utils import _pair

#define _l2normalization
def _l2normalize(v, eps=1e-12):
    return v / (torch.norm(v) + eps)

def max_singular_value(W, u=None, Ip=1):
    """
    power iteration for weight parameter
    """
    #xp = W.data
    if not Ip >= 1:
        raise ValueError("Power iteration should be a positive integer")
    if u is None:
        u = torch.FloatTensor(1, W.size(0)).normal_(0, 1).cuda()
    _u = u
    for _ in range(Ip):
        _v = _l2normalize(torch.matmul(_u, W.data), eps=1e-12)
        _u = _l2normalize(torch.matmul(_v, torch.transpose(W.data, 0, 1)), eps=1e-12)
    sigma = torch.sum(F.linear(_u, torch.transpose(W.data, 0, 1)) * _v)
    return sigma, _u

class SNConv2d(conv._ConvNd):

    r"""Applies a 2D convolution over an input signal composed of several input
    planes.
    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{in}, H, W)` and output :math:`(N, C_{out}, H_{out}, W_{out})`
    can be precisely described as:
    .. math::
        \begin{array}{ll}
        out(N_i, C_{out_j})  = bias(C_{out_j})
                       + \sum_{{k}=0}^{C_{in}-1} weight(C_{out_j}, k)  \star input(N_i, k)
        \end{array}
    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.
    | :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.
    | :attr:`padding` controls the amount of implicit zero-paddings on both
    |  sides for :attr:`padding` number of points for each dimension.
    | :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.
    | :attr:`groups` controls the connections between inputs and outputs.
      `in_channels` and `out_channels` must both be divisible by `groups`.
    |       At groups=1, all inputs are convolved to all outputs.
    |       At groups=2, the operation becomes equivalent to having two conv
                 layers side by side, each seeing half the input channels,
                 and producing half the output channels, and both subsequently
                 concatenated.
            At groups=`in_channels`, each input channel is convolved with its
                 own set of filters (of size `out_channels // in_channels`).
    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:
        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension
    .. note::
         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.
    .. note::
         The configuration when `groups == in_channels` and `out_channels = K * in_channels`
         where `K` is a positive integer is termed in literature as depthwise convolution.
         In other words, for an input of size :math:`(N, C_{in}, H_{in}, W_{in})`, if you want a
         depthwise convolution with a depthwise multiplier `K`,
         then you use the constructor arguments
         :math:`(in\_channels=C_{in}, out\_channels=C_{in} * K, ..., groups=C_{in})`
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
          :math:`H_{out} = floor((H_{in}  + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1) / stride[0] + 1)`
          :math:`W_{out} = floor((W_{in}  + 2 * padding[1] - dilation[1] * (kernel\_size[1] - 1) - 1) / stride[1] + 1)`
    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (out_channels, in_channels, kernel_size[0], kernel_size[1])
        bias (Tensor):   the learnable bias of the module of shape (out_channels)
        W(Tensor): Spectrally normalized weight
        u (Tensor): the right largest singular value of W.
    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation
    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros', lower_bound=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(SNConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode=padding_mode)
        self.register_buffer('u', torch.Tensor(1, out_channels).normal_())
        self.lower_bound = lower_bound

    @property
    def W_(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = max_singular_value(w_mat, self.u)
        self.u.copy_(_u)
        if self.lower_bound:
            sigma = sigma + 1e-6
            return self.weight / sigma * torch.min(sigma, torch.ones_like(sigma))
        else:
            return self.weight / sigma

    def forward(self, input):
        return F.conv2d(input, self.W_, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class SNLinear(Linear):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`
       Args:
           in_features: size of each input sample
           out_features: size of each output sample
           bias: If set to False, the layer will not learn an additive bias.
               Default: ``True``
       Shape:
           - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
             additional dimensions
           - Output: :math:`(N, *, out\_features)` where all but the last dimension
             are the same shape as the input.
       Attributes:
           weight: the learnable weights of the module of shape
               `(out_features x in_features)`
           bias:   the learnable bias of the module of shape `(out_features)`
           W(Tensor): Spectrally normalized weight
           u (Tensor): the right largest singular value of W.
       """
    def __init__(self, in_features, out_features, bias=True):
        super(SNLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('u', torch.Tensor(1, out_features).normal_())

    @property
    def W_(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = max_singular_value(w_mat, self.u)
        self.u.copy_(_u)
        return self.weight / sigma

    def forward(self, input):
        return F.linear(input, self.W_, self.bias)
from math import ceil
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

    def __init__(self, in_dim, out_dim, activation):
        super(ResnetEBMBlock, self).__init__()
        self.activation = activation
        self.conv_block = self.build_conv_block(in_dim, out_dim)
        self.pooling_layer = nn.AvgPool2d(kernel_size=2, stride=2)
        if in_dim != out_dim:
            # Spectral normalization with Conv2D
            shortcut_conv = SNConv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
            shortcut_conv.weight.data.normal_(mean=0.0, std=1.0)
            shortcut_conv.bias.data.fill_(0.0)
            self.shortcut =  nn.Sequential(
                shortcut_conv,
            )
        else:
            self.shortcut = nn.Sequential()

    def build_conv_block(self, in_dim, out_dim):
        # Spectral normalizaion with Conv2D
        conv_layer_1 = SNConv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        conv_layer_1.weight.data.normal_(mean=0.0, std=1.0)
        conv_layer_1.bias.data.fill_(0.0)
        # Spectral normalization with Conv2D
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




import torch.nn as nn


class ResnetEBM(nn.Module):
    def __init__(self, nec, n_blocks=7):
        assert (n_blocks > 0)
        super(ResnetEBM, self).__init__()
        self.n_blocks=n_blocks
        self.fromRGB = ConvBlock(2*3, nec, activation=nn.LeakyReLU(), kernel_size=3, stride=1)

        input_nc = nec
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

def MCMC(ebm, student_im, step_size=50.0, sigma=0.005, MCMC_steps=10):
    MCMC_img = torch.autograd.Variable(student_im.data.clone(), requires_grad=True)

    for k in range(MCMC_steps):
        out = ebm(MCMC_img, student_im.detach())
        gradient = torch.autograd.grad(out.sum(), [MCMC_img], only_inputs=True)[0]
        MCMC_img = MCMC_img - step_size * gradient         + sigma * torch.randn_like(MCMC_img)

    MCMC_img = MCMC_img.detach().clamp(min=-1.0, max=1.0)
    return MCMC_img
