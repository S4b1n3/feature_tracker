import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn import init
from torch.autograd import Function

class InT(nn.Module):
    """
    Generate a recurrent cell
    """

    def __init__(self, hidden_size, kernel_size, timesteps, batchnorm=True, grad_method='bptt', use_attention=False,
                 no_inh=False, lesion_alpha=False, lesion_gamma=False, lesion_mu=False, lesion_kappa=False):
        super(InT, self).__init__()
        self.padding = kernel_size // 2
        self.hidden_size = hidden_size
        self.batchnorm = batchnorm
        self.timesteps = timesteps
        self.use_attention = use_attention
        self.no_inh = no_inh

        if self.use_attention:
            self.a_w_gate = nn.Conv2d(hidden_size, hidden_size, 1, padding=1 // 2)
            self.a_u_gate = nn.Conv2d(hidden_size, hidden_size, 1, padding=1 // 2)
            init.orthogonal_(self.a_w_gate.weight)
            init.orthogonal_(self.a_u_gate.weight)
            init.constant_(self.a_w_gate.bias, 1.)
            init.constant_(self.a_u_gate.bias, 1.)

        self.i_w_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        self.i_u_gate = nn.Conv2d(hidden_size, hidden_size, 1)

        self.e_w_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        self.e_u_gate = nn.Conv2d(hidden_size, hidden_size, 1)

        spatial_h_size = kernel_size
        self.h_padding = spatial_h_size // 2
        self.w_exc = nn.Parameter(torch.empty(hidden_size, hidden_size, spatial_h_size, spatial_h_size))
        init.orthogonal_(self.w_exc)

        if not no_inh:
            self.w_inh = nn.Parameter(torch.empty(hidden_size, hidden_size, spatial_h_size, spatial_h_size))
            init.orthogonal_(self.w_inh)

        self.alpha = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.mu = nn.Parameter(torch.empty((hidden_size, 1, 1)))

        self.gamma = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.kappa = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.w = nn.Parameter(torch.empty((hidden_size, 1, 1)))

        self.bn = nn.ModuleList(
            [nn.BatchNorm2d(hidden_size, eps=1e-03, affine=True, track_running_stats=False) for i in range(2)])

        init.orthogonal_(self.i_w_gate.weight)
        init.orthogonal_(self.i_u_gate.weight)
        init.orthogonal_(self.e_w_gate.weight)
        init.orthogonal_(self.e_u_gate.weight)

        for bn in self.bn:
            init.constant_(bn.weight, 0.1)

        if not no_inh:
            init.constant_(self.alpha, 1.)
            init.constant_(self.mu, 0.)
        # init.constant_(self.alpha, 0.1)
        # init.constant_(self.mu, 1)
        init.constant_(self.gamma, 0.)
        # init.constant_(self.w, 1.)
        init.constant_(self.kappa, 1.)

        if self.use_attention:
            self.i_w_gate.bias.data = -self.a_w_gate.bias.data
            self.e_w_gate.bias.data = -self.a_w_gate.bias.data
            self.i_u_gate.bias.data = -self.a_u_gate.bias.data
            self.e_u_gate.bias.data = -self.a_u_gate.bias.data
        else:
            init.uniform_(self.i_w_gate.bias.data, 1, self.timesteps - 1)
            self.i_w_gate.bias.data.log()
            self.i_u_gate.bias.data.log()
            self.e_w_gate.bias.data = -self.i_w_gate.bias.data
            self.e_u_gate.bias.data = -self.i_u_gate.bias.data
        if lesion_alpha:
            self.alpha.requires_grad = False
            self.alpha.weight = 0.
        if lesion_mu:
            self.mu.requires_grad = False
            self.mu.weight = 0.
        if lesion_gamma:
            self.gamma.requires_grad = False
            self.gamma.weight = 0.
        if lesion_kappa:
            self.kappa.requires_grad = False
            self.kappa.weight = 0.

    def forward(self, input_, inhibition, excitation, activ=F.softplus,
                testmode=False):  # Worked with tanh and softplus
        # Attention gate: filter input_ and excitation
        if self.use_attention:
            att_gate = torch.sigmoid(
                self.a_w_gate(input_) + self.a_u_gate(excitation))  # Attention Spotlight -- MOST RECENT WORKING

        # Gate E/I with attention immediately
        if self.use_attention:
            gated_input = input_  # * att_gate  # In activ range
            gated_excitation = att_gate * excitation  # att_gate * excitation
        else:
            gated_input = input_
            gated_excitation = excitation
        gated_inhibition = inhibition

        if not self.no_inh:
            # Compute inhibition
            inh_intx = self.bn[0](F.conv2d(gated_excitation, self.w_inh, padding=self.h_padding))  # in activ range
            inhibition_hat = activ(input_ - activ(inh_intx * (self.alpha * gated_inhibition + self.mu)))

            # Integrate inhibition
            inh_gate = torch.sigmoid(self.i_w_gate(gated_input) + self.i_u_gate(gated_inhibition))
            inhibition = (1 - inh_gate) * inhibition + inh_gate * inhibition_hat  # In activ range
        else:
            inhibition, gated_inhibition = gated_excitation, excitation

        # Pass to excitatory neurons
        exc_gate = torch.sigmoid(self.e_w_gate(gated_inhibition) + self.e_u_gate(gated_excitation))
        exc_intx = self.bn[1](F.conv2d(inhibition, self.w_exc, padding=self.h_padding))  # In activ range
        excitation_hat = activ(
            exc_intx * (self.kappa * inhibition + self.gamma))  # Skip connection OR add OR add by self-sim

        excitation = (1 - exc_gate) * excitation + exc_gate * excitation_hat
        if testmode:
            return inhibition, excitation, att_gate
        else:
            return inhibition, excitation

class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, padding_mode='zeros'):
        " Referenced from https://github.com/happyjin/ConvGRU-pytorch"
        super(ConvGRUCell, self).__init__()
        self.hidden_dim = hidden_dim

        if padding_mode == 'zeros':
            if not isinstance(kernel_size, (list, tuple)):
                kernel_size = (kernel_size, kernel_size)

            padding = kernel_size[0] // 2, kernel_size[1] // 2
            self.conv_reset = nn.Conv2d(input_dim + hidden_dim, self.hidden_dim, kernel_size, padding=padding)
            self.conv_update = nn.Conv2d(input_dim + hidden_dim, self.hidden_dim, kernel_size, padding=padding)

            self.conv_state_new = nn.Conv2d(input_dim + hidden_dim, self.hidden_dim, kernel_size, padding=padding)
        else:
            self.conv_reset = conv_block(input_dim + hidden_dim, hidden_dim, kernel_size=kernel_size, stride=1,
                                         padding=int(kernel_size // 2), batch_norm=False, relu=False,
                                         padding_mode=padding_mode)

            self.conv_update = conv_block(input_dim + hidden_dim, hidden_dim, kernel_size=kernel_size, stride=1,
                                          padding=int(kernel_size // 2), batch_norm=False, relu=False,
                                          padding_mode=padding_mode)

            self.conv_state_new = conv_block(input_dim + hidden_dim, hidden_dim, kernel_size=kernel_size, stride=1,
                                             padding=int(kernel_size // 2), batch_norm=False, relu=False,
                                             padding_mode=padding_mode)

    def forward(self, input, state_cur, _, testmode=False):
        input_state_cur = torch.cat([input, state_cur], dim=1)

        reset_gate = torch.sigmoid(self.conv_reset(input_state_cur))
        update_gate = torch.sigmoid(self.conv_update(input_state_cur))

        input_state_cur_reset = torch.cat([input, reset_gate * state_cur], dim=1)
        state_new = torch.tanh(self.conv_state_new(input_state_cur_reset))

        state_next = (1.0 - update_gate) * state_cur + update_gate * state_new
        if testmode:
            return state_next, reset_gate
        else:
            return state_next, _


def conv_block(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True,
               batch_norm=True, relu=True, padding_mode='zeros'):
    layers = []
    assert padding_mode == 'zeros' or padding_mode == 'replicate'

    if padding_mode == 'replicate' and padding > 0:
        assert isinstance(padding, int)
        layers.append(nn.ReflectionPad2d(padding))
        padding = 0

    layers.append(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, bias=bias))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_planes))
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        i_t = torch.sigmoid(self.Wxi(x) + self.Whi(h))  # + c * self.Wci)
        f_t = torch.sigmoid(self.Wxf(x) + self.Whf(h))  # + c * self.Wcf)
        c_t = f_t * c + i_t * torch.tanh(self.Wxc(x) + self.Whc(h))
        o_t = torch.sigmoid(self.Wxo(x) + self.Who(h))  # + cc * self.Wco)
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_channels, height, width, device=self.Wxi.weight.device),
                torch.zeros(batch_size, self.hidden_channels, height, width, device=self.Wxi.weight.device))
