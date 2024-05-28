import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
from torch.nn import init
from torch.autograd import Function
from utils import complex_functions as cf


# torch.manual_seed(42)
torch.autograd.set_detect_anomaly(True)

class dummyhgru(Function):
    @staticmethod
    def forward(ctx, state_2nd_last, last_state, *args):
        ctx.save_for_backward(state_2nd_last, last_state)
        ctx.args = args
        return last_state

    @staticmethod
    def backward(ctx, grad):
        neumann_g = neumann_v = None
        neumann_g_prev = grad.clone()
        neumann_v_prev = grad.clone()

        state_2nd_last, last_state = ctx.saved_tensors

        args = ctx.args
        truncate_iter = args[-1]
        exp_name = args[-2]
        i = args[-3]
        epoch = args[-4]

        normsv = []
        normsg = []
        normg = torch.norm(neumann_g_prev)
        normsg.append(normg.data.item())
        normsv.append(normg.data.item())
        for ii in range(truncate_iter):
            neumann_v = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=neumann_v_prev,
                                            retain_graph=True, allow_unused=True)
            normv = torch.norm(neumann_v[0])
            neumann_g = neumann_g_prev + neumann_v[0]
            normg = torch.norm(neumann_g)

            if normg > 1 or normv > normsv[-1] or normv < 1e-9:
                normsg.append(normg.data.item())
                normsv.append(normv.data.item())
                neumann_g = neumann_g_prev
                break

            neumann_v_prev = neumann_v
            neumann_g_prev = neumann_g

            normsv.append(normv.data.item())
            normsg.append(normg.data.item())

        return (None, neumann_g, None, None, None, None)


class rCell(nn.Module):
    """
    Generate a recurrent cell
    """

    def __init__(self, hidden_size, kernel_size, timesteps, batchnorm=True, grad_method='bptt', use_attention=False,
                 no_inh=False, lesion_alpha=False, lesion_gamma=False, lesion_mu=False, lesion_kappa=False):
        super(rCell, self).__init__()
        self.padding = kernel_size // 2
        self.hidden_size = hidden_size
        self.batchnorm = batchnorm
        self.timesteps = timesteps
        self.use_attention = use_attention
        self.no_inh = no_inh

        if self.use_attention:
            self.bn_amp = nn.BatchNorm2d(hidden_size, affine=True, track_running_stats=False)
            self.a_u_gate = cf.RealToComplexConvolution2D_1D(hidden_size, hidden_size, kernel_size=1, stride=1,
                                                          padding=1 // 2,
                                                          biases=False,
                                                          apply_activ=True)
            self.a_w_gate = cf.RealToComplexConvolution2D_1D(hidden_size, hidden_size, kernel_size=1, stride=1,
                                                          padding=1 // 2,
                                                          biases=False,
                                                          apply_activ=True)

            self.a_h_gate = cf.RealToComplexConvolution2D_1D(hidden_size, hidden_size, kernel_size=1, stride=1,
                                                          padding=1 // 2,
                                                          biases=False,
                                                          apply_activ=True)
            self.h_gate = cf.ComplexConvolution2D_1D(hidden_size, hidden_size, kernel_size=1, stride=1,
                                                       padding=1 // 2,
                                                       biases=False,
                                                       apply_activ=True)
            self.decode_gate = cf.ComplexConvolution2D_1D(hidden_size, 1, kernel_size=1, stride=1, padding=1 // 2,
                                                            biases=False,
                                                            apply_activ=True)

        self.i_w_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        self.i_u_gate = nn.Conv2d(hidden_size, hidden_size, 1)

        self.e_w_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        self.e_u_gate = nn.Conv2d(hidden_size, hidden_size, 1)

        spatial_h_size = 1 #kernel_size
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
            init.constant_(self.i_w_gate.bias.data, -1.)
            init.constant_(self.i_u_gate.bias.data, -1.)
            init.constant_(self.e_w_gate.bias.data, -1.)
            init.constant_(self.e_u_gate.bias.data, -1.)
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

    def forward(self, input_, cinput, inhibition, excitation, activ=F.softplus,
                testmode=False, v_target=0):  # Worked with tanh and softplus
        # Attention gate: filter input_ and excitation
        if self.use_attention:
            ff_drive = self.a_w_gate(input_)
            att_gate = ff_drive + self.a_u_gate(excitation)
            hidden_phases = self.h_gate(ff_drive + cinput)
            new_cinput = hidden_phases
            att_gate_decoded = cf.stable_angle(self.decode_gate(hidden_phases))
            att_gate = torch.sigmoid(self.bn_amp((att_gate + hidden_phases).abs()))

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
            return inhibition, excitation, att_gate_decoded, new_cinput
        else:
            return inhibition, excitation, att_gate_decoded, new_cinput


class CVInT(nn.Module):

    def __init__(self, dimensions, in_channels, timesteps=8, kernel_size=15, jacobian_penalty=False, grad_method='bptt',
                 no_inh=False, lesion_alpha=False, lesion_mu=False, lesion_gamma=False, lesion_kappa=False,
                 nl=F.softplus):
        '''
        '''
        super(CVInT, self).__init__()
        self.timesteps = timesteps
        self.jacobian_penalty = jacobian_penalty
        self.grad_method = grad_method
        self.hgru_size = dimensions
        self.complex_preproc = cf.ComplexConvolution2D(1, dimensions, kernel_size=1, stride=1,
                                                       padding=3 // 2, biases=False, apply_activ=True)
        self.unit1 = rCell(
            hidden_size=self.hgru_size,
            kernel_size=kernel_size,
            use_attention=True,
            no_inh=no_inh,
            lesion_alpha=lesion_alpha,
            lesion_mu=lesion_mu,
            lesion_gamma=lesion_gamma,
            lesion_kappa=lesion_kappa,
            timesteps=timesteps)
        self.nl = nl


    def forward(self, x, testmode=False):

        # Now run RNN
        x_shape = x.shape
        # print(x_shape)
        excitation = torch.zeros((x_shape[0], x_shape[1], x_shape[3], x_shape[4]), requires_grad=False).to(x.device)
        inhibition = torch.zeros((x_shape[0], x_shape[1], x_shape[3], x_shape[4]), requires_grad=False).to(x.device)

        # Loop over frames
        states = []
        gates = []
        cinputs = []
        complex_loss = 0
        prediction_loss = 0
        for t in range(x_shape[2]):

            if t == 0:
                cinput = x[:, :, t]

            out = self.unit1(
                input_=x[:, :, t].abs(),
                cinput=cinput,
                inhibition=inhibition,
                excitation=excitation,
                activ=self.nl,
                testmode=testmode)
            if testmode:
                inhibition, excitation, gate, cinput = out
                gates.append(gate)  # This should learn to keep the winner
                states.append(self.readout_conv(excitation))  # This should learn to keep the winner
                cinputs.append(cinput)
            else:
                inhibition, excitation, gate, cinput = out
                gates.append(gate)
                cinputs.append(cinput)

            # complex_loss = complex_loss + cf.synch_loss(gate, m[:, :, t])

        return excitation


class FC(nn.Module):

    def __init__(self, dimensions, in_channels, nb_frames=32, timesteps=8, kernel_size=15, jacobian_penalty=False,
                 grad_method='bptt', nl=F.tanh, init_phase='ideal', batch_size=128):
        '''
        '''
        super(FC, self).__init__()
        self.timesteps = timesteps
        self.jacobian_penalty = jacobian_penalty
        self.grad_method = grad_method
        self.hgru_size = dimensions
        self.nl = nl
        self.init_phase = init_phase

        self.bn1 = nn.BatchNorm3d(dimensions, eps=1e-03, track_running_stats=False)
        self.bn2 = nn.BatchNorm3d(dimensions * 2, eps=1e-03, track_running_stats=False)
        self.bn3 = nn.BatchNorm3d(dimensions * 2, eps=1e-03, track_running_stats=False)
        self.preproc = cf.ComplexConvolution3D(in_channels, dimensions, kernel_size=3, stride=(1, 2, 2), padding=3 // 2,
                                               biases=True,
                                               apply_activ=False)  # nn.Conv3d(in_channels, dimensions, kernel_size=1, padding=1 // 2)
        # self.preproc = nn.Conv3d(in_channels, dimensions, kernel_size=1, padding=1 // 2)
        self.conv1 = cf.ComplexConvolution3D(dimensions, dimensions * 2, kernel_size=3, stride=(1, 2, 2),
                                             padding=3 // 2, biases=True,
                                             apply_activ=False)  # nn.Conv3d(in_channels, dimensions, kernel_size=1, padding=1 // 2)
        self.conv2 = cf.ComplexConvolution3D(dimensions * 2, dimensions * 2, kernel_size=3, stride=(1, 2, 2),
                                             padding=3 // 2, biases=True,
                                             apply_activ=False)  # nn.Conv3d(in_channels, dimensions, kernel_size=1, padding=1 // 2)
        self.readout = cf.ComplexLinear(dimensions * 2 * nb_frames * 4 * 4, 1, biases=True, last=True,
                                        apply_activ=False)  # nn.Linear(64*32*32*32, 1) # the first 2 is for batch size, the second digit is for the dimension
        # self.readout = nn.Linear(timesteps * self.hgru_size, 1) # the first 2 is for batch size, the second digit is for the dimension

        if self.init_phase == 'learnable':
            self.phase_init = nn.Parameter(
                (torch.rand((batch_size, in_channels, nb_frames, 32, 32)) * 2 * math.pi) - math.pi, requires_grad=True)

    def forward(self, x, m, testmode=False, color=False, phases=None):
        # First step: replicate x over the channel dim self.hgru_size times
        # x = x.repeat(1, self.hgru_size, 1, 1, 1)
        if self.init_phase == 'ideal':
            if color:
                phase_init = cf.initialize_phases_color(x, m)
            else:
                phase_init = cf.initialize_phases(x, m)
        elif self.init_phase == 'ideal2':
            phase_init = cf.initialize_phases2(x, m)
        elif self.init_phase == 'tag':
            phase_init = cf.initialize_phases_tag(x, m)
        elif self.init_phase == 'ideal3':
            if color:
                phase_init = cf.initialize_phases3_color(x, m)
            else:
                phase_init = cf.initialize_phases3(x, m)
        elif self.init_phase == 'random':
            phase_init = (torch.rand_like(x) * 2 * math.pi) - math.pi
        elif self.init_phase == 'learnable':
            # print(x.shape)
            # print(self.phase_init.shape)
            phase_init = self.phase_init
        elif self.init_phase == 'last':
            if color:
                phase_init = cf.initialize_phases_last_color(x, m)
            else:
                phase_init = cf.initialize_phases_last(x, m)
        elif self.init_phase == 'cae':
            phase_init = phases.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        else:
            raise NotImplementedError
        z_in = cf.get_complex_number(x, phase_init)

        x = self.preproc(z_in)
        # x = cf.apply_activation(x, self.nl)
        x = cf.apply_activation_function(x.abs(), cf.stable_angle_2(x), self.bn1)
        x = self.conv1(x)
        # x = cf.apply_activation(x, self.nl)
        x = cf.apply_activation_function(x.abs(), cf.stable_angle_2(x), self.bn2)
        x = self.conv2(x)
        # x = cf.apply_activation(x, self.nl)
        x = cf.apply_activation_function(x.abs(), cf.stable_angle_2(x), self.bn3)

        x_shape = x.shape
        x, complex_output = self.readout(x.reshape(x_shape[0], -1))
        jv_penalty = torch.tensor([1]).float().cuda()
        if testmode: return x, None, None
        return x, jv_penalty
