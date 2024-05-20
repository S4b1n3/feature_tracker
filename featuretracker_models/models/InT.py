import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn import init
from torch.autograd import Function


# torch.manual_seed(42)


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


class InT(nn.Module):

    def __init__(self, dimensions, in_channels, timesteps=8, kernel_size=15, jacobian_penalty=False, grad_method='bptt',
                 no_inh=False, lesion_alpha=False, lesion_mu=False, lesion_gamma=False, lesion_kappa=False,
                 nl=F.softplus):
        '''
        '''
        super(InT, self).__init__()
        self.timesteps = timesteps
        self.jacobian_penalty = jacobian_penalty
        self.grad_method = grad_method
        self.hgru_size = dimensions
        self.preproc = nn.Conv3d(in_channels, dimensions, kernel_size=(1, 3, 3), padding=1 // 2)
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
        # self.bn = nn.BatchNorm2d(self.hgru_size, eps=1e-03, track_running_stats=False)
        self.readout_conv = nn.Conv2d(dimensions, 1, 3, padding=2)
        self.target_conv = nn.Conv2d(2, 1, 5, padding=5 // 2)
        torch.nn.init.zeros_(self.target_conv.bias)
        self.readout_dense = nn.Linear(1, 1)
        self.nl = nl

    def forward(self, x, testmode=False):
        # First step: replicate x over the channel dim self.hgru_size times
        xbn = self.preproc(x)
        xbn = self.nl(xbn)

        # Now run RNN
        x_shape = xbn.shape
        excitation = torch.zeros((x_shape[0], x_shape[1], x_shape[3], x_shape[4]), requires_grad=False).to(x.device)
        inhibition = torch.zeros((x_shape[0], x_shape[1], x_shape[3], x_shape[4]), requires_grad=False).to(x.device)

        # Loop over frames
        states = []
        gates = []
        states_inh = []
        for t in range(x_shape[2]):
            out = self.unit1(
                input_=xbn[:, :, t],
                inhibition=inhibition,
                excitation=excitation,
                activ=self.nl,
                testmode=testmode)
            if testmode:
                inhibition, excitation, gate = out
                gates.append(gate)  # This should learn to keep the winner
                states.append(self.readout_conv(excitation))  # This should learn to keep the winner
                states_inh.append(self.readout_conv(inhibition))  # This should learn to keep the winner
            else:
                inhibition, excitation = out
        output = torch.cat([self.readout_conv(excitation), x[:, 2, 0][:, None]], 1)

        output = self.target_conv(output)  # output.sum(1, keepdim=True))  # 2 channels -> 1. Is the dot in the target?
        output = F.avg_pool2d(output, kernel_size=output.size()[2:])
        output = output.reshape(x_shape[0], -1)
        output = self.readout_dense(output)
        pen_type = 'l1'
        jv_penalty = torch.tensor([1]).float().cuda()
        if testmode: return output, torch.stack(states, 1), torch.stack(gates, 1)#, torch.stack(states_inh, 1)
        return output, jv_penalty


class FC(nn.Module):

    def __init__(self, dimensions, in_channels, nb_frames=32, timesteps=8, kernel_size=15, jacobian_penalty=False,
                 grad_method='bptt', nl=F.softplus):
        '''
        '''
        super(FC, self).__init__()
        self.timesteps = timesteps
        self.jacobian_penalty = jacobian_penalty
        self.grad_method = grad_method
        self.hgru_size = dimensions
        self.bn1 = nn.BatchNorm3d(dimensions, eps=1e-03, track_running_stats=False)
        self.bn2 = nn.BatchNorm3d(dimensions * 2, eps=1e-03, track_running_stats=False)
        self.bn3 = nn.BatchNorm3d(dimensions * 2, eps=1e-03, track_running_stats=False)

        self.preproc = nn.Conv3d(in_channels, dimensions, kernel_size=kernel_size, padding=kernel_size // 2, stride=(1, 2, 2))
        # self.conv1 = nn.Conv3d(dimensions, dimensions * 2, kernel_size=kernel_size, padding=kernel_size // 2, stride=(1, 2, 2))
        # self.conv2 = nn.Conv3d(dimensions * 2, dimensions * 2, kernel_size=kernel_size, padding=kernel_size // 2, stride=(1, 2, 2))

        self.readout = nn.Linear(dimensions * nb_frames * 18 * 18,
                                 1)  # the first 2 is for batch size, the second digit is for the dimension
        self.nl = nl
        # self.readout = nn.Linear(timesteps * self.hgru_size, 1) # the first 2 is for batch size, the second digit is for the dimension

    def forward(self, x, testmode=False):
        # First step: replicate x over the channel dim self.hgru_size times
        # x = x.repeat(1, self.hgru_size, 1, 1, 1)
        x = self.nl(self.preproc(x))
        # print(x.shape)
        x = self.bn1(x)
        # x = self.nl(self.conv1(x))
        # x = self.bn2(x)
        # x = self.nl(self.conv2(x))
        # x = self.bn3(x)
        x_shape = x.shape
        # print(x_shape)
        x = self.readout(x.reshape(x_shape[0], -1))
        jv_penalty = torch.tensor([1]).float().cuda()
        if testmode: return x, None, None
        return x, jv_penalty


class FC_Siamese(nn.Module):

    def __init__(self, dimensions, in_channels, nb_frames=32, timesteps=8, kernel_size=15, jacobian_penalty=False,
                 grad_method='bptt', nl=F.softplus):
        '''
        '''
        super(FC_Siamese, self).__init__()
        self.timesteps = timesteps
        self.jacobian_penalty = jacobian_penalty
        self.grad_method = grad_method
        self.hgru_size = dimensions
        self.bn1 = nn.BatchNorm3d(dimensions, eps=1e-03, track_running_stats=False)
        self.bn2 = nn.BatchNorm3d(dimensions * 2, eps=1e-03, track_running_stats=False)
        self.bn3 = nn.BatchNorm3d(dimensions * 2, eps=1e-03, track_running_stats=False)

        self.preproc = nn.Conv3d(in_channels, dimensions, kernel_size=3, padding=3 // 2, stride=(1, 2, 2))
        self.conv1 = nn.Conv3d(dimensions, dimensions * 2, kernel_size=3, padding=3 // 2, stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(dimensions * 2, dimensions * 2, kernel_size=3, padding=3 // 2, stride=(1, 2, 2))

        self.readout = nn.Linear(2*dimensions * 2 * nb_frames * 4 * 4,
                                 1)  # the first 2 is for batch size, the second digit is for the dimension
        self.nl = nl
        # self.readout = nn.Linear(timesteps * self.hgru_size, 1) # the first 2 is for batch size, the second digit is for the dimension

    def forward_one(self, x, testmode=False, color=False):
        # First step: replicate x over the channel dim self.hgru_size times
        # x = x.repeat(1, self.hgru_size, 1, 1, 1)
        x = self.nl(self.preproc(x))
        # print(x.shape)
        x = self.bn1(x)
        x = self.nl(self.conv1(x))
        x = self.bn2(x)
        x = self.nl(self.conv2(x))
        x = self.bn3(x)
        x_shape = x.shape
        # print(x_shape)
        x = x.reshape(x_shape[0], -1)
        #jv_penalty = torch.tensor([1]).float().cuda()
        #if testmode: return x, None, None
        return x#, jv_penalty

    def forward(self, x, mask, testmode=False, color=False):
        x1 = self.forward_one(x)
        x2 = self.forward_one(mask)
        x = torch.cat([x1, x2], dim=1)
        x = self.readout(x)
        jv_penalty = torch.tensor([1]).float().cuda()
        if testmode: return x, None, None
        return x, jv_penalty



