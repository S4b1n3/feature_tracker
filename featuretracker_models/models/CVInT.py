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
            # self.a_w_gate = cf.FullyComplexConvolution2D(hidden_size, hidden_size, kernel_size=3, stride=1, padding=3 // 2, biases=False,
            #                                         apply_activ=True)
            self.bn_amp = nn.InstanceNorm2d(hidden_size, affine=True, track_running_stats=False)
            # self.a_u_gate = cf.ComplexConvolution2D(hidden_size, hidden_size, kernel_size=1, stride=1, padding=1 // 2, biases=False,
            #                                         apply_activ=True)
            self.a_u_gate = cf.RealToComplexConvolution2D(hidden_size, hidden_size, kernel_size=3, stride=1,
                                                          padding=3 // 2,
                                                          biases=False,
                                                          apply_activ=True)
            self.a_w_gate = cf.RealToComplexConvolution2D(hidden_size, hidden_size, kernel_size=3, stride=1,
                                                          padding=3 // 2,
                                                          biases=False,
                                                          apply_activ=True)

            self.a_h_gate = cf.RealToComplexConvolution2D(hidden_size, hidden_size, kernel_size=3, stride=1,
                                                          padding=3 // 2,
                                                          biases=False,
                                                          apply_activ=True)
            # self.h_gate = cf.FullyComplexConvolution2D(hidden_size, hidden_size, kernel_size=3, stride=1,
            #                                            padding=3 // 2,
            #                                            biases=False,
            #                                            apply_activ=True)
            self.h_gate = cf.ComplexConvolution2D(hidden_size, hidden_size, kernel_size=3, stride=1,
                                                       padding=3 // 2,
                                                       biases=False,
                                                       apply_activ=True)
            self.decode_gate = cf.ComplexConvolution2D(hidden_size, 1, kernel_size=3, stride=1, padding=3 // 2,
                                                            biases=False,
                                                            apply_activ=True)
            # self.a_w_gate = nn.Conv2d(hidden_size, hidden_size, 1, padding=1 // 2)
            # self.a_u_gate = nn.Conv2d(hidden_size, hidden_size, 1, padding=1 // 2)
            # init.orthogonal_(self.a_w_gate.kernel_conv.weight)
            # init.orthogonal_(self.a_u_gate.kernel_conv.weight) #.kernel_conv
            # init.constant_(self.a_w_gate.kernel_conv.bias, 1.)
            # init.constant_(self.a_u_gate.bias, 1.) #.kernel_conv

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
            init.constant_(self.i_w_gate.bias.data, -1.)
            init.constant_(self.i_u_gate.bias.data, -1.)
            init.constant_(self.e_w_gate.bias.data, -1.)
            init.constant_(self.e_u_gate.bias.data, -1.)
            # self.i_w_gate.bias.data = -self.a_u_gate.bias.data #-self.a_w_gate.kernel_conv.bias.data
            # self.e_w_gate.bias.data = -self.a_u_gate.bias.data #-self.a_w_gate.kernel_conv.bias.data
            # self.i_u_gate.bias.data = -self.a_u_gate.bias.data #.kernel_conv
            # self.e_u_gate.bias.data = -self.a_u_gate.bias.data #.kernel_conv
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
            # att_gate_complex = self.a_w_gate(cinput)
            # att_gate = att_gate_complex.abs() + self.a_u_gate(excitation)
            # att_gate_decoded = cf.stable_angle(self.decode_gate(att_gate_complex))
            # att_gate = torch.sigmoid(att_gate)
            # new_cinput = att_gate_complex

            # new_cinput = self.a_w_gate(input_) #self.a_w_gate(cinput)
            ff_drive = self.a_w_gate(input_)
            att_gate = ff_drive + self.a_u_gate(excitation)  # self.a_u_gate(cf.get_complex_number(excitation, cf.stable_angle(cinput))) # #cf.stable_angle(cinput)  # Attention Spotlight -- MOST RECENT WORKING
            hidden_phases = self.h_gate(ff_drive + cinput)  # ff_drive +
            new_cinput = hidden_phases
            att_gate_decoded = cf.stable_angle(self.decode_gate(hidden_phases))
            att_gate = torch.sigmoid(self.bn_amp((att_gate + hidden_phases).abs()))  # att_gate.abs() + self.sig_bias #self.bn_amp(att_gate.abs()) #att_gate.real #att_gate.abs()-shift/3

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
        self.preproc = nn.Conv3d(3, dimensions, kernel_size=(1, 3, 3), padding=(1 // 2, 3 // 2, 3 // 2))
        self.complex_preproc = cf.ComplexConvolution2D(1, dimensions, kernel_size=3, stride=1,
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
        # self.bn = nn.BatchNorm2d(self.hgru_size, eps=1e-03, track_running_stats=False)
        self.readout_conv = nn.Conv2d(dimensions, 1, 3, padding=3 // 2)
        self.target_conv = nn.Conv2d(2, 1, 5, padding=5 // 2)
        torch.nn.init.zeros_(self.target_conv.bias)
        self.readout_dense = nn.Linear(1, 1)
        self.nl = nl

        # self.init_phases = nn.Conv2d(3, 1, 1, padding=1 // 2)

        self.init_phases = cf.RealToComplexConvolution2D(3, 3, kernel_size=5, stride=1,
                                                       padding=5 // 2, biases=False, apply_activ=True)

    def forward(self, x, m, testmode=False):
        # First step: replicate x over the channel dim self.hgru_size times

        # create luminance channel
        # luminance = 1/3 * x[:, 0] + 1/3 * x[:, 1] + 1/3 * x[:, 2]
        # x = torch.cat([x, luminance.unsqueeze(1)], 1)
        # xbn = self.preproc(x)
        # xbn = self.nl(xbn)

        # x_channel = torch.zeros((x.shape[0], self.hgru_size, x.shape[2], x.shape[3], x.shape[4]), requires_grad=False).to(x.device)
        # for i in range(3):
        #     x_channel += 1/3*self.nl(self.preproc(x[:, i][:, None]))
        # xbn = x_channel

        xbn = self.preproc(x)
        xbn = self.nl(xbn)

        # Now run RNN
        x_shape = xbn.shape
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

                # phases = cf.initialize_phases_first_channels_color(x[:, :, t], m[:, :, t])
                phases = (torch.rand_like(x[:, :, t]) * 2 * np.pi) - np.pi #
                # phases = cf.stable_angle(self.init_phases(x[:, :, t])).repeat(1, 3, 1, 1)
                # phases = self.init_phases(x[:, :, t])#.repeat(1, 3, 1, 1)


                # phases = cf.initialize_phases_first(x[:,:,t], m[:,:,t])
                cinput = cf.get_complex_number(x[:, :, t], phases)

                # cinput = self.complex_preproc(cinput)

                cinput_ = torch.zeros((phases.shape[0], self.hgru_size, phases.shape[2], phases.shape[3]), requires_grad=False, dtype=torch.cfloat).to(x.device)
                for i in range(3):
                    cinput_ += self.complex_preproc(cinput[:, i][:, None])
                cinput = cinput_
            # else:
            #     phases = torch.rand_like(cf.stable_angle(cinput)) * 2 * np.pi - np.pi
            #     cinput = cf.get_complex_number(cinput.abs(), phases)

            # else:
            #     cinput = cf.get_complex_number(xbn[:, :, t], cf.stable_angle(cinput))
            # cinput = cf.get_complex_number(x[:, :, t], gates[-1])
            # cinput = self.complex_preproc(cinput)
            out = self.unit1(
                input_=xbn[:, :, t],
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
            # if t == x_shape[2] - 1:
            #     import pdb; pdb.set_trace()
            # if t < x_shape[2] - 1:
            complex_loss = complex_loss + cf.synch_loss(gate, m[:, :, t])
            if t > 0:
                # prediction_loss = prediction_loss + torch.nn.functional.mse_loss(gate.squeeze(), gates[-2].squeeze())
                # phases_gt = cf.initialize_phases_first_channels_color(x[:, :, t], m[:, :, t])
                # prediction_loss = prediction_loss + torch.nn.functional.cross_entropy(torch.nn.functional.one_hot(phases_hidden.to(torch.int64), ).to(torch.float), phases_previous.to(torch.int64))
                # prediction_loss = prediction_loss + torch.nn.functional.mse_loss(cf.stable_angle(cinput.squeeze()), cf.stable_angle(cinputs[-2].squeeze()))
                # prediction_loss = prediction_loss + torch.nn.functional.mse_loss(gate.squeeze(), phases_gt[:,0].squeeze())
                prediction_loss = prediction_loss + torch.nn.functional.mse_loss(cinput.real, cinputs[-2].real)
                prediction_loss = prediction_loss + torch.nn.functional.mse_loss(cinput.imag, cinputs[-2].imag)

            # complex_loss += torch.nn.CrossEntropyLoss(weight=torch.Tensor([10,1]).cuda())(gate.squeeze(), m[:, 1:, t])
            # complex_loss += torch.nn.BCEWithLogitsLoss()(gate.squeeze(), m[:,-1, t]) #torch.nn.CrossEntropyLoss()(gate, m[:,-1, t].long())
        # torch.save(gates, './outputs/1h_1c_no_color_initf1_synchloss_gates.pt')
        # torch.save(states, './outputs/1h_1c_no_color_initf1_synchloss_states.pt')
        # torch.save(cinputs, './outputs/1h_1c_no_color_initf1_synchloss_cinputs.pt')
        # torch.save(x, './outputs/1h_1c_no_color_initf1_synchloss_x.pt')
        # torch.save(m, './outputs/1h_1c_no_color_initf1_synchloss_m.pt')

        output = torch.cat([self.readout_conv(excitation), x[:, 2, 0][:, None]], 1)

        output = self.target_conv(output)  # output.sum(1, keepdim=True))  # 2 channels -> 1. Is the dot in the target?
        output = F.avg_pool2d(output, kernel_size=output.size()[2:])
        output = output.reshape(x_shape[0], -1)
        output = self.readout_dense(output)
        pen_type = 'l1'
        jv_penalty = torch.tensor([1]).float().cuda()

        if testmode: return (output, complex_loss, prediction_loss), jv_penalty
        return (output, complex_loss, prediction_loss), jv_penalty


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


class KomplexNet(nn.Module):

    def __init__(self, kuramoto_channels=8, dimensions=32, test_mode=False, kuramoto_args=None, nb_frames=32):
        super(KomplexNet, self).__init__()
        self.test_mode = test_mode
        self.mean_r = kuramoto_args.mean_r
        self.epsilon = kuramoto_args.epsilon
        self.timesteps = kuramoto_args.timesteps
        self.lr_kuramoto = kuramoto_args.lr_kuramoto
        self.lr_kuramoto_update = kuramoto_args.lr_kuramoto_update
        self.from_input = kuramoto_args.from_input
        self.std_r = kuramoto_args.std_r
        self.h = kuramoto_args.k
        self.w = kuramoto_args.k
        self.distraction_masks = kuramoto_args.distractor_masks
        self.nb_frames = nb_frames
        self.kuramoto_channels = kuramoto_channels
        if self.from_input:
            self.kuramoto_channels = 3

        self.downsampling = nn.Conv2d(3, self.kuramoto_channels, 3, 1, padding=3 // 2, bias=False)
        # self.downsampling.weight.data = nn.Parameter(torch.load('gabors.pt', map_location='cuda')[:, :, 1:-1, 1:-1],
        #                                         requires_grad=True).repeat(self.kuramoto_channels//8, 3, 1, 1)

        # self.downsampling.weight = nn.Parameter(torch.load('../pt_utils/gabors.pt', map_location='cuda'), requires_grad=True)  # .repeat(1,3,1,1)

        self.bn1 = nn.BatchNorm3d(dimensions, eps=1e-03, track_running_stats=False)
        self.bn2 = nn.BatchNorm3d(dimensions * 2, eps=1e-03, track_running_stats=False)
        self.bn3 = nn.BatchNorm3d(dimensions * 2, eps=1e-03, track_running_stats=False)
        self.preproc = cf.ComplexConvolution3D(self.kuramoto_channels, dimensions, kernel_size=3, stride=(1, 2, 2),
                                               padding=3 // 2,
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

        x, y = torch.meshgrid([torch.linspace(-1, 1, self.h), torch.linspace(-1, 1, self.w)])
        dst = torch.sqrt(x ** 2 + y ** 2)
        g = torch.exp(-((dst - self.mean_r) ** 2 / (2.0 * self.std_r ** 2)))
        g = g.unsqueeze(0).unsqueeze(0).repeat(self.kuramoto_channels, self.kuramoto_channels, 1, 1)

        self.kernel_kuramoto = nn.Parameter(g, requires_grad=True)

        self.epsilon = nn.Parameter(torch.Tensor([self.epsilon]), requires_grad=False)
        self.lr_kuramoto = nn.Parameter(torch.Tensor([self.lr_kuramoto]), requires_grad=False)

        # self.phases_init = nn.Parameter((torch.rand([args.batch_size, 8, 32, 32]) * 2 * math.pi) - math.pi, requires_grad=False)

    def forward(self, input, masks, testmode=False, color=False, phases=None):
        loss_synch = 0
        if self.from_input:
            phases_frames = torch.zeros_like(input)
            for f in range(input.shape[2]):
                if f == 0:
                    for t in range(self.timesteps):
                        x = input[:, :, 0].to(torch.float)

                        if t == 0:
                            phases = (torch.rand_like(x) * 2 * math.pi) - math.pi

                        phases = phases + self.update_phases(x, phases, self.lr_kuramoto)
                    phases_frames[:, :, f] = phases
                    loss_synch += (self.synch_loss(phases, masks[:, :, 0])).mean()
                else:
                    phases = phases + self.update_phases(input[:, :, f], phases, self.lr_kuramoto_update)
                    phases_frames[:, :, f] = phases
                    # loss_synch += (self.synch_loss(phases, masks[:, :, f])).mean()
            z_in = cf.get_complex_number(input, phases_frames)
        else:
            phases_frames = torch.zeros((input.shape[0], self.kuramoto_channels, self.nb_frames, 32, 32)).to(
                input.device)
            amp_frames = torch.zeros((input.shape[0], self.kuramoto_channels, self.nb_frames, 32, 32)).to(input.device)
            for f in range(input.shape[2]):
                if f == 0:
                    for t in range(self.timesteps):
                        x = input[:, :, 0].to(torch.float)
                        amp = torch.relu(self.downsampling(x))
                        if t == 0:
                            phases = (torch.rand_like(amp) * 2 * math.pi) - math.pi

                        phases = phases + self.update_phases(amp, phases, self.lr_kuramoto)
                    phases_frames[:, :, f] = phases
                    amp_frames[:, :, f] = amp
                    loss_synch += (self.synch_loss(phases, masks[:, :, f])).mean()
                else:
                    x = input[:, :, f].to(torch.float)

                    amp = torch.relu(self.downsampling(x))

                    phases = phases + self.update_phases(amp, phases, self.lr_kuramoto)
                    phases_frames[:, :, f] = phases
                    amp_frames[:, :, f] = amp
                    loss_synch += (self.synch_loss(phases, masks[:, :, f])).mean()

            z_in = cf.get_complex_number(amp_frames, phases_frames)
        loss_synch = loss_synch / input.shape[2]

        x = self.preproc(z_in)
        x = cf.apply_activation_function(x.abs(), cf.stable_angle_2(x), self.bn1)
        x = self.conv1(x)
        x = cf.apply_activation_function(x.abs(), cf.stable_angle_2(x), self.bn2)
        x = self.conv2(x)
        x = cf.apply_activation_function(x.abs(), cf.stable_angle_2(x), self.bn3)

        x_shape = x.shape
        x, complex_output = self.readout(x.reshape(x_shape[0], -1))
        jv_penalty = torch.tensor([1]).float().cuda()
        if testmode: return x, None, None
        return (x, loss_synch), jv_penalty

    def update_phases(self, amp, phases, lr):
        # lr_kuramoto = torch.sigmoid(self.lr_kuramoto)
        # epsilon = torch.sigmoid(self.epsilon)
        b = torch.tanh(amp)

        B_cos = torch.cos(phases) * b
        B_sin = torch.sin(phases) * b

        C_cos = torch.nn.functional.conv2d(B_cos, self.kernel_kuramoto, padding="same")
        C_sin = torch.nn.functional.conv2d(B_sin, self.kernel_kuramoto, padding="same")

        S_cos = torch.sum(B_cos, dim=(1, 2, 3))[:, None, None, None]
        S_sin = torch.sum(B_sin, dim=(1, 2, 3))[:, None, None, None]

        phases_update = torch.cos(phases) * (C_sin - self.epsilon * S_sin) - torch.sin(phases) * (
                C_cos - self.epsilon * S_cos)
        final_phases = lr * phases_update

        return final_phases

    def synch_loss(self, phases, masks):
        # phases = phases + math.pi
        real = torch.sin(phases)
        imag = torch.cos(phases)
        phases = torch.atan2(real, imag)
        new_masks = masks[:, 1:]
        masks = new_masks.unsqueeze(2)
        num_groups = masks.shape[1]
        group_size = masks.sum((3, 4))
        group_size = torch.where(group_size == 0, torch.ones_like(group_size), group_size)

        # Loss is at least as large as the maxima of each individual loss (total desynchrony + total synchrony)
        loss_bound = 1 + .5 * num_groups * (1. /
                                            np.arange(1, num_groups + 1) ** 2)[:int(num_groups / 2.)].sum()

        # Consider only the phases with active amplitude
        active_phases = phases

        # Calculate global order within each group

        masked_phases = active_phases.unsqueeze(1) * masks.repeat(1, 1, self.kuramoto_channels, 1, 1)

        xx = torch.where(masks.bool(), torch.cos(masked_phases), torch.zeros_like(masked_phases))
        yy = torch.where(masks.bool(), torch.sin(masked_phases), torch.zeros_like(masked_phases))
        go = torch.sqrt((xx.sum((3, 4))) ** 2 + (yy.sum((3, 4))) ** 2) / group_size
        synch = 1 - go.mean(-1).sum(-1) / num_groups

        # Average angle within a group
        mean_angles = torch.atan2(yy.sum((3, 4)).mean(-1), xx.sum((3, 4)).mean(-1))

        # Calculate desynchrony between average group phases
        desynch = 0
        for m in np.arange(1, int(np.floor(num_groups / 2.)) + 1):
            #         K_m = 1 if m < int(np.floor(num_groups/2.)) + 1 else -1 # This is specified in Eq 36 of the cited paper and may have an effect on the values of the minimum though not its location
            desynch += (1.0 / (2 * num_groups * m ** 2)) * (
                    torch.cos(m * mean_angles).sum(-1) ** 2 + torch.sin(m * mean_angles).sum(-1) ** 2)

        # Total loss is average of invidual losses, averaged over time
        loss = (synch + desynch) / loss_bound

        return loss.mean(dim=-1)
