# This is part of the work arxiv:2607.24337 by HL Dao. 
# This code defines the NQS wavefunction (J1J2J3 model) using either Euclidean GRU/RNN or Poincare RNN/GRU combined with Dense layers.

import torch
import torch.nn as nn
import torch.nn.functional as F
from j1j2j3_poincare_definitions import *
import numpy as np
import random
from util_torch_poincare import *

def project_to_ball(x, eps=1e-5):
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    maxnorm = 1.0 - eps
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)

def sqsoftmax(inputs):
    return torch.sqrt(F.softmax(inputs, dim=-1))

def softsign_ (inputs):
    return np.pi * torch.tanh(inputs) # PyTorch equivalent of tf.nn.softsign is often replaced or used as F.softsign

def heavyside(inputs):
    # sign = tf.sign(tf.sign(inputs) + 0.1 ) 
    # return 0.5*(sign+1.0)
    sign = torch.sign(torch.sign(inputs) + 0.1)
    return 0.5 * (sign + 1.0)

class wfModel(nn.Module):
    def __init__(self, rnn_layer, dense_amp, dense_phase, is_hyp):
        super().__init__()
        self.rnn = rnn_layer
        self.dense_a = dense_amp
        self.dense_p = dense_phase
        self.hyp = is_hyp

    def forward(self, inputs, rnn_state, compute_phase):
        if not self.hyp: 
            rnn_output, rnn_state = self.rnn(inputs, rnn_state) 
        if self.hyp:
            rnn_outp, rnn_state = self.rnn(inputs, rnn_state)
            # Force the hidden state to stay strictly inside the ball
            rnn_outp = project_to_ball(rnn_outp)
            # Utilizing the hyperbolic log map to move to tangent space for dense layers
            rnn_output = th_log_map_zero(rnn_outp, 1.0)

        #if self.hyp:
        #    rnn_outp, rnn_state = self.rnn(inputs, rnn_state)
            # --- PROBE START ---
            # Before the log map, check the raw hidden state distribution
        #    if torch.rand(1) < 0.05: # Print once every 20 calls to avoid log spam
        #        print(f"GRU Output Min: {rnn_outp.min():.4f}, Max: {rnn_outp.max():.4f}, Mean: {rnn_outp.mean():.4f}")
            # --- PROBE END ---
        #    rnn_output = th_log_map_zero(rnn_outp, 1.0)

        output_a = self.dense_a(rnn_output) 
        if not compute_phase:   
            return output_a, rnn_state
        if compute_phase:
            output_p = self.dense_p(rnn_output)
            return output_a, output_p, rnn_state

class RNNwavefunction(object):
    def __init__(self, systemsize, cell_type, units, seed=111):
        self.N = systemsize 
        self.dtype = torch.float32
        self.units = units
        self.inputdim = 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        random.seed(seed) 
        np.random.seed(seed) 
        torch.manual_seed(seed)

        if cell_type == 'EuclRNN':
            self.rnn = EuclRNN(self.inputdim, self.units) 
            self.name = f'{cell_type}_{self.units}'
        if cell_type == 'EuclGRU':
            self.rnn = EuclGRU(self.inputdim, self.units) 
            self.name = f'{cell_type}_{self.units}'

        self.dense_ampl = nn.Linear(self.units, 2)
        self.dense_phase = nn.Linear(self.units, 2)
        self.model = wfModel(self.rnn, self.dense_ampl, self.dense_phase, is_hyp=False)

    def get_manifold_parameters(self):
        rnn_params = self.rnn.get_manifold_parameters()[0] 
        dense_params = list(self.dense_ampl.parameters())+ list(self.dense_phase.parameters())
        return rnn_params+dense_params, []

    def sample(self, numsamples):
        a = torch.zeros(numsamples, dtype=self.dtype)
        b = torch.zeros(numsamples, dtype=self.dtype)
        inputs_ampl = torch.stack([a, b], dim=1)

        self.outputdim = self.inputdim
        self.numsamples = numsamples

        samples = []
        rnn_state = torch.zeros((self.numsamples, self.units), dtype=self.dtype) 

        for n in range(self.N):
            output_ampl, rnn_state = self.model.forward(inputs_ampl, rnn_state, compute_phase=False)
            output_ampl = sqsoftmax(output_ampl)

            if n >= self.N/2: 
                if len(samples) > 0:
                    num_up = torch.sum(torch.stack(samples, dim=1), dim=1).float()
                else:
                    num_up = torch.zeros(self.numsamples, dtype=self.dtype)
                
                baseline = (self.N//2 - 1) * torch.ones(self.numsamples, dtype=self.dtype)
                num_down = n * torch.ones(self.numsamples, dtype=self.dtype) - num_up
                activations_up = heavyside(baseline - num_up)
                activations_down = heavyside(baseline - num_down)

                output_ampl = output_ampl * torch.stack([activations_down, activations_up], dim=1)
                output_ampl = F.normalize(output_ampl, p=2, dim=1, eps=1e-30)

            sample_temp = torch.multinomial(output_ampl**2, num_samples=1).view(-1)
            samples.append(sample_temp)
            inputs_ampl = F.one_hot(sample_temp, num_classes=self.outputdim).float()

        self.samples = torch.stack(samples, dim=1)
        return self.samples

    def log_amplitude(self, samples):
        self.outputdim = self.inputdim
        self.numsamples = samples.shape[0]
        
        a = torch.zeros(self.numsamples, dtype=self.dtype)
        b = torch.zeros(self.numsamples, dtype=self.dtype)
        inputs_ampl = torch.stack([a, b], dim=1)
        
        amplitudes = []
        rnn_state = torch.zeros((self.numsamples, self.units), dtype=self.dtype) 

        for n in range(self.N):
            output_ampl, output_phase, rnn_state = self.model.forward(inputs_ampl, rnn_state, compute_phase=True)
            output_ampl = sqsoftmax(output_ampl)
            output_phase = softsign_(output_phase)

            if n >= self.N/2:
                num_up = torch.sum(samples[:, :n], dim=1).float()
                baseline = (self.N//2 - 1) * torch.ones(self.numsamples, dtype=self.dtype)
                num_down = n * torch.ones(self.numsamples, dtype=self.dtype) - num_up
                activations_up = heavyside(baseline - num_up)
                activations_down = heavyside(baseline - num_down)

                output_ampl = output_ampl * torch.stack([activations_down, activations_up], dim=1)
                output_ampl = F.normalize(output_ampl, p=2, dim=1, eps=1e-30)

            # Complex amplitude construction
            amplitude = torch.complex(output_ampl, torch.zeros_like(output_ampl)) * \
                        torch.exp(torch.complex(torch.zeros_like(output_phase), output_phase))
            amplitudes.append(amplitude)

            inputs_ampl = F.one_hot(samples[:, n].long(), num_classes=self.outputdim).float()

        amplitudes = torch.stack(amplitudes, dim=1) 
        one_hot_samples = F.one_hot(samples.long(), num_classes=self.inputdim).float()

        # Log-amplitude calculation
        inner_prod = torch.sum(amplitudes * torch.complex(one_hot_samples, torch.zeros_like(one_hot_samples)), dim=2)
        self.log_amplitudes = torch.sum(torch.log(inner_prod), dim=1)

        return self.log_amplitudes

class RNNwavefunction_hyp(object):
    def __init__(self, systemsize, cell_type,  bias_geom, hyp_non_lin, units, r_max,seed=111):
        self.N = systemsize
        self.inputdim = 2
        self.dtype = torch.float32
        self.units = units
        self.inputs_geom = 'eucl'
        self.bias_geom = bias_geom
        self.h_non_lin = hyp_non_lin
        self.r_max = r_max
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        random.seed(seed) 
        np.random.seed(seed) 
        torch.manual_seed(seed)

        if cell_type == 'HypRNN':
            self.rnn = HypRNN(self.inputdim, self.units, self.r_max, self.inputs_geom, self.bias_geom, 1.0, self.h_non_lin)
            self.name = f'{cell_type}_{self.units}_{self.h_non_lin}_{self.bias_geom}'
        if cell_type == 'HypGRU':
            self.rnn = HypGRU(self.inputdim, self.units, self.r_max, self.inputs_geom, self.bias_geom, 1.0, self.h_non_lin)
            self.name = f'{cell_type}_{self.units}_{self.h_non_lin}_{self.bias_geom}'

        self.dense_ampl = nn.Linear(self.units, 2)
        self.dense_phase = nn.Linear(self.units, 2)
        self.model = wfModel(self.rnn, self.dense_ampl, self.dense_phase, is_hyp=True)

    def get_manifold_parameters(self):
        cell_eucl, cell_hyp = self.rnn.get_manifold_parameters()
        dense_params = list(self.dense_ampl.parameters())+list(self.dense_phase.parameters())
        return cell_eucl + dense_params, cell_hyp

    def sample(self, numsamples):
        a = torch.zeros(numsamples, dtype=self.dtype)
        b = torch.zeros(numsamples, dtype=self.dtype)
        inputs_ampl = torch.stack([a, b], dim=1)
       
        self.outputdim = self.inputdim
        self.numsamples = numsamples

        samples = []
        rnn_state = torch.zeros((self.numsamples, self.units), dtype=self.dtype) 

        for n in range(self.N):
            output_ampl, rnn_state = self.model.forward(inputs_ampl, rnn_state, compute_phase=False)
            output_ampl = sqsoftmax(output_ampl)

            if n >= self.N/2: 
                if len(samples) > 0:
                    num_up = torch.sum(torch.stack(samples, dim=1), dim=1).float()
                else:
                    num_up = torch.zeros(self.numsamples, dtype=self.dtype)
                    
                baseline = (self.N//2 - 1) * torch.ones(self.numsamples, dtype=self.dtype)
                num_down = n * torch.ones(self.numsamples, dtype=self.dtype) - num_up
                activations_up = heavyside(baseline - num_up)
                activations_down = heavyside(baseline - num_down)

                output_ampl = output_ampl * torch.stack([activations_down, activations_up], dim=1)
                output_ampl = F.normalize(output_ampl, p=2, dim=1, eps=1e-30)

            sample_temp = torch.multinomial(output_ampl**2, num_samples=1).view(-1)
            samples.append(sample_temp)
            inputs_ampl = F.one_hot(sample_temp, num_classes=self.outputdim).float()

        self.samples = torch.stack(samples, dim=1)
        return self.samples

    def log_amplitude(self, samples):
        self.outputdim = self.inputdim
        self.numsamples = samples.shape[0]

        a = torch.zeros(self.numsamples, dtype=self.dtype)
        b = torch.zeros(self.numsamples, dtype=self.dtype)
        inputs_ampl = torch.stack([a, b], dim=1)
        
        amplitudes = []
        rnn_state = torch.zeros((self.numsamples, self.units), dtype=self.dtype) 

        for n in range(self.N):
            output_ampl, output_phase, rnn_state = self.model.forward(inputs_ampl, rnn_state, compute_phase=True)
            output_ampl = sqsoftmax(output_ampl)
            output_phase = softsign_(output_phase)

            if n >= self.N/2:
                num_up = torch.sum(samples[:, :n], dim=1).float()
                baseline = (self.N//2 - 1) * torch.ones(self.numsamples, dtype=self.dtype)
                num_down = n * torch.ones(self.numsamples, dtype=self.dtype) - num_up
                activations_up = heavyside(baseline - num_up)
                activations_down = heavyside(baseline - num_down)

                output_ampl = output_ampl * torch.stack([activations_down, activations_up], dim=1)
                output_ampl = F.normalize(output_ampl, p=2, dim=1, eps=1e-30)

            amplitude = torch.complex(output_ampl, torch.zeros_like(output_ampl)) * \
                        torch.exp(torch.complex(torch.zeros_like(output_phase), output_phase))
            amplitudes.append(amplitude)

            inputs_ampl = F.one_hot(samples[:, n].long(), num_classes=self.outputdim).float()

        amplitudes = torch.stack(amplitudes, dim=1)
        one_hot_samples = F.one_hot(samples.long(), num_classes=self.inputdim).float()

        inner_prod = torch.sum(amplitudes * torch.complex(one_hot_samples, torch.zeros_like(one_hot_samples)), dim=2)
        self.log_amplitudes = torch.sum(torch.log(inner_prod), dim=1)

        return self.log_amplitudes
