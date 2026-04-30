# This is part of the work arxiv:2607.24337 by HL Dao. 
# This code defines the Lorentzwavefunction class for J1J2J3 model which combines the LorentzRNN/GRU with Dense layers.
# Main functions include `sample' and 'log_probability'

import torch
import torch.nn as nn
import torch.nn.functional as F
#this file contains the tangent space update for LorentzGRU
#from j1j2j3_definitions_tangent_update import *
#This file contains the manifold update for LorentzGRU
from j1j2j3_definitions_manifold_update import *
import numpy as np
import random

def sqsoftmax(inputs):
    return torch.sqrt(F.softmax(inputs, dim=-1))

def softsign_ (inputs):
    return np.pi * torch.tanh(inputs) 

def heavyside(inputs):
    # sign = tf.sign(tf.sign(inputs) + 0.1 ) 
    # return 0.5*(sign+1.0)
    sign = torch.sign(torch.sign(inputs) + 0.1)
    return 0.5 * (sign + 1.0)

class wfModel(nn.Module):
    def __init__(self, rnn_layer, dense_amp, dense_phase):
        super(wfModel, self).__init__()
        self.rnn = rnn_layer
        self.dense_a = dense_amp
        self.dense_p = dense_phase
        self.c = rnn_layer.k
        self.manifold = hc_math.Lorentz()

    def forward(self, inputs, rnn_state, compute_phase):        
        rnn_output, rnn_state = self.rnn(inputs, rnn_state) 

        # To avoid rnn_output from drifting off the manifold
        rnn_output =project_lorentz_manual(rnn_output, k=1.0)

        rnn_output_eucl = hc_math.logmap0(rnn_output, k=self.rnn.k_tensor,is_tan_normalize=False)
        output_a = self.dense_a(rnn_output_eucl) 
        if not compute_phase:   
            return output_a, rnn_state
        if compute_phase:
            output_p = self.dense_p(rnn_output_eucl)
            return output_a, output_p, rnn_state

class Lorentzwavefunction(object):
    def __init__(self, systemsize, cell_type, units, spatial_clamp, seed=111):
        self.N = systemsize
        self.inputdim = 2
        self.dtype = torch.float32
        self.units = units
        self.current_tau = 1.05
        self.spatial_clamp = spatial_clamp
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")            

        if cell_type == 'LorentzRNN':
            self.cell = LorentzRNN(self.inputdim,self.units, self.spatial_clamp, 1.0)
            self.name = f'N={self.N}_{cell_type}_{self.units}'
        if cell_type == 'LorentzGRU':
            self.cell = LorentzGRU(self.inputdim,self.units, self.spatial_clamp, 1.0)
            self.name = f'{self.N}_{cell_type}_{self.units}'
        random.seed(seed) 
        np.random.seed(seed) 
        torch.manual_seed(seed)

        self.rnn = self.cell
        self.dense_ampl = nn.Linear(self.units, 2)
        self.dense_phase = nn.Linear(self.units, 2)
        self.model = wfModel(self.rnn, self.dense_ampl, self.dense_phase)

    def get_manifold_parameters(self):
        cell_eucl, cell_hyp = self.rnn.get_manifold_parameters()
        dense_params = list(self.dense_ampl.parameters())+list(self.dense_phase.parameters())
        return cell_eucl + dense_params, cell_hyp

    def sample_no_tau(self, numsamples):
        a = torch.zeros(numsamples, dtype=self.dtype)
        b = torch.zeros(numsamples, dtype=self.dtype)
        inputs_ampl = torch.stack([a, b], dim=1)
       
        self.outputdim = self.inputdim
        self.numsamples = numsamples

        samples = []
        rnn_state = hc_math.expmap0(torch.zeros((self.numsamples, self.units)), k=self.cell.k_tensor)

        for n in range(self.N):
            output_ampl, rnn_state = self.model.forward(inputs_ampl, rnn_state, compute_phase=False)
            output_ampl = sqsoftmax(output_ampl) 

            #Additional steps on output_ampl if n>=N/2
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

    def sample(self, numsamples):
        a = torch.zeros(numsamples, dtype=self.dtype)
        b = torch.zeros(numsamples, dtype=self.dtype)
        inputs_ampl = torch.stack([a, b], dim=1)
       
        self.outputdim = self.inputdim
        self.numsamples = numsamples

        samples = []
        rnn_state = hc_math.expmap0(torch.zeros((self.numsamples, self.units)), k=self.cell.k_tensor)

        for n in range(self.N):
            output_ampl, rnn_state = self.model.forward(inputs_ampl, rnn_state, compute_phase=False)
            
            # --- SAFETY CHECK 1: Catch Lorentz Explosions ---
            if torch.isnan(output_ampl).any() or torch.isinf(output_ampl).any():
                # Replace NaN/Inf with zeros so the softmax/normalization doesn't crash
                output_ampl = torch.nan_to_num(output_ampl, nan=0.0, posinf=20.0, neginf=-20.0)    
            output_ampl = sqsoftmax(output_ampl)

            if n >= self.N/2: 
                # Create a mask to enforce the strict number of spin up and spin down
                if len(samples) > 0:
                    num_up = torch.sum(torch.stack(samples, dim=1), dim=1).float()
                else:
                    num_up = torch.zeros(self.numsamples, dtype=self.dtype)
                        
                baseline = (self.N//2 - 1) * torch.ones(self.numsamples, dtype=self.dtype)
                num_down = n * torch.ones(self.numsamples, dtype=self.dtype) - num_up
                
                # --- SAFETY CHECK 2: Validate the Mask ---
                # Ensure activations are finite
                activations_up = torch.nan_to_num(heavyside(baseline - num_up), nan=0.0)
                activations_down = torch.nan_to_num(heavyside(baseline - num_down), nan=0.0)
                mask = torch.stack([activations_down, activations_up], dim=1)
                output_ampl = output_ampl * mask
                    
                # --- SAFETY CHECK 3: The "Dead End" Recovery ---
                # If the mask zeroed out BOTH options (sum is 0), force a valid choice
                norm_val = torch.norm(output_ampl, p=2, dim=1, keepdim=True)
                is_dead = (norm_val < 1e-10)
                    
                # If dead, provide a small epsilon to the mask so multinomial has something to pick
                output_ampl = torch.where(is_dead, mask + 1e-9, output_ampl)
                output_ampl = F.normalize(output_ampl, p=2, dim=1, eps=1e-30)

            # --- FINAL PREPARATION FOR MULTINOMIAL ---
            # 1. Apply Temperature Scaling (tau > 1.0 increases exploration)            
            # Use a tiny epsilon for the power base to ensure stability
            base = torch.clamp(output_ampl**2, min=1e-12)
            weights_sq = torch.pow(base, 1.0 / self.current_tau)
            
            # 2. Final safety guard: Replace NaNs/Infs just in case
            weights_sq = torch.nan_to_num(weights_sq, nan=1e-9, posinf=1.0, neginf=1e-9)

            # 3. Re-normalize to satisfy multinomial's sum constraint
            weights_sq = weights_sq / weights_sq.sum(dim=-1, keepdim=True)
            sample_temp = torch.multinomial(weights_sq, num_samples=1).view(-1)
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
        rnn_state = hc_math.expmap0(torch.zeros((self.numsamples, self.units)),k=self.rnn.k_tensor)

        for n in range(self.N):
            output_ampl, output_phase, rnn_state = self.model.forward(inputs_ampl, rnn_state, compute_phase=True)
            output_ampl = torch.nan_to_num(output_ampl, nan=0.0)
            output_ampl = sqsoftmax(output_ampl)
            output_phase = softsign_(output_phase)

            if n >= self.N/2:
                num_up = torch.sum(samples[:, :n], dim=1).float()
                baseline = (self.N//2 - 1) * torch.ones(self.numsamples, dtype=self.dtype)
                num_down = n * torch.ones(self.numsamples, dtype=self.dtype) - num_up
                activations_up = heavyside(baseline - num_up)
                activations_down = heavyside(baseline - num_down)
                mask=torch.stack([activations_down, activations_up], dim=1)

                output_ampl = output_ampl * mask
                output_ampl = F.normalize(output_ampl, p=2, dim=1, eps=1e-30)

            amplitude = torch.complex(output_ampl, torch.zeros_like(output_ampl)) * \
                        torch.exp(torch.complex(torch.zeros_like(output_phase), output_phase))
            amplitudes.append(amplitude)

            inputs_ampl = F.one_hot(samples[:, n].long(), num_classes=self.outputdim).float()

        amplitudes = torch.stack(amplitudes, dim=1)
        one_hot_samples = F.one_hot(samples.long(), num_classes=self.inputdim).float()

        inner_prod = torch.sum(amplitudes * torch.complex(one_hot_samples, torch.zeros_like(one_hot_samples)), dim=2)
        #Clamping
        inner_prod_abs = torch.abs(inner_prod)
        # Amplitude
        inner_prod_clamped = torch.clamp(inner_prod_abs, min=1e-12)
        # Phase
        # We use torch.angle() to keep the phase info intact
        phase = torch.angle(inner_prod)
        # Recombine for log(complex_number) = log(abs) + i*phase
        self.log_amplitudes = torch.sum(torch.log(inner_prod_clamped) + 1j * phase, dim=1)

        return self.log_amplitudes
