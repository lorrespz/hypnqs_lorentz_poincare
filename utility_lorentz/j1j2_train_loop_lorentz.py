# This is part of the work arxiv:2607.24337 by HL Dao. 
# This code defines the train loop for J1J2 model using Lorentz NQS.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import random
import sys
from math import ceil
from j1j2_wf_lorentz import *

# Check GPU
print("GPU Available: ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading Functions --------------------------
def J1J2MatrixElements(J1, J2, Bz, sigmap, sigmaH, matrixelements, periodic=False, Marshall_sign=False):
    N = len(Bz)
    # The diagonal part is simply the sum of all Sz-Sz interactions plus a B field
    diag = np.dot(np.float32(sigmap) - 0.5, Bz)

    num = 0 # Number of basis elements
    if periodic:
        limit = N
    else:
        limit = N - 1    
 
    if periodic:
        limit2 = N
    else:
        limit2 = N - 2     

    for site in range(limit):
        if sigmap[site] != sigmap[(site + 1) % N]: # if the two neighboring spins are opposite
            diag -= 0.25 * J1[site] # add a negative energy contribution
        else:
            diag += 0.25 * J1[site]
    
    for site in range(limit2):
        if J2[site] != 0.0:
            if sigmap[site] != sigmap[(site + 2) % N]: # if the two second neighboring spins are opposite
                diag -= 0.25 * J2[site] # add a negative energy contribution
            else:
                diag += 0.25 * J2[site]

    matrixelements[num] = diag # add the diagonal part to the matrix elements
    sig = np.copy(sigmap)
    sigmaH[num] = sig

    num += 1
    # off-diagonal part:
    for site in range(limit):
        if J1[site] != 0.0:
            if sigmap[site] != sigmap[(site + 1) % N]:
                sig = np.copy(sigmap)
                sig[site] = sig[(site + 1) % N] # Make the two neighboring spins equal.
                sig[(site + 1) % N] = sigmap[site]
                sigmaH[num] = sig 

                if Marshall_sign:
                    matrixelements[num] = -J1[site] / 2
                else:
                    matrixelements[num] = +J1[site] / 2

                num += 1

    for site in range(limit2):
        if J2[site] != 0.0:
            if sigmap[site] != sigmap[(site + 2) % N]:
                sig = np.copy(sigmap)
                sig[site] = sig[(site + 2) % N] # Make the two next-neighboring spins equal.
                sig[(site + 2) % N] = sigmap[site]
                sigmaH[num] = sig 
                matrixelements[num] = +J2[site] / 2
                num += 1
    return num

def J1J2Slices(J1, J2, Bz, sigmasp, sigmas, H, sigmaH, matrixelements, Marshall_sign):
    slices = []
    sigmas_length = 0

    for n in range(sigmasp.shape[0]):
        sigmap = sigmasp[n, :]
        num = J1J2MatrixElements(J1, J2, Bz, sigmap, sigmaH, matrixelements, Marshall_sign)
        slices.append(slice(sigmas_length, sigmas_length + num))
        s = slices[n]
        H[s] = matrixelements[:num]
        sigmas[s] = sigmaH[:num]
        sigmas_length += num 

    return slices, sigmas_length

#---------------------------------------------------------------------------------------
def J1J2_local_energies_old(wf, N, J1, J2, Bz, numsamples, samples, Marshall_sign):

    local_energies = np.zeros(numsamples, dtype=np.complex64)
    log_amplitudes = np.zeros(2 * N * numsamples, dtype=np.complex64) 
    sigmas = np.zeros((2 * N * numsamples, N), dtype=np.int32)
    H = np.zeros(2 * N * numsamples, dtype=np.float32)
    sigmaH = np.zeros((2 * N, N), dtype=np.int32) 
    matrixelements = np.zeros(2 * N, dtype=np.float32)

    slices, len_sigmas = J1J2Slices(J1, J2, Bz, samples, sigmas, H, sigmaH, matrixelements, Marshall_sign)

    steps = ceil(len_sigmas / numsamples)

    wf.model.eval()
    with torch.no_grad():
        for i in range(steps):
            start = (i * len_sigmas) // steps
            end = ((i + 1) * len_sigmas) // steps if i < steps - 1 else len_sigmas
            cut = slice(start, end)
            
            # Convert to torch tensor for the model
            sigmas_tensor = torch.from_numpy(sigmas[cut]).to(device)
            log_amps = wf.log_amplitude(sigmas_tensor)    
            log_amplitudes[cut] = log_amps.cpu().numpy()

    for n in range(len(slices)):
        s = slices[n]
        local_energies[n] = H[s].dot(np.exp(log_amplitudes[s] - log_amplitudes[s][0]))

    return local_energies
#---------------------------------------------------------------------------------------
def J1J2_local_energies(wf, N, J1, J2, Bz, numsamples, samples, Marshall_sign, log_amplitudes=None):
    # If samples is a torch tensor, move to CPU and convert to numpy for logic
    if torch.is_tensor(samples):
        samples_np = samples.detach().cpu().numpy()
    else:
        samples_np = samples

    # 1. Generate all connected states using the NumPy logic
    sigmas = np.zeros((2 * N * numsamples, N), dtype=np.int32)
    H = np.zeros(2 * N * numsamples, dtype=np.float32)
    sigmaH = np.zeros((2 * N, N), dtype=np.int32) 
    matrixelements = np.zeros(2 * N, dtype=np.float32)

    slices, len_sigmas = J1J2Slices(J1, J2, Bz, samples_np, sigmas, H, sigmaH, matrixelements, Marshall_sign)

    # 2. Handle log_amplitudes for the base samples
    if log_amplitudes is None:
        wf.model.eval()
        with torch.no_grad():
            # Convert numpy samples back to torch for the model
            samples_torch = torch.from_numpy(samples_np).to(device)
            log_amplitudes = wf.log_amplitude(samples_torch)
    
    # Ensure log_amplitudes is a numpy array for the final energy sum
    if torch.is_tensor(log_amplitudes):
        log_amplitudes_np = log_amplitudes.detach().cpu().numpy()
    else:
        log_amplitudes_np = log_amplitudes

    # 3. Calculate log_amplitudes for ALL connected states (including samples)
    # We slice sigmas to only include the populated rows to save CPU time
    sigmas_torch = torch.from_numpy(sigmas[:len_sigmas]).to(device)
    
    wf.model.eval()
    with torch.no_grad():
        # Running the whole batch through the RNN/Model at once is the most efficient CPU path
        log_amps_connected = wf.log_amplitude(sigmas_torch).detach().cpu().numpy()

    # 4. Final local energy summation in NumPy
    local_energies = np.zeros(numsamples, dtype=np.complex64)
    for n in range(len(slices)):
        s = slices[n]
        # H[s] contains the matrix elements H_ss'
        # log_amps_connected[s] contains log(psi(s'))
        # log_amplitudes_np[n] contains log(psi(s))
        #local_energies[n] = np.sum(H[s] * np.exp(log_amps_connected[s] - log_amplitudes_np[n]))
        diff = log_amps_connected[s] - log_amplitudes_np[n]
        # Clip the exponent to avoid overflow/underflow
        diff = np.clip(diff, -30.0, 30.0) 
        ratio = np.exp(diff)
        local_energies[n] = np.sum(H[s] * ratio)

    return local_energies
#----------------------------------------------------------------------------------------------------
def cost_fn(Eloc, log_amplitudes_):
    # cost = 2*real(mean(conj(log_amps) * stop_grad(Eloc)) - conj(mean(log_amps)) * mean(stop_grad(Eloc)))
    # Eloc is numpy, log_amplitudes_ is torch tensor
    Eloc_torch = torch.from_numpy(Eloc).to(log_amplitudes_.device)
    
    term1 = torch.mean(torch.conj(log_amplitudes_) * Eloc_torch)
    term2 = torch.conj(torch.mean(log_amplitudes_)) * torch.mean(Eloc_torch)
    
    cost = 2 * torch.real(term1 - term2)
    return cost

#----------------------------------------------------------------------------------------------------
def train_step_old(wf, numsamples, J1, J2, Bz, Marshall_sign, opt_eucl, opt_hyp):
    wf.model.train()
    opt_eucl.zero_grad()
    opt_hyp.zero_grad()
    
    # Generate samples
    tsamp = wf.sample(numsamples) 
    # Forward pass: log_amplitudes are needed for the cost function anyway
    log_amplitudes_ = wf.log_amplitude(tsamp)
    
    N = tsamp.shape[1]

    # Pass the precalculated log_amplitudes_ into the energy function
    Eloc = J1J2_local_energies(wf, N, J1, J2, Bz, numsamples, tsamp, Marshall_sign, log_amplitudes=log_amplitudes_)
               
    loss_value = cost_fn(Eloc, log_amplitudes_)
    loss_value.backward()
    
    # --- Clipping Strategy ---
    # Euclidean weights can handle larger gradients
    torch.nn.utils.clip_grad_norm_(opt_eucl.param_groups[0]['params'], max_norm=1.0)
    # Hyperbolic biases are sensitive because of expmap (sinh/cosh)
    torch.nn.utils.clip_grad_norm_(opt_hyp.param_groups[0]['params'], max_norm=0.5)
    opt_eucl.step()
    opt_hyp.step()
    
    return loss_value.item(), Eloc, wf.model
#----------------------------------------------------------------------------------------------------
def train_step(wf, N, numsamples, J1, J2, Bz, Marshall_sign, opt_eucl, opt_hyp):
    wf.model.train()
    
    try:
        # 1. Generate samples and amplitudes
        tsamp = wf.sample(numsamples) 
        log_amplitudes_ = wf.log_amplitude(tsamp)
        
        # 2. Local Energy
        Eloc = J1J2_local_energies(wf, N, J1, J2, Bz, numsamples, tsamp, Marshall_sign, log_amplitudes=log_amplitudes_)
        
        # 3. Calculate Loss
        loss_value = cost_fn(Eloc, log_amplitudes_)
        
        # 4. Check for NaN BEFORE backward pass
        if torch.isnan(loss_value) or torch.isinf(loss_value):
            raise ValueError("NaN/Inf detected in loss!")

        # 5. Gradient computation
        opt_eucl.zero_grad()
        opt_hyp.zero_grad()
        loss_value.backward()
        
        # 6. Aggressive Clipping for Lorentz manifold stability
        # Euclidean weights can handle larger gradients
        torch.nn.utils.clip_grad_norm_(opt_eucl.param_groups[0]['params'], max_norm=1.0)
        # Hyperbolic biases are sensitive because of expmap (sinh/cosh)
        torch.nn.utils.clip_grad_norm_(opt_hyp.param_groups[0]['params'], max_norm=0.5)
        
        # 7. Update
        opt_eucl.step()
        opt_hyp.step()
        
        return loss_value.item(), Eloc, wf.model

    except Exception as e:
        print(f"Skipping update due to: {e}")
        # Before we clear the gradients, check which ones are NaN
        found_nan = False
        for name, param in wf.model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"-> Exploding gradient detected in: {name}")
                found_nan = True
        
        if not found_nan:
            print("-> No NaNs found in gradients; the explosion is in the Forward pass (e.g., loss calculation or manifold projection).")
            
        # Clean up
        opt_eucl.zero_grad()
        opt_hyp.zero_grad()
        
        import traceback; traceback.print_exc()
        return None, None, None
# ---------------- Running VMC with RNNs for J1J2 Model with hyperbolic parameters -------------------------------------
def run_J1J2(wf, numsteps, systemsize, var_tol, J1_=1.0, J2_=0.0, Marshall_sign=True, 
                   numsamples=500, lr1=1e-2, lr2=1e-3, seed=111, fname='results'):
        
    random.seed(seed)  
    np.random.seed(seed) 
    torch.manual_seed(seed)
    J1 = +J1_ * np.ones(systemsize)
    J2 = +J2_ * np.ones(systemsize)
    Bz = +0.0 * np.ones(systemsize)
    if not os.path.exists(fname):
        os.makedirs(fname)
    meanEnergy, varEnergy, best_E_list = [], [], []
    max_patience = 200
    patience = 0

    #Get parameters
    eucl_vars, hyp_vars = wf.get_manifold_parameters()

    # lr1 for weights (e.g., 1e-3)
    opt_eucl = torch.optim.Adam(eucl_vars, lr=lr1)    
    # lr2 for hyperbolic biases (e.g., 1e-4 or lr1/10)
    # We use standard Adam because these are D-dim tangent vectors (not (D+1)-dim Lorentz manifold vectors)
    opt_hyp = torch.optim.Adam(hyp_vars, lr=lr2)
    sched_eucl = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_eucl, mode='min', factor=0.5, patience=50, verbose=True)
    sched_hyp = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_hyp, mode='min', factor=0.5, patience=50, verbose=True)
      
    for step in range(numsteps):
        cost, E, wfmodel = train_step(wf, systemsize, numsamples, J1, J2, Bz, Marshall_sign, opt_eucl, opt_hyp) 
        if E is None:
            continue     # Skip the rest of the loop for this step  
            
        meanE = np.mean(E)
        varE = np.var(E)
        meanEnergy.append(meanE)
        varEnergy.append(varE)
        # Decay of the temperature scaling inside 'sample'
        wf.model.current_tau = max(1.0, 1.05 - 0.0001 * step)
        np.save(f'{fname}/N{systemsize}_J1={J1_}|J2={J2_}_{wf.name}_ns={numsamples}_Ms{Marshall_sign}_meanE.npy', meanEnergy)
        np.save(f'{fname}/N{systemsize}_J1={J1_}|J2={J2_}_{wf.name}_ns={numsamples}_Ms{Marshall_sign}_varE.npy', varEnergy)

        # Update the schedulers based on the mean energy
        sched_eucl.step(meanE)
        sched_hyp.step(meanE)
        
        if step == 0:
            best_E = meanE
            best_E_list.append(best_E)
            patience = 0
        elif np.real(meanE) < min(np.real(best_E_list)) and np.abs(varE) < var_tol:
            best_E = meanE
            best_E_list.append(meanE)
            torch.save(wfmodel.state_dict(), f'{fname}/N{systemsize}_J1={J1_}|J2={J2_}_{wf.name}_ns={numsamples}_Ms{Marshall_sign}_checkpoint.pt') 
            print(f'Best model saved at epoch {step} with best E={meanE:.5f}, varE={varE:.5f}')    
            patience =0
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping at step {step} - no improvement for {max_patience} steps.")
                break
        
        if step % 10 == 0:
            curr_lr_h = opt_hyp.param_groups[0]['lr']
            curr_lr_e = opt_eucl.param_groups[0]['lr']
            #access the buffer from RNN cell:
            max_h0 = wf.model.rnn.max_h0.item()
            max_h_spatial_norm = wf.model.rnn.max_spatial_norm.item()
            max_h_violation = wf.model.rnn.max_violation.item()
            print(f'step: {step}, loss: {cost:.5f}, mean energy: {meanE:.5f}|varE: {varE:.5f}| Hyp LR: {curr_lr_h:.2e}| LR: {curr_lr_e:.2e}| tau={wf.model.current_tau}')
            print(f'max_h0 = {max_h0} | max_h_spatial_norm = {max_h_spatial_norm}| max_h_violation = {max_h_violation}')
    return meanEnergy, varEnergy
