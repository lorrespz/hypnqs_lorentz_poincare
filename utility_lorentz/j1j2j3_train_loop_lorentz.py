# This is part of the work arxiv:2607.24337 by HL Dao. 
# This code defines the J1J2J3 train loop 

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import random
import sys
from math import ceil
from j1j2j3_wf_lorentz import *

# Check GPU
print("GPU Available: ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading Functions --------------------------
def J1J2J3MatrixElements(J1,J2,J3,Bz,sigmap, sigmaH, matrixelements, periodic = False, Marshall_sign = False):
    N=len(Bz)
    #the diagonal part is simply the sum of all Sz-Sz interactions plus a B field
    diag=np.dot(np.float32(sigmap)-0.5,Bz)

    num = 0 #Number of basis elements
    if periodic:
        limit = N
    else:
        limit = N-1

    if periodic:
        limit2 = N
    else:
        limit2 = N-2

    if periodic:
        limit3 = N
    else:
        limit3 = N-3

    for site in range(limit):
        if sigmap[site]!=sigmap[(site+1)%N]: #if the two neighouring spins are opposite
            diag-=0.25*J1[site] #add a negative energy contribution
        else:
            diag+=0.25*J1[site]

    for site in range(limit2):
        if J2[site] != 0.0:
            if sigmap[site]!=sigmap[(site+2)%N]: #if the two second neighouring spins are opposite
                diag-=0.25*J2[site] #add a negative energy contribution
            else:
                diag+=0.25*J2[site]

    for site in range(limit3):
        if J3[site] != 0.0:
            if sigmap[site]!=sigmap[(site+3)%N]: #if the two third neighouring spins are opposite
                diag-=0.25*J3[site] #add a negative energy contribution
            else:
                diag+=0.25*J3[site]

    matrixelements[num] = diag #add the diagonal part to the matrix elements
    sig = np.copy(sigmap)
    sigmaH[num] = sig

    num += 1

    #off-diagonal part:
    for site in range(limit):
        if J1[site] != 0.0:
          if sigmap[site]!=sigmap[(site+1)%N]:
              sig=np.copy(sigmap)
              sig[site]=sig[(site+1)%N] #Make the two neighbouring spins equal.
              sig[(site+1)%N]=sigmap[site]
              sigmaH[num] = sig #The last three lines are meant to flip the two neighbouring spins (that the effect of applying J+ and J-)
              if Marshall_sign:
                  matrixelements[num] = -J1[site]/2
              else:
                  matrixelements[num] = +J1[site]/2
              num += 1

    for site in range(limit2):
      if J2[site] != 0.0:
        if sigmap[site]!=sigmap[(site+2)%N]:
            sig=np.copy(sigmap)
            sig[site]=sig[(site+2)%N] #Make the two next-neighbouring spins equal.
            sig[(site+2)%N]=sigmap[site]
            sigmaH[num] = sig #The last three lines are meant to flip the two next-neighbouring spins (that the effect of applying J+ and J-)
            matrixelements[num] = +J2[site]/2
            num += 1

    for site in range(limit3):
      if J3[site] != 0.0:
        if sigmap[site]!=sigmap[(site+3)%N]:
            sig=np.copy(sigmap)
            sig[site]=sig[(site+3)%N] #Make the two next-next-neighbouring spins equal.
            sig[(site+3)%N]=sigmap[site]
            sigmaH[num] = sig #The last three lines are meant to flip the two next-neighbouring spins (that the effect of applying J+ and J-)
            matrixelements[num] = +J3[site]/2
            num += 1
    return num

def J1J2J3Slices(J1, J2, J3, Bz, sigmasp, sigmas, H, sigmaH, matrixelements, Marshall_sign):
    slices=[]
    sigmas_length = 0

    for n in range(sigmasp.shape[0]):
        sigmap=sigmasp[n,:]
        num = J1J2J3MatrixElements(J1,J2,J3,Bz,sigmap, sigmaH, matrixelements, Marshall_sign)
        #note that sigmas[0,:]==sigmap, matrixelements and sigmaH are updated
        slices.append(slice(sigmas_length,sigmas_length + num))
        s = slices[n]

        H[s] = matrixelements[:num]
        sigmas[s] = sigmaH[:num]

        sigmas_length += num #Increasing the length of matrix elements sigmas

    return slices, sigmas_length
#---------------------------------------------------------------------------------------
def J1J2J3_local_energies(wf, N,J1,J2,J3,Bz,numsamples, samples, Marshall_sign,log_amplitudes=None):
    # If samples is a torch tensor, move to CPU and convert to numpy for logic
    if torch.is_tensor(samples):
        samples_np = samples.detach().cpu().numpy()
    else:
        samples_np = samples

    # 1. Generate all connected states using the NumPy logic
    #Array to store all the diagonal and non diagonal sigmas for all the samples
    sigmas = np.zeros((3 * N * numsamples, N), dtype=np.int32)
    #Array to store all the diagonal and non diagonal matrix elements for all the samples
    H = np.zeros(3 * N * numsamples, dtype=np.float32)
    #Array to store all the diagonal and non diagonal sigmas for each sample sigma
    sigmaH = np.zeros((3 * N, N), dtype=np.int32)
    #Array to store all the diagonal and non diagonal matrix elements for each sample sigma
    matrixelements = np.zeros(3 * N, dtype=np.float32)

    slices, len_sigmas = J1J2J3Slices(J1, J2, J3, Bz, samples_np, sigmas, H, sigmaH, matrixelements, Marshall_sign)

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
        local_energies[n] = np.sum(H[s] * np.exp(log_amps_connected[s] - log_amplitudes_np[n]))

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
def train_step(wf, N, numsamples, J1, J2, J3, Bz, Marshall_sign, opt_eucl, opt_hyp):
    wf.model.train()
    
    try:
        # 1. Generate samples and amplitudes
        tsamp = wf.sample(numsamples) 
        log_amplitudes_ = wf.log_amplitude(tsamp)
        
        # 2. Local Energy
        Eloc = J1J2J3_local_energies(wf, N, J1, J2, J3, Bz, numsamples, tsamp, Marshall_sign, log_amplitudes=log_amplitudes_)
        
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
def run_J1J2J3(wf, numsteps, systemsize, var_tol, J1_, J2_, J3_, Marshall_sign,
                   numsamples, lr1, lr2, seed=111, fname='results'):
        
    random.seed(seed)  
    np.random.seed(seed) 
    torch.manual_seed(seed)

    J1=+J1_*np.ones(systemsize) # nearest neighbours couplings
    J2=+J2_*np.ones(systemsize) # next-nearest neighbours couplings
    J3=+J3_*np.ones(systemsize) # next-next-nearest neighbours couplings
    Bz=+0.0*np.ones(systemsize) # magnetic field along z

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
    sched_eucl = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_eucl, mode='min', factor=0.5, patience=40, verbose=True)
    sched_hyp = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_hyp, mode='min', factor=0.5, patience=40, verbose=True)
      
    for step in range(numsteps):
        cost, E, wfmodel = train_step(wf, systemsize, numsamples, J1, J2,J3, Bz, Marshall_sign, opt_eucl, opt_hyp)                 
        meanE = np.mean(E)
        varE = np.var(E)
        meanEnergy.append(meanE)
        varEnergy.append(varE)
        # Update the schedulers based on the mean energy
        sched_eucl.step(meanE)
        sched_hyp.step(meanE)
        # Decay of the temperature scaling inside 'sample'
        # When tau=1.0, the model has no temperature
        wf.model.current_tau = max(1.0, 1.05 - 0.0001 * step)
        np.save(f'{fname}/N{systemsize}_J1={J1_}|J2={J2_}|J3={J3_}_{wf.name}_ns={numsamples}_Ms{Marshall_sign}_meanE.npy', meanEnergy)
        np.save(f'{fname}/N{systemsize}_J1={J1_}|J2={J2_}|J3={J3_}_{wf.name}_ns={numsamples}_Ms{Marshall_sign}_varE.npy', varEnergy)

        if step == 0:
            best_E = meanE
            best_E_list.append(best_E)
            patience = 0
        elif np.real(meanE) < min(np.real(best_E_list)) and np.imag(meanE) <0.09 and np.abs(varE) < var_tol:
            best_E = meanE
            best_E_list.append(meanE)
            torch.save(wfmodel.state_dict(), f'{fname}/N{systemsize}_J1={J1_}|J2={J2_}|J3={J3_}_{wf.name}_ns={numsamples}_Ms{Marshall_sign}_checkpoint.pt')
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
