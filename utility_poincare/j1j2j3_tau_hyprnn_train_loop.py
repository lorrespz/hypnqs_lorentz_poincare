# This is part of the work arxiv:2607.24337 by HL Dao. 
# This code defines the train loop for J1J2J3 model, with the NQS being either Euclidean or Poincare RNN/GRU NQS.
# The `sample' method contains the temperature scaling parameter `tau'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import random
import sys
from math import ceil
# call the file where sample has tau
from j1j2j3_tau_hyprnn_wf import *
from hyp_rsgd_torch import RSGD 

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
def train_step(wf, numsamples, input_dim, J1, J2,J3, Bz, Marshall_sign, opt_eucl, opt_hyp):
    wf.model.train()
    opt_eucl.zero_grad()
    if opt_hyp: opt_hyp.zero_grad()
    
    # Generate samples
    tsamp = wf.sample(numsamples) 
    # Forward pass: log_amplitudes are needed for the cost function anyway
    log_amplitudes_ = wf.log_amplitude(tsamp)
    
    N = tsamp.shape[1]
    
    # Pass the precalculated log_amplitudes_ into the energy function
    Eloc = J1J2J3_local_energies(wf, N, J1, J2, J3, Bz, numsamples, tsamp, Marshall_sign, log_amplitudes=log_amplitudes_)
    
    loss_value = cost_fn(Eloc, log_amplitudes_)
    loss_value.backward()
    
    #for r_max = 0.7, 0.78, 0.82, etc.
    torch.nn.utils.clip_grad_norm_(wf.model.parameters(), max_norm=0.5)
    # for r_max = 0.95: 0.1 might be too small
    #torch.nn.utils.clip_grad_norm_(wf.model.parameters(), max_norm=0.1)
    
    opt_eucl.step()
    if opt_hyp: opt_hyp.step()
    return loss_value.item(), Eloc, wf.model


# ---------------------------------------- Running VMC with RNNs for J1J2J3 Model ------------------------------------
def run_J1J2J3(wf, numsteps, systemsize, var_tol, J1_ , J2_, J3_, Marshall_sign,
                   numsamples,  lr1, lr2, seed = 111, fname = 'results'):
    
    J1=+J1_*np.ones(systemsize) # nearest neighbours couplings
    J2=+J2_*np.ones(systemsize) # next-nearest neighbours couplings
    J3=+J3_*np.ones(systemsize) # next-next-nearest neighbours couplings
    Bz=+0.0*np.ones(systemsize) # magnetic field along z
    
    #Seeding
    random.seed(seed)  
    np.random.seed(seed) 
    torch.manual_seed(seed)
    
    input_dim = 2
    wf.model.to(device)

    # Filter parameters for different optimizers
    eucl_vars, hyp_vars = wf.get_manifold_parameters()
    opt_eucl = torch.optim.Adam(eucl_vars, lr=lr1)
    sched_eucl = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_eucl, mode='min', factor=0.5, patience=40, verbose=True)
    opt_hyp = None
    if hyp_vars:
        opt_hyp = RSGD(params=hyp_vars, lr=lr2, c_val=1.0, hyp_opt='rsgd')
        sched_hyp = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_hyp, mode='min', factor=0.5, patience=40, verbose=True)

    meanEnergy, varEnergy, best_E_list = [], [], []
    max_patience = 200
    patience = 0

    if not os.path.exists(fname):
        os.makedirs(fname)

    for step in range(numsteps):
        cost, E, wfmodel = train_step(wf, numsamples, input_dim, J1, J2,J3, Bz, Marshall_sign, opt_eucl, opt_hyp)
        meanE = np.mean(E)
        varE = np.var(E)
        meanEnergy.append(meanE)
        varEnergy.append(varE)
        # Update the schedulers based on the mean energy
        sched_eucl.step(meanE)
        if hyp_vars:
            sched_hyp.step(meanE)
        # Decay of the temperature scaling inside 'sample'
        # When tau=1.0, the model has no temperature
        # tau reduces to 1.0 when number of steps = 500
        wf.model.current_tau = max(1.0, 1.05 - 0.0001 * step)

        np.save(f'{fname}/N{systemsize}_J1={J1_}|J2={J2_}|J3={J3_}_{wf.name}_rmax={wf.r_max}_ns={numsamples}_Ms{Marshall_sign}_meanE.npy',meanEnergy)
        np.save(f'{fname}/N{systemsize}_J1={J1_}|J2={J2_}|J3={J3_}_{wf.name}_rmax={wf.r_max}_ns={numsamples}_Ms{Marshall_sign}_varE.npy',varEnergy)
        
        if step==0:
            best_E = meanE
            best_E_list.append(best_E)
            patience = 0
        elif np.real(meanE) < min(np.real(best_E_list)) and np.imag(meanE) <0.09 and np.abs(varE) < var_tol:
            best_E = meanE
            best_E_list.append(meanE)
            torch.save(wfmodel.state_dict(), f'{fname}/N{systemsize}_J1={J1_}|J2={J2_}|J3={J3_}_{wf.name}_rmax={wf.r_max}_ns={numsamples}_Ms{Marshall_sign}_checkpoint.pt')
            print(f'Best model saved at epoch {step} with best E={meanE:.5f}, varE={varE:.5f}')    
            patience =0
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping at step {step} - no improvement for {max_patience} steps.")
                break
        r_val_list = []
        if step % 10 == 0:        
            curr_lr_e = opt_eucl.param_groups[0]['lr']
            if hyp_vars:
                # Access the buffer from the model's RNN layer
                r_val = wf.model.rnn.current_max_r.item() 
                r_val_list.append(r_val)
                curr_lr_h = opt_hyp.param_groups[0]['lr']
                # save the list of rmax
                np.save(f'{fname}/N{systemsize}_J1={J1_}|J2={J2_}|J3={J3_}_{wf.name}_rmax={wf.r_max}_ns{numsamples}_Ms{Marshall_sign}_rmax.npy',r_val_list)
                print(f'step: {step}, loss: {cost:.5f}, mean energy: {meanE:.5f}, varE: {varE:.5f}| Max Radius: {r_val:.4f} | Hyp LR: {curr_lr_h:.2e}| LR: {curr_lr_e:.2e}|tau={wf.model.current_tau}')
            else:
                print(f'step: {step}, loss: {cost:.5f}, mean energy: {meanE:.5f}, varE: {varE:.5f}, |LR: {curr_lr_e:.2e}|tau={wf.model.current_tau}')
        
    return meanEnergy, varEnergy
 
