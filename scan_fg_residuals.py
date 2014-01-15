#!/usr/bin/python
"""
Scan a range of values of the FG subtraction efficiency, epsilon. Output 1D 
marginal constraints for several parameters as a fn. of epsilon.
"""
import numpy as np
import pylab as P
import baofisher
import matplotlib.patches
from units import *
from mpi4py import MPI
import experiments

comm = MPI.COMM_WORLD
myid = comm.Get_rank()
size = comm.Get_size()

# Choose fiducial values for parameters (f(z) is defined later)
fiducial = {
  'aperp':     1.,
  'apar':      1.,
  'bHI0':      0.702,
  'A':         1.,
  'sigma_nl':  7. #14. #0.5
}

# Load cosmology and experimental settings
cosmo = experiments.cosmo
expt = experiments.exptA
#[experiments.exptA, experiments.exptB, experiments.exptC, experiments.exptD, experiments.exptE]

# Define redshift bins
zs, zc = baofisher.zbins_equal_spaced(expt, dz=0.1) # FIXME: dz=0.1
#zs, zc = baofisher.zbins_const_dr(expt, cosmo, bins=14)

# Precompute cosmological functions and derivs.
camb_matterpower = "/home/phil/oslo/iswfunction/cosmomc/camb/testX_matterpower.dat"
cosmo_fns, cosmo = baofisher.precompute_for_fisher(cosmo, camb_matterpower)
eos_derivs = baofisher.eos_fisher_matrix_derivs(cosmo)

# Loop through epsilon values
sigmas = []
epsilon_vals = np.logspace(-3., -8., 12)
for epsilon in epsilon_vals:
    expt['epsilon_fg'] = epsilon

    # Loop through redshift bins, assigning them to each process
    Ftot = 0
    for i in range(zs.size-1):
        if i % size != myid:
          continue
        print ">>>", myid, "working on redshift bin", i, " -- z =", zc[i]
        
        # Calculate basic Fisher matrix
        F = baofisher.fisher( zs[i], zs[i+1], fiducial, cosmo, expt, 
                              cosmo_fns=cosmo_fns, return_pk=False )
        
        # Expand Fisher matrix with EOS parameters and append to list
        F = baofisher.fisher_with_excluded_params(F, [6]) # Exclude P(k)
        F = baofisher.expand_fisher_matrix(zc[i], eos_derivs, F)
        Ftot += F
    
    # Sum all Ftot and invert on root to get 1D marginal constraints
    Fall = np.zeros(Ftot.shape)
    comm.Reduce(Ftot, Fall, op=MPI.SUM, root=0)
    
    if myid == 0:
        cov = np.linalg.inv(Fall)
        
        # (Abao, Omega_K, Omega_DE, w_0, w_a)
        # ['A', 'b_HI', 'f', 'sigma_NL', 'omegak', 'omegaDE', 'w0', 'wa']
        sig = np.array([ cov[0,0], cov[4,4], cov[5,5], cov[6,6], cov[7,7] ])
        sig = np.sqrt(sig)
        sigmas.append(sig)

# Save results
if myid == 0:
    print "Calculations complete. Saving to file."
    np.save("eos-fg-scan", sigmas)
    
