#!/usr/bin/python
"""
Calculate Fisher matrix and P(k) constraints for all redshift bins for a given 
experiment.
"""

import numpy as np
import pylab as P
import baofisher
import matplotlib.patches
from units import *
from mpi4py import MPI
import experiments
import sys

comm = MPI.COMM_WORLD
myid = comm.Get_rank()
size = comm.Get_size()

################################################################################
# Set-up experiment parameters
################################################################################

# Load cosmology and experimental settings
e = experiments
cosmo = experiments.cosmo
expts = [e.exptL, e.exptL, e.exptL]
names = ["cL_dzbin", "cL_drbin", "cL_dnubin"]

# Define redshift bins
zs1, zc1 = baofisher.zbins_equal_spaced(expts[0], dz=0.1)
zs2, zc2 = baofisher.zbins_const_dr(expts[1], cosmo, initial_dz=0.1)
zs3, zc3 = baofisher.zbins_const_dnu(expts[2], cosmo, initial_dz=0.1)
_zs = [zs1, zs2, zs3]
_zc = [zc1, zc2, zc3]

exit()

# Define kbins (used for output)
kbins = np.logspace(np.log10(0.001), np.log10(50.), 2) # Only 1 kbin

# Precompute cosmological functions
cosmo_fns, cosmo = baofisher.precompute_for_fisher(cosmo, "camb/baofisher_matterpower.dat")
H, r, D, f = cosmo_fns

# Massive neutrinos
cosmo['mnu'] = 0. #15
massive_nu_fn = None #baofisher.deriv_logpk_mnu(cosmo['mnu'], cosmo, dmnu=0.01, kmax=130.)
transfer_fn = None #baofisher.deriv_transfer(cosmo, "camb/baofisher_transfer_out.dat")

# Loop over models
for k in range(len(names)):

    if myid == 0:
        print "="*50
        print "Survey:", names[k]
        print "="*50

    # Tweak settings depending on chosen experiment
    cv_limited = False
    if names[k][0] == "i": expts[k]['mode'] = "interferom."
    if names[k][0] == "c": expts[k]['mode'] = "combined"

    expt = expts[k]
    survey_name = names[k]

    root = "output/" + survey_name
    
    # Redshift bins
    zs = _zs[k]
    zc = _zc[k]
    
    ################################################################################
    # Store cosmological functions
    ################################################################################

    # Store values of cosmological functions
    if myid == 0:
        # Calculate cosmo fns. at redshift bin centroids and save
        _H = H(zc)
        _dA = r(zc) / (1. + np.array(zc))
        _D = D(zc)
        _f = f(zc)
        np.savetxt(root+"-cosmofns-zc.dat", np.column_stack((zc, _H, _dA, _D, _f)))
        
        # Save bin edges
        np.savetxt(root+"-zbins.dat", zs)
        
        # Calculate cosmo fns. as smooth fns. of z and save
        zz = np.linspace(0., 1.05*np.max(zc), 1000)
        _H = H(zz)
        _dA = r(zz) / (1. + zz)
        _D = D(zz)
        _f = f(zz)
        np.savetxt(root+"-cosmofns-smooth.dat", np.column_stack((zz, _H, _dA, _D, _f)) )
    
    # Precompute derivs for all processes
    eos_derivs = baofisher.eos_fisher_matrix_derivs(cosmo, cosmo_fns)


    ################################################################################
    # Loop through redshift bins, assigning them to each process
    ################################################################################

    for i in range(zs.size-1):
        if i % size != myid:
          continue
        
        print ">>>", myid, "working on redshift bin", i, " -- z =", zc[i]
        
        # Calculate basic Fisher matrix
        # (A, bHI, Tb, sigma_NL, sigma8, n_s, f, aperp, apar, [Mnu], [fNL], [pk]*Nkbins)
        F_pk, kc, binning_info = baofisher.fisher( zs[i], zs[i+1], cosmo, expt, 
                                             cosmo_fns=cosmo_fns,
                                             transfer_fn=transfer_fn,
                                             massive_nu_fn=massive_nu_fn,
                                             return_pk=True,
                                             cv_limited=cv_limited, 
                                             kbins=kbins )
        
        # Expand Fisher matrix with EOS parameters
        ##F_eos = baofisher.fisher_with_excluded_params(F, [10, 11, 12]) # Exclude P(k)
        F_eos = baofisher.expand_fisher_matrix(zc[i], eos_derivs, F_pk, exclude=[])
        
        # Expand Fisher matrix for H(z), dA(z)
        # Replace aperp with dA(zi), using product rule. aperp(z) = dA(fid,z) / dA(z)
        # (And convert dA to Gpc, to help with the numerics)
        da = r(zc[i]) / (1. + zc[i]) / 1000. # Gpc
        F_eos[7,:] *= -1. / da
        F_eos[:,7] *= -1. / da
        
        # Replace apar with H(zi)/100, using product rule. apar(z) = H(z) / H(fid,z)
        F_eos[8,:] *= 1. / H(zc[i]) * 100.
        F_eos[:,8] *= 1. / H(zc[i]) * 100.
        
        # Save Fisher matrix and k bins
        np.savetxt(root+"-fisher-full-%d.dat" % i, F_eos)
        if myid == 0: np.savetxt(root+"-fisher-kc.dat", kc)
        
        # Save P(k) rebinning info
        #np.savetxt(root+"-rebin-Fbase-%d.dat" % i, np.array(binning_info['F_base']) )
        #np.savetxt(root+"-rebin-cumul-%d.dat" % i, np.array(binning_info['cumul']) )
        #np.savetxt(root+"-rebin-kgrid-%d.dat" % i, np.array(binning_info['kgrid']) )
        #np.savetxt(root+"-rebin-Vfac-%d.dat" % i, np.array([binning_info['Vfac'],]) )

comm.barrier()
