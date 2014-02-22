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

#expts = [e.GBT, e.BINGO, e.WSRT, e.APERTIF, e.JVLA, e.ASKAP, e.KAT7, e.MeerKAT, e.SKA1, e.SKAMID, e.SKAMID, e.exptS, e.exptM, e.exptL, e.exptL, e.exptM, e.exptL, e.exptL, e.exptL]
#names = ["GBT", "BINGO", "WSRT", "APERTIF", "JVLA", "ASKAP", "KAT7", "MeerKAT", "SKA1", "SKAMID", "iSKAMID", "exptS", "iexptM", "exptL", "cexptL", "iexptM2", "cNEWexptL", "cNEW2exptL", "cNEW3exptL"]

expts = [e.exptS, e.exptM, e.exptL]
names = ['exptS', 'iexptM2', 'cexptL']

#expts = [e.SKA1MID, e.SKA1MID, e.SKA1MID, e.SKA1SUR, e.superSKA1MID]
#names = ["SKA1MID", "iSKA1MID", "cSKA1MID", "SKA1SUR", "superSKA1MID"]

# Take command-line argument for which survey to calculate, or set manually
if len(sys.argv) > 1:
    k = int(sys.argv[1])
else:
    raise IndexError("Need to specify ID for experiment.")
if myid == 0:
    print "="*50
    print "Survey:", names[k]
    print "="*50

# Tweak settings depending on chosen experiment
cv_limited = False
expts[k]['mode'] = "dish"
if names[k][0] == "i": expts[k]['mode'] = "interferom."
if names[k][0] == "c": expts[k]['mode'] = "combined"

expt = expts[k]
survey_name = names[k]
#expt.pop('n(x)', None)
root = "output/" + survey_name


# FIXME
#expt['Sarea'] /= 6.

# Define redshift bins
expt_zbins = baofisher.overlapping_expts(expt)
#zs, zc = baofisher.zbins_equal_spaced(expt_zbins, dz=0.1)
#zs, zc = baofisher.zbins_const_dr(expt_zbins, cosmo, bins=14)
zs, zc = baofisher.zbins_const_dnu(expt_zbins, cosmo, dnu=60.)

# Define kbins (used for output)
kbins = np.logspace(np.log10(0.001), np.log10(50.), 91)

# Precompute cosmological functions, P(k), massive neutrinos, and T(k) for f_NL
cosmo['mnu'] = 0.1
cosmo_fns = baofisher.background_evolution_splines(cosmo)
cosmo = baofisher.load_power_spectrum(cosmo, "cache_pk.dat", comm=comm)
massive_nu_fn = baofisher.deriv_logpk_mnu(cosmo['mnu'], cosmo, "cache_mnu010", comm=comm)
#transfer_fn = baofisher.deriv_transfer(cosmo, "cache_transfer.dat", comm=comm)
transfer_fn = None
H, r, D, f = cosmo_fns

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
    
    print ">>> %2d working on redshift bin %2d -- z = %3.3f" % (myid, i, zc[i])
    
    # Calculate effective experimental params. in the case of overlapping expts.
    expt_eff = baofisher.overlapping_expts(expt, zs[i], zs[i+1])
    
    # Calculate basic Fisher matrix
    # (A, bHI, Tb, sigma_NL, sigma8, n_s, f, aperp, apar, [Mnu], [fNL], [pk]*Nkbins)
    F_pk, kc, binning_info = baofisher.fisher( zs[i], zs[i+1], cosmo, expt_eff, 
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
    np.savetxt(root+"-rebin-Fbase-%d.dat" % i, np.array(binning_info['F_base']) )
    np.savetxt(root+"-rebin-cumul-%d.dat" % i, np.array(binning_info['cumul']) )
    np.savetxt(root+"-rebin-kgrid-%d.dat" % i, np.array(binning_info['kgrid']) )
    np.savetxt(root+"-rebin-Vfac-%d.dat" % i, np.array([binning_info['Vfac'],]) )

comm.barrier()
