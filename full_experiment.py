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
expts = [e.GBT, e.BINGO, e.WSRT, e.APERTIF, e.JVLA, e.ASKAP, e.KAT7, e.MeerKAT, e.SKA1, e.SKAMID, e.SKAMID, e.exptS, e.exptM, e.exptL, e.exptL]
names = ["GBT", "BINGO", "WSRT", "APERTIF", "JVLA", "ASKAP", "KAT7", "MeerKAT", "SKA1", "SKAMID", "iSKAMID", "exptS", "iexptM", "exptL", "cexptL"]

#"SKA1_CV", "SKAMID_5kdeg", "SKAMID_mnu01", "iSKAMID_mnu01", "SKAMID_COMP_BIGZ", "iSKAMID_COMP_BIGZ", "iSKAMID_BIGZ"]

# Take command-line argument for which survey to calculate, or set manually
if len(sys.argv) > 1:
    k = int(sys.argv[1])
else:
    raise IndexError("Need to specify ID for experiment.")
if myid == 0:
    print "="*50
    print "Survey:", names[k]
    print "="*50

"""
#===============================================================================
# FIXME: Special mode
expts = [e.SKA1, e.SKA1]
names = ["xSKA", "xSKA_extended"]
#===============================================================================
if k == 0:
    # SKA + MK (full z range)
    expts[k]['survey_dnutot'] = 800.
    expts[k]['survey_numax'] = 1150. 
else:
    # SKA + MK, with extension to low z
    expts[k]['survey_dnutot'] = 1050.
    expts[k]['survey_numax'] = 1400.
#===============================================================================
"""

# Tweak settings depending on chosen experiment
cv_limited = False
#if k == 13: cv_limited = True
#if k == 14: expts[k]['Sarea'] = 39e3*(D2RAD)**2.
if names[k][0] == "i": expts[k]['mode'] = "interferom."
if names[k][0] == "c": expts[k]['mode'] = "combined"

expt = expts[k]
survey_name = names[k]

root = "output/" + survey_name

# Define redshift bins
zs, zc = baofisher.zbins_equal_spaced(expt, dz=0.1)
#zs, zc = baofisher.zbins_const_dr(expt, cosmo, bins=14)

# Define kbins (used for output)
kbins = np.logspace(np.log10(0.001), np.log10(50.), 91)

# Precompute cosmological functions
cosmo_fns, cosmo = baofisher.precompute_for_fisher(cosmo, "camb/baofisher_matterpower.dat")
H, r, D, f = cosmo_fns

# Massive neutrinos
cosmo['mnu'] = 0.15
massive_nu_fn = baofisher.deriv_logpk_mnu(cosmo['mnu'], cosmo, dmnu=0.01, kmax=130.)

# Non-gaussianity
transfer_fn = None #baofisher.deriv_transfer(cosmo, "camb/baofisher_transfer_out.dat")


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
    
    print ">>>", myid, "working on redshift bin", i, " -- z =", zc[i]
    
    """
    ##################################
    # FIXME: Special test to bump dish numbers in SKA+MK overlap bands
    zmin_mk = (1420. / 1015.) - 1.
    zmax_mk = (1420. / 580.) - 1.
    if zs[i] > zmin_mk and zs[i+1] < zmax_mk:
        expt['Ndish'] = 254.
        print "\tDISHES: 254"
    else:
        expt['Ndish'] = 190.
        print "\tDISHES: 190"
    ##################################
    """
    
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
    np.savetxt(root+"-rebin-Fbase-%d.dat" % i, np.array(binning_info['F_base']) )
    np.savetxt(root+"-rebin-cumul-%d.dat" % i, np.array(binning_info['cumul']) )
    np.savetxt(root+"-rebin-kgrid-%d.dat" % i, np.array(binning_info['kgrid']) )
    np.savetxt(root+"-rebin-Vfac-%d.dat" % i, np.array([binning_info['Vfac'],]) )

comm.barrier()
