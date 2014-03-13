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

#expts = [e.exptS, e.exptM, e.exptL]
#names = ['exptS', 'iexptM', 'cexptL']

expts = [ e.exptS, e.exptM, e.exptL, e.GBT, e.BINGO, e.WSRT, e.APERTIF, 
          e.JVLA, e.ASKAP, e.KAT7, e.MeerKAT_band1, e.MeerKAT, e.SKA1MID,
          e.SKA1SUR, e.SKA1SUR_band1, e.SKAMID_PLUS, e.SKAMID_PLUS_band1, 
          e.SKASUR_PLUS, e.SKASUR_PLUS_band1 ]

names = ['exptS', 'iexptM', 'cexptL', 'GBT', 'BINGO', 'WSRT', 'APERTIF', 
         'JVLA', 'cASKAP', 'cKAT7', 'cMeerKAT_band1', 'cMeerKAT', 'cSKA1MID',
         'SKA1SUR', 'SKA1SUR_band1', 'SKAMID_PLUS', 'SKAMID_PLUS_band1', 
         'SKASUR_PLUS', 'SKASUR_PLUS_band1']

#names = ['exptS_mnu02', 'iexptM_mnu02', 'cexptL_mnu02']

#expts = [e.exptL, e.exptL, e.exptL, e.exptL, e.exptL, e.exptL, e.exptL, e.exptL]
#names = ['cexptL_Sarea2k', 'cexptL_Sarea5k', 'cexptL_Sarea10k', 'cexptL_Sarea15k', 'cexptL_Sarea20k', 'cexptL_Sarea30k', 'cexptL_Sarea25k', 'cexptL_Sarea1k']

#names = ['cexptL_bao',]
#names = ['cexptL_bao_rsd',]
#names = ['cexptL_bao_pkshift',]
#names = ['cexptL_bao_vol',]
#names = ['cexptL_bao_allap',]
#names = ['cexptL_bao_all',]
#expts = [e.exptL,]

#expts[0]['use'] = {
#  'f_rsd':             True,     # RSD constraint on f(z)
#  'f_growthfactor':    False,    # D(z) constraint on f(z)
#  'alpha_all':         False,     # Use all constraints on alpha_{perp,par}
#  'alpha_volume':      True,
#  'alpha_rsd_angle':   True, #False, #t
#  'alpha_rsd_shift':   True, #False, #t
#  'alpha_bao_shift':   True,
#  'alpha_pk_shift':    True # True
#}


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


# FIXME:
if "exptM" in names[k]:
    expt['Sarea'] /= 6. # 5,000deg^2
    print "Setting survey area to 5,000 deg^2."

# FIXME
#areas = [2., 5., 10., 15., 20., 30.,   25., 1.]
#expt['Sarea'] *= (areas[k] / 30.)
#if myid == 0: print "Setting Sarea to", areas[k], "k deg^2"

# Define redshift bins
expt_zbins = baofisher.overlapping_expts(expt)
#zs, zc = baofisher.zbins_equal_spaced(expt_zbins, dz=0.1)
#zs, zc = baofisher.zbins_const_dr(expt_zbins, cosmo, bins=14)
zs, zc = baofisher.zbins_const_dnu(expt_zbins, cosmo, dnu=60.)

# Define kbins (used for output)
kbins = np.logspace(np.log10(0.001), np.log10(50.), 91)

# Neutrino mass
cosmo['mnu'] = 0.
#cosmo['mnu'] = 0.05
#cosmo['mnu'] = 0.10
#cosmo['mnu'] = 0.20

# Precompute cosmological functions, P(k), massive neutrinos, and T(k) for f_NL
cosmo_fns = baofisher.background_evolution_splines(cosmo)
if cosmo['mnu'] != 0.:
    # Massive neutrinos
    mnu_str = "mnu%03d" % (cosmo['mnu']*100.)
    fname_pk = "cache_pk_%s.dat" % mnu_str
    fname_nu = "cache_%s" % mnu_str
    survey_name += mnu_str; root += mnu_str
    
    cosmo = baofisher.load_power_spectrum(cosmo, fname_pk, comm=comm)
    Neff_fn = baofisher.deriv_neutrinos(cosmo, fname_nu, Neff=cosmo['N_eff'], comm=comm)
else:
    # Normal operation (no massive neutrinos or non-Gaussianity)
    cosmo = baofisher.load_power_spectrum(cosmo, "cache_pk.dat", comm=comm)
    massive_nu_fn = None

# Non-Gaussianity
#transfer_fn = baofisher.deriv_transfer(cosmo, "cache_transfer.dat", comm=comm)
transfer_fn = None

# Effective no. neutrinos, N_eff
Neff_fn = baofisher.deriv_neutrinos(cosmo, "cache_Neff", Neff=cosmo['N_eff'], comm=comm)
#Neff_fn = None

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
    F_pk, kc, binning_info, paramnames = baofisher.fisher( 
                                         zs[i], zs[i+1], cosmo, expt_eff, 
                                         cosmo_fns=cosmo_fns,
                                         transfer_fn=transfer_fn,
                                         massive_nu_fn=massive_nu_fn,
                                         Neff_fn=Neff_fn,
                                         return_pk=True,
                                         cv_limited=cv_limited, 
                                         kbins=kbins )
    
    # Expand Fisher matrix with EOS parameters
    ##F_eos = baofisher.fisher_with_excluded_params(F, [10, 11, 12]) # Exclude P(k)
    F_eos, paramnames = baofisher.expand_fisher_matrix(zc[i], eos_derivs, F_pk, 
                                                  names=paramnames, exclude=[])
    
    # Expand Fisher matrix for H(z), dA(z)
    # Replace aperp with dA(zi), using product rule. aperp(z) = dA(fid,z) / dA(z)
    # (And convert dA to Gpc, to help with the numerics)
    paramnames[paramnames.index('aperp')] = 'DA'
    da = r(zc[i]) / (1. + zc[i]) / 1000. # Gpc
    F_eos[7,:] *= -1. / da
    F_eos[:,7] *= -1. / da
    
    # Replace apar with H(zi)/100, using product rule. apar(z) = H(z) / H(fid,z)
    paramnames[paramnames.index('apar')] = 'H'
    F_eos[8,:] *= 1. / H(zc[i]) * 100.
    F_eos[:,8] *= 1. / H(zc[i]) * 100.
    
    # Save Fisher matrix and k bins
    np.savetxt(root+"-fisher-full-%d.dat" % i, F_eos, header=" ".join(paramnames))
    if myid == 0: np.savetxt(root+"-fisher-kc.dat", kc)
    
    # Save P(k) rebinning info
    np.savetxt(root+"-rebin-Fbase-%d.dat" % i, np.array(binning_info['F_base']) )
    np.savetxt(root+"-rebin-cumul-%d.dat" % i, np.array(binning_info['cumul']) )
    np.savetxt(root+"-rebin-kgrid-%d.dat" % i, np.array(binning_info['kgrid']) )
    np.savetxt(root+"-rebin-Vfac-%d.dat" % i, np.array([binning_info['Vfac'],]) )

comm.barrier()
