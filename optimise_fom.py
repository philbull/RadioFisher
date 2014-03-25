#!/usr/bin/python
"""
Find parameters that optimise a given figure of merit.
"""
import numpy as np
import baofisher
from units import *
from mpi4py import MPI
import experiments
import euclid
import sys, copy

comm = MPI.COMM_WORLD
myid = comm.Get_rank()
size = comm.Get_size()

fname = "opt_expt_grid.dat"

################################################################################
# Set-up experiment parameters
################################################################################

# Precompute cosmology functions
default_cosmo = experiments.cosmo
default_cosmo['mnu'] = 0.
cosmo_fns = baofisher.background_evolution_splines(default_cosmo)
default_cosmo = baofisher.load_power_spectrum(default_cosmo, "cache_pk.dat", comm=comm)
eos_derivs = baofisher.eos_fisher_matrix_derivs(default_cosmo, cosmo_fns)

# Load Planck prior
F_detf = euclid.detf_to_baofisher("DETF_PLANCK_FISHER.txt", default_cosmo, omegab=False)


def calculate_fom(expt):
    """
    Return FOM for a given set of experimental parameters.
    """
    # Set-up parameters, bins etc.
    #cosmo = copy.deepcopy(default_cosmo)
    cosmo = default_cosmo
    zs, zc = baofisher.zbins_const_dnu(expt, cosmo, dnu=60.)
    
    # Calculate Fisher matrices (spread over all processes)
    Fz = [0 for i in range(zs.size-1)]
    for i in range(zs.size-1):
        
        # Calculate basic Fisher matrix
        F, paramnames = baofisher.fisher( zs[i], zs[i+1], cosmo, expt, 
                                          cosmo_fns=cosmo_fns,
                                          transfer_fn=None, massive_nu_fn=None,
                                          Neff_fn=None, return_pk=False,
                                          cv_limited=False )
        
        # Expand Fisher matrix with EOS parameters
        Fz[i], paramnames = baofisher.expand_fisher_matrix(zc[i], eos_derivs, F, 
                                                    names=paramnames, exclude=[])
    
    # Once all bins are completed, sum Fisher matrices into root node
    #F_list = comm.gather(Fz, [], root=0)
    #fom = 0.
    #if myid == 0:
    #    F_list = np.sum(F_list, axis=0)
    F_list = Fz
        
    # Create combined Fisher matrix with fns. of z
    zfns = ['b_HI',]
    excl = ['Tb', 'f', 'aperp', 'apar', 'DA', 'H', 'gamma', 'N_eff', 'pk*']
    F, lbls = baofisher.combined_fisher_matrix(F_list, expand=zfns, 
                                               names=paramnames, exclude=excl)
    # Add Planck prior
    l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h', 'sigma8']
    Fpl, lbls = baofisher.add_fisher_matrices(F, F_detf, lbls, l2, expand=True)
   
    # Invert matrix, get FOM
    cov_pl = np.linalg.inv(Fpl)
    pw0 = lbls.index('w0'); pwa = lbls.index('wa')
    fom = baofisher.figure_of_merit(pw0, pwa, None, cov=cov_pl)
    return fom    


# Set base experiment specification
expt = {
    'mode':             'combined',        # Interferometer or single dish
    'Ndish':            250,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            2.5,               # Single dish diameter [m]
    'Tinst':            25.*(1e3),         # System temp. [mK]
    'survey_dnutot':    700.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1100.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            1e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'Dmax':             60.,               # Max. interferom. baseline [m]
    'Dmin':             2.5,                # Min. interferom. baseline [m]
    'ttot':             10e3*HRS_MHZ,      # Total integration time [MHz^-1]
    'nu_line':          1420.406,          # Rest-frame freq. of emission line [MHz]
    'epsilon_fg':       1e-6,              # FG subtraction residual amplitude
    'use':              experiments.USE    # Which constraints to use/ignore
    }

# Define grid of parameters
Dmin = np.linspace(2., 12., 7)
Dmax = np.logspace(np.log10(40.), np.log10(400.), 8)
Sarea = np.logspace(np.log10(0.5), np.log10(25.), 7)

# Message about how long this is going to take...
if myid == 0:
    print "No. of evaluations required: %d" % (Dmin.size * Dmax.size * Sarea.size)

# Loop through grid and calculate FOM for each
j = -1
all_vals = []
for _Dmin in Dmin:
  for _Dmax in Dmax:
    for _Sarea in Sarea:
      
      # Decide if this set of params belongs to this processor
      j += 1
      if j % size != myid: continue
      
      # Update expt settings
      _expt = copy.deepcopy(expt)
      _expt['Ddish'] = _Dmin # Matched to Dmin
      _expt['Dmin'] = _Dmin
      _expt['Dmax'] = _Dmax
      _expt['Sarea'] *= _Sarea # In 1000's of deg^2
      
      # Calculate filling factor and FOM
      ff = _expt['Ndish'] * (_expt['Ddish'] / _expt['Dmax'])**2.
      fom = 0.
      if ff < 1.:
          fom = calculate_fom(_expt)
      else:
          print "*** SKIPPING: filling factor too large"
      
      # Output results
      f = open(fname, 'a')
      f.write(" ".join([myid, _Dmin, _Dmax, _Sarea, ff, fom]))
      f.close()
      all_vals.append([myid, _Dmin, _Dmax, _Sarea, ff, fom])
      #print myid, _Dmin, _Dmax, _Sarea, ff, fom

print "LISTING:", myid, all_vals
comm.barrier()
if myid == 0: print "\nFinished."
