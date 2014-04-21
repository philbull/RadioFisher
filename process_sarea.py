#!/usr/bin/python
"""
Output a table of 1D marginals for a set of parameters.
"""
import numpy as np
import pylab as P
import baofisher
import matplotlib.patches
import matplotlib.cm
import matplotlib.ticker
from units import *
from mpi4py import MPI
import experiments
import os
import euclid

USE_DETF_PLANCK_PRIOR = True # If False, use Euclid prior instead
MARGINALISE_OVER_W0WA = True # Whether to fix or marginalise over (w0, wa)

cosmo = experiments.cosmo

names = [
  'exptS', 'iexptM', 'exptL', 'iexptL', 'cexptL',
  'GBT', 'Parkes', 'GMRT', 'WSRT', 'APERTIF',
  'VLBA', 'JVLA', 'iJVLA', 'BINGO', 'iBAOBAB32',   'iBAOBAB128',
  'yCHIME', 'iAERA3', 'KAT7', 'iKAT7', 'cKAT7',
  'MeerKATb1', 'iMeerKATb1', 'cMeerKATb1', 'MeerKATb2', 'iMeerKATb2',
  'cMeerKATb2', 'ASKAP', 'SKA1MIDbase1', 'iSKA1MIDbase1', 'cSKA1MIDbase1',
  'SKA1MIDbase2', 'iSKA1MIDbase2', 'cSKA1MIDbase2', 'SKA1MIDfull1', 'iSKA1MIDfull1',
  'cSKA1MIDfull1', 'SKA1MIDfull2', 'iSKA1MIDfull2', 'cSKA1MIDfull2', 'SKA1SURbase1',
  'SKA1SURbase2', 'SKA1SURfull1', 'SKA1SURfull2' ]

sarea = [100, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000]

idx_max = []; fom_max = []; sarea_max = []

for k in range(len(names)):
    _fom_max = 0.
    _idx_max = -1
    for j in range(len(sarea)):
        root = "output/" + names[k] + ("_%d" % sarea[j])
        
        # Load cosmo fns.
        dat = np.atleast_2d( np.genfromtxt(root+"-cosmofns-zc.dat") ).T
        zc, Hc, dAc, Dc, fc = dat
        zs, Hs, dAs, Ds, fs = np.genfromtxt(root+"-cosmofns-smooth.dat").T
        kc = np.genfromtxt(root+"-fisher-kc.dat").T
        
        # Load Fisher matrices as fn. of z
        Nbins = zc.size
        F_list = [np.genfromtxt(root+"-fisher-full-%d.dat" % i) for i in range(Nbins)]
        
        # EOS FISHER MATRIX
        pnames = baofisher.load_param_names(root+"-fisher-full-0.dat")
        zfns = ['b_HI',]
        excl = ['Tb', 'f', 'aperp', 'apar', 'DA', 'H', 'gamma', 'N_eff', 'pk*', 
                'fs8', 'bs8']
        F, lbls = baofisher.combined_fisher_matrix( F_list,
                                                    expand=zfns, names=pnames,
                                                    exclude=excl )
        # DETF Planck prior
        l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h', 'sigma8']
        F_detf = euclid.detf_to_baofisher("DETF_PLANCK_FISHER.txt", cosmo, omegab=False)
        Fpl, lbls = baofisher.add_fisher_matrices(F, F_detf, lbls, l2, expand=True)
                
        # Decide whether to fix various parameters
        fixed_params = []
        if not MARGINALISE_CURVATURE: fixed_params += ['omegak',]
        if not MARGINALISE_INITIAL_PK: fixed_params += ['n_s', 'sigma8']
        if not MARGINALISE_OMEGAB: fixed_params += ['omega_b',]
        
        if len(fixed_params) > 0:
            Fpl, lbls = baofisher.combined_fisher_matrix( [Fpl,], expand=[], 
                         names=lbls, exclude=fixed_params )
        
        # Get indices of w0, wa
        pw0 = lbls.index('w0'); pwa = lbls.index('wa'); pA = lbls.index('A')
        
        # Invert matrix
        cov_pl = np.linalg.inv(Fpl)
        
        # Calculate FOM
        fom = baofisher.figure_of_merit(pw0, pwa, None, cov=cov_pl)
        print "%s:%6d: FOM = %3.2f" % (names[k], sarea[j], fom)
        
        # Select best FOM
        if fom > _fom_max:
            _fom_max = fom
            _idx_max = j
    
    idx_max.append(_idx_max)
    fom_max.append(_fom_max)
    sarea_max.append(sarea[j])
    
    print "-"*50

print "\n\n"
print "id  name  Sarea  FOM"
for i in range(len(idx_max)):
    print i, names[i], sarea_max[i], fom_max[i]
