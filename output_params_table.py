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
#from mpi4py import MPI
import experiments
import os
import euclid

USE_DETF_PLANCK_PRIOR = True # If False, use Euclid prior instead
MARGINALISE_OVER_W0WA = True # Whether to fix or marginalise over (w0, wa)

cosmo = experiments.cosmo

#names = ['exptS', 'iexptM', 'cexptL', 'GBT', 'BINGO', 'WSRT', 'APERTIF', 
#         'JVLA', 'cASKAP', 'cKAT7', 'cMeerKAT_band1', 'cMeerKAT', 'cSKA1MID',
#         'SKA1SUR', 'SKA1SUR_band1', 'SKAMID_PLUS', 'SKAMID_PLUS_band1', 
#         'SKASUR_PLUS', 'SKASUR_PLUS_band1', 'EuclidRef', 'EuclidOpt']
#labels = ['Snapshot', 'Mature', 'Facility', 'GBT', 'BINGO', 'WSRT', 'APERTIF', 
#         'JVLA', 'ASKAP', 'KAT7', 'MeerKAT (Band 1)', 'MeerKAT', 'SKA1-MID',
#         'SKA1-SUR', 'SKA1-SUR (Band 1)', 'SKA1-MID+', 'SKA1-MID+ (Band 1)', 
#         'SKA1-SUR+', 'SKA1-SUR+ (Band 1)', 'Euclid (ref.)', 'Euclid (opt.)']

names = [
  'exptS', 'iexptM', 'cexptL',
  'GBT', 'Parkes', 'GMRT', 'WSRT', 'APERTIF',
  'VLBA', 'JVLA', 'BINGO', 'iBAOBAB32',
  'iBAOBAB128', 'yCHIME', 'iAERA3', 
  'cKAT7', 'cMeerKATb1', 'cMeerKATb2', 'ASKAP',
  'SKA1MIDbase1', 'iSKA1MIDbase1', 'cSKA1MIDbase1',
  'SKA1MIDbase2', 'iSKA1MIDbase2', 'cSKA1MIDbase2',
  'SKA1MIDfull1', 'iSKA1MIDfull1', 'cSKA1MIDfull1',
  'SKA1MIDfull2', 'iSKA1MIDfull2', 'cSKA1MIDfull2',
  'SKA1SURbase1', 'SKA1SURbase2',
  'SKA1SURfull1', 'SKA1SURfull2' ]

labels = [
  'Stage I', 'Stage II', 'Facility',
  'GBT', 'Parkes', 'GMRT (ASKAP PAF)', 'WSRT', 'WSRT + APERTIF',
  'VLBA', 'JVLA', 'BINGO', 'BAOBAB-32',
  'BAOBAB-128', 'CHIME Full', 'AERA3 / Dense AA', 
  'KAT7', 'MeerKAT (B1)', 'MeerKAT (B2)', 'ASKAP',
  'SKA1-MID Base (B1) SD', 'SKA1-MID Base (B1) Int.', 'SKA1-MID Base (B1) Comb.',
  'SKA1-MID Base (B2) SD', 'SKA1-MID Base (B2) Int.', 'SKA1-MID Base (B2) Comb.',
  'SKA1-MID Full (B1) SD', 'SKA1-MID Full (B1) Int.', 'SKA1-MID Full (B1) Comb.',
  'SKA1-MID Full (B2) SD', 'SKA1-MID Full (B2) Int.', 'SKA1-MID Full (B2) Comb.',
  'SKA1-SUR Base (B1)', 'SKA1-SUR Base (B2)',
  'SKA1-SUR Full (B1)', 'SKA1-SUR Full (B2)' ]

sarea = [
  5000, 2000, 25000,
  100, 500, 2000, 100, 25000,
  5000, 1000, 5000, 500,
  2000, 5000, 500,
  5000, 15000, 25000, 25000,
  25000, 25000, 25000,
  25000, 25000, 25000,
  25000, 25000, 25000,
  25000, 25000, 25000,
  25000, 25000,
  25000, 25000 ]

# Define output parameters
params = ['h', 'omega_b', 'omegak', 'omegaDE', 'n_s', 'sigma8']
plabels = ['$h$', '$\Omega_b$', '$\Omega_K$', '$\Omega_\mathrm{DE}$', '$n_s$', '$\sigma_8$']
exponent = np.array([-3, -4, -4, -3, -4, -3])
plabels = [plabels[i] + r" $/ 10^{%d}$" % exponent[i] for i in range(len(plabels))]

# Prepare to save 1D marginals
params_1d = []; params_lbls = []; params_exptname = []

# Loop though experiments
_k = range(len(names))[::-1] # Reverse order of experiments
for k in _k:
    root = "output/" + names[k] + ("_%d" % sarea[k])
    
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
    #if not MARGINALISE_CURVATURE: fixed_params += ['omegak',]
    #if not MARGINALISE_INITIAL_PK: fixed_params += ['n_s', 'sigma8']
    #if not MARGINALISE_OMEGAB: fixed_params += ['omega_b',]
    
    if len(fixed_params) > 0:
        Fpl, lbls = baofisher.combined_fisher_matrix( [Fpl,], expand=[], 
                     names=lbls, exclude=fixed_params )
    
    # Get indices of w0, wa
    pw0 = lbls.index('w0'); pwa = lbls.index('wa'); pA = lbls.index('A')
    
    # Invert matrix
    cov_pl = np.linalg.inv(Fpl)
    
    # Store 1D marginals
    sigma = []
    for p in params:
        idx = lbls.index(p)
        sigma.append( np.sqrt(cov_pl[idx,idx]) )
    params_1d.append(sigma)
    params_exptname.append(labels[k])
    

# Reverse order of rows
params_1d = params_1d[::-1]
params_exptname = params_exptname[::-1]

# Table header
tbl = []
tbl.append("\hline")
tbl.append(r"{\bf Experiments} & " + " & ".join(plabels) + " \\\\")
tbl.append("\hline")

# Table rows (one per experiment)
for j in range(len(params_exptname)):
    vals = [val for val in params_1d[j]]
    s = ["%3.1f" % (vals[i] / 10.**exponent[i]) for i in range(len(vals))]
    line = params_exptname[j] + " & " + " & ".join(s) + " \\\\"
    tbl.append(line)

# Output table
print "-"*50
print ""
for t in tbl:
    print t
print ""
