#!/usr/bin/python
"""
Output a table of 1D marginals for a set of parameters.
"""
import numpy as np
import pylab as P
import radiofisher as rf
import matplotlib.patches
import matplotlib.cm
import matplotlib.ticker
from radiofisher.units import *
from radiofisher import experiments, euclid
import os

USE_DETF_PLANCK_PRIOR = True # If False, use Euclid prior instead
MARGINALISE_OVER_W0WA = True # Whether to fix or marginalise over (w0, wa)
LCDM_ONLY = False

cosmo = experiments.cosmo

names = ['exptS', 'aexptM', 'exptL', 'iexptL', 'cexptL', 'GBT', 'Parkes', 
         'GMRT', 'WSRT', 'APERTIF', 'VLBA', 'JVLA', 'iJVLA', 'BINGO', 'iBAOBAB32',
         'iBAOBAB128', 'yCHIME', 'iAERA3', 'iMFAA', 'yTIANLAIpath', 'yTIANLAI', 
         'yTIANLAIband2', 'FAST', 'KAT7', 'iKAT7', 'cKAT7', 'MeerKATb1', 'iMeerKATb1',
         'cMeerKATb1', 'MeerKATb2', 'iMeerKATb2', 'cMeerKATb2', 'ASKAP', 
         'SKA1MIDbase1', 'iSKA1MIDbase1', 'cSKA1MIDbase1', 'SKA1MIDbase2', 
         'iSKA1MIDbase2', 'cSKA1MIDbase2', 'SKA1MIDfull1', 'iSKA1MIDfull1',
         'cSKA1MIDfull1', 'SKA1MIDfull2', 'iSKA1MIDfull2', 'cSKA1MIDfull2',
         'fSKA1SURbase1', 'fSKA1SURbase2', 'fSKA1SURfull1', 'fSKA1SURfull2', 
         'exptCV', 'GBTHIM', 'SKA0MID', 'fSKA0SUR', 'SKA1MID900', 'SKA1MID350',
         'iSKA1MID900', 'iSKA1MID350', 'fSKA1SUR650', 'fSKA1SUR350', 'aSKA1LOW',
         'SKAMID_PLUS', 'SKAMID_PLUS2', 'yCHIME_nocut', 'yCHIME_avglow',
         'EuclidRef_scan']

labels = ['Stage I', 'Stage II', 'Facility', 'iexptL', 'cexptL', 'GBT', 'Parkes', 
         'GMRT', 'WSRT', 'WSRT + APERTIF', 'VLBA', 'JVLA', 'iJVLA', 'BINGO', 'iBAOBAB32',
         'BAOBAB-128', 'yCHIME', 'Dense AA / AERA3', 'MFAA', 'TIANLAI Pathfinder',
         'TIANLAI (B1)', 'TIANLAI (B2)', 'FAST', 'KAT7', 'iKAT7', 'cKAT7', 'MeerKAT (B1)',
          'iMeerKATb1', 'cMeerKATb1', 'MeerKAT (B2)', 'iMeerKATb2', 'cMeerKATb2', 
          'ASKAP', 'SKA1MIDbase1', 'iSKA1MIDbase1', 'cSKA1MIDbase1', 'SKA1MIDbase2', 
         'iSKA1MIDbase2', 'cSKA1MIDbase2', 'SKA1MIDfull1', 'iSKA1MIDfull1',
         'cSKA1MIDfull1', 'SKA1MIDfull2', 'iSKA1MIDfull2', 'cSKA1MIDfull2',
         'fSKA1SURbase1', 'fSKA1SURbase2', 'fSKA1SURfull1', 'fSKA1SURfull2', 
         'exptCV', 'GBT-HIM', 'SKA0MID', 'fSKA0SUR', 'SKA1-MID (B2)', 'SKA1-MID (B1)',
         'iSKA1MID900', 'iSKA1MID350', 'SKA1-SUR (B2)', 'SKA1-SUR (B1)', 'aSKA1LOW',
         'SKA1-MID + MeerKAT (B1)', 'SKA1-MID + MeerKAT (B2)', 'CHIME nocut', 
         'CHIME avglow', 'DETF Stage IV (gal. survey)']

sarea = [
  5000, 2000, 25000, None, None, 100, 5000,
  1000, None, 25000, 5000, 1000, None, 5000, None,
  1000, 10000, 100, 5000, 2000,
  25000, 2000, 2000, 2000, None, None, 25000,
  None, None, 25000, None, None,
  25000, None, None, None, None,
  None, None, None, None,
  None, None, None, None,
  None, None, None, None, None, 1000,
  None, None, 25000, 25000, None, None,
  25000, 25000, None, 25000, 25000, 25000, 25000, -1]

#for j in range(len(names)):
#    print j, names[j], labels[j], sarea[j]
###########

print len(sarea), len(names)

# Define output parameters
#params = ['h', 'omega_b', 'omegaDE', 'n_s', 'sigma8']
#plabels = ['$h$', '$\Omega_b$', '$\Omega_\mathrm{DE}$', '$n_s$', '$\sigma_8$']
#exponent = np.array([-3, -4, -3, -4, -3])

if LCDM_ONLY:
    params = ['h', 'omega_b', 'omegaDE', 'n_s', 'sigma8']
    plabels = ['$h$', '$\omega_b$', '$\Omega_\mathrm{DE}$', '$n_s$', '$\sigma_8$']
    exponent = np.array([-3, -4, -3, -4, -3])
else:
    params = ['A', 'h', 'omegak', 'omegaDE', 'n_s', 'sigma8', 'gamma', 'w0', 'wa', 'fom']
    plabels = ['$A$', '$h$', '$\Omega_K$', '$\Omega_\mathrm{DE}$', '$n_s$', '$\sigma_8$', '$\gamma$', '$w_0$', '$w_a$', 'FOM']
    exponent = np.array([-2, -3, -4, -3, -4, -3, -2, -2, -2, 0])

plabels = [plabels[i] + r" $/ 10^{%d}$" % exponent[i] for i in range(len(plabels))]

# Prepare to save 1D marginals
params_1d = []; params_lbls = []; params_exptname = []

# Loop though experiments
_k = range(len(names))[::-1] # Reverse order of experiments
for k in _k:
    try:
        #root = "output/" + names[k]
        if sarea[k] == -1:
            root = "output/" + names[k]
        else:
            root = "output/%s_scan_%d" % (names[k], sarea[k])
    except:
        continue
    
    # Load cosmo fns.
    dat = np.atleast_2d( np.genfromtxt(root+"-cosmofns-zc.dat") ).T
    zc, Hc, dAc, Dc, fc = dat
    zs, Hs, dAs, Ds, fs = np.genfromtxt(root+"-cosmofns-smooth.dat").T
    kc = np.genfromtxt(root+"-fisher-kc.dat").T
    
    # Load Fisher matrices as fn. of z
    Nbins = zc.size
    F_list = [np.genfromtxt(root+"-fisher-full-%d.dat" % i) for i in range(Nbins)]
    
    # EOS FISHER MATRIX
    pnames = rf.load_param_names(root+"-fisher-full-0.dat")
    zfns = ['b_HI',]
    excl = ['Tb', 'f', 'aperp', 'apar', 'DA', 'H', 'N_eff', 'pk*', 
            'fs8', 'bs8'] #'gamma'
    F, lbls = rf.combined_fisher_matrix( F_list, expand=zfns, names=pnames,
                                         exclude=excl )
    # DETF Planck prior
    l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h', 'sigma8']
    F_detf = euclid.detf_to_rf("DETF_PLANCK_FISHER.txt", cosmo, omegab=False)
    Fpl, lbls = rf.add_fisher_matrices(F, F_detf, lbls, l2, expand=True)
            
    # Decide whether to fix various parameters
    if LCDM_ONLY:
        fixed_params = ['w0', 'wa', 'omegak', 'gamma', 'A']
    else:
        fixed_params = []
    
    if len(fixed_params) > 0:
        Fpl, lbls = rf.combined_fisher_matrix( [Fpl,], expand=[], 
                     names=lbls, exclude=fixed_params )
    
    # Get indices of w0, wa
    #pw0 = lbls.index('w0'); pwa = lbls.index('wa'); pA = lbls.index('A')
    
    # Invert matrix
    cov_pl = np.linalg.inv(Fpl)
    
    # Store 1D marginals
    sigma = []
    for p in params:
        if p == 'fom':
            fom = rf.figure_of_merit(lbls.index('w0'), lbls.index('wa'), 
                                     None, cov=cov_pl)
            sigma.append(fom)
            print "FOM:", fom
        else:
            idx = lbls.index(p)
            print "--", p, idx, np.sqrt(cov_pl[idx,idx])
            sigma.append( np.sqrt(cov_pl[idx,idx]) )
    params_1d.append(sigma)
    params_exptname.append(labels[k])
    print "xxx:", lbls
    

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
