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

names = ['EuclidRef', 'cexptL', 'iexptM']
labels = ['Euclid (ref.)', 'Facility', 'Mature']


names = ['exptS', 'iexptM', 'cexptL', 'GBT', 'BINGO', 'WSRT', 'APERTIF', 
         'JVLA', 'cASKAP', 'cKAT7', 'cMeerKAT_band1', 'cMeerKAT', 'cSKA1MID',
         'SKA1SUR', 'SKA1SUR_band1', 'SKAMID_PLUS', 'SKAMID_PLUS_band1', 
         'SKASUR_PLUS', 'SKASUR_PLUS_band1', 'EuclidRef', 'EuclidOpt']
labels = ['Snapshot', 'Mature', 'Facility', 'GBT', 'BINGO', 'WSRT', 'APERTIF', 
         'JVLA', 'ASKAP', 'KAT7', 'MeerKAT (Band 1)', 'MeerKAT', 'SKA1-MID',
         'SKA1-SUR', 'SKA1-SUR (Band 1)', 'SKA1-MID+', 'SKA1-MID+ (Band 1)', 
         'SKA1-SUR+', 'SKA1-SUR+ (Band 1)', 'Euclid (ref.)', 'Euclid (opt.)']

# Prepare to save 1D marginals
params_1d = []; params_lbls = []; params_exptname = []

# Loop though experiments
_k = range(len(names))[::-1] # Reverse order of experiments
for k in _k:
    root = "output/" + names[k]
    
    print "-"*50
    print names[k]
    print "-"*50

    # Load cosmo fns.
    dat = np.atleast_2d( np.genfromtxt(root+"-cosmofns-zc.dat") ).T
    zc, Hc, dAc, Dc, fc = dat
    zs, Hs, dAs, Ds, fs = np.genfromtxt(root+"-cosmofns-smooth.dat").T
    kc = np.genfromtxt(root+"-fisher-kc.dat").T
    
    # Load Fisher matrices as fn. of z
    Nbins = zc.size
    F_list = [np.genfromtxt(root+"-fisher-full-%d.dat" % i) for i in range(Nbins)]
    
    # EOS FISHER MATRIX
    # Actually, (aperp, apar) are (D_A, H)
    pnames = ['A', 'b_HI', 'Tb', 'sigma_NL', 'sigma8', 'n_s', 'f', 'aperp', 'apar', 
             'omegak', 'omegaDE', 'w0', 'wa', 'h', 'gamma']
    pnames += ["pk%d" % i for i in range(kc.size)]
    
    zfns = [1,]
    excl = [2,  6,7,8,   14,] #15] # 4:sigma8
    excl  += [i for i in range(len(pnames)) if "pk" in pnames[i]]
    
    F, lbls = baofisher.combined_fisher_matrix( F_list,
                                                expand=zfns, names=pnames,
                                                exclude=excl )
    # Apply Planck prior
    if USE_DETF_PLANCK_PRIOR:
        # DETF Planck prior
        print "*** Using DETF Planck prior ***"
        l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h']
        F_detf = euclid.detf_to_baofisher("DETF_PLANCK_FISHER.txt", cosmo)
        Fpl, lbls = baofisher.add_fisher_matrices(F, F_detf, lbls, l2, expand=True)
    else:
        # Euclid Planck prior
        print "*** Using Euclid (Mukherjee) Planck prior ***"
        l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h']
        Fe = euclid.planck_prior_full
        F_eucl = euclid.euclid_to_baofisher(Fe, cosmo)
        Fpl, lbls = baofisher.add_fisher_matrices(F, F_eucl, lbls, l2, expand=True)
    
    # Remove w0, wa from Fisher matrix
    if MARGINALISE_OVER_W0WA:
        print "*** Marginalising over (w0, wa) ***"
    else:
        print "*** Fixing (w0, wa) ***"
        Fpl, lbls = baofisher.combined_fisher_matrix( [Fpl,], expand=[], names=lbls,
                                     exclude=[lbls.index('w0'), lbls.index('wa')] )
    # Invert matrices
    cov_pl = np.linalg.inv(Fpl)
    
    # Store 1D marginals
    params_1d.append(np.sqrt(np.diag(cov_pl)))
    params_lbls.append(lbls)
    params_exptname.append(labels[k])
    

# Define output parameters
params = ['h', 'omega_b', 'omegak', 'omegaDE', 'n_s', 'sigma8']
plabels = ['$h$', '$\Omega_b$', '$\Omega_K$', '$\Omega_\mathrm{DE}$', '$n_s$', '$\sigma_8$']
exponent = np.array([-3, -4, -4, -3, -4, -3])
plabels = [plabels[i] + r" $/ 10^{%d}$" % exponent[i] for i in range(len(plabels))]

# Reverse order of rows
params_1d = params_1d[::-1]
params_lbls = params_lbls[::-1]
params_exptname = params_exptname[::-1]

# Table header
tbl = []
tbl.append("\hline")
tbl.append(r"{\bf Experiments} & " + " & ".join(plabels) + " \\\\")
tbl.append("\hline")

# Table rows (one per experiment)
for j in range(len(params_exptname)):
    vals = [params_1d[j][params_lbls[j].index(p)] for p in params]
    s = ["%3.1f" % (vals[i] / 10.**exponent[i]) for i in range(len(vals))]
    line = params_exptname[j] + " & " + " & ".join(s) + " \\\\"
    tbl.append(line)

# Output table
print "-"*50
print ""
for t in tbl:
    print t
print ""
