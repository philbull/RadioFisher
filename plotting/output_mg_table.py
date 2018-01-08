#!/usr/bin/python
"""
Plot 2D constraints on (w0, wa), with a low-z survey added in for good measure.
"""
import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
import matplotlib.ticker
from radiofisher.units import *
import radiofisher.euclid as euclid
import os, copy

cosmo = rf.experiments.cosmo

exclude_all = ['Tb', 'f', 'aperp', 'apar', 'DA', 'H', 'N_eff', 'pk*', 'fs8', 'b_1', 'sigma8tot', 'k*', 'gamma', 'bs8', ]

# Experiments
#names = ['exptS_mgtest', 'exptS_mgtest_fnl1', 'exptS_mgtest_fnl10'] #'SKA1MID350_mg_Dz_kmg0.01', 'gSKA2_mg_Dz_kmg0.01', 'EuclidRef_mg_Dz_kmg0.01']
#labels = ['test', 'test fNL1', 'test fNL10', 'SKA1-IM', 'SKA2', 'Euclid']

#names = ['hMID_B1_Octave_mgD', 'hMID_B2_Octave_mgD', 'gSKA2MG_mgD', 'EuclidRef_mgD',]# 'aLOW_Alt_mgD',]
#labels = ['MID1', 'MID2', 'SKA2', r'H$\alpha$',] # 'LOW',]

names = ['hMID_B1_Octave_mg', 'hMID_B2_Octave_mg', 'gSKA2MG_mg', 'EuclidRef_mg',]# 'aLOW_Alt_mgD',]
labels = ['MID1', 'MID2', 'SKA2', r'H$\alpha$',] # 'LOW',]


# List of parameters to include in table
paramlist = ['h', 'sigma_8', 'w0', 'wa', 'A_xi', 'k_mg', 'gamma0', 'gamma1', 'eta0', 'eta1',] # 'f_NL', 'f0k0', 'f0k1', 'f0k2']
paramnames = [r'$\bm{h}$', r'$\bm{\sigma_8}$', r'\bm{$w_0}$', r'$\bm{w_a}$', r"$\bm{A_\xi}$", r"$\bm{k_\xi}$", r'$\bm{\gamma_0}$', r'$\bm{\gamma_1}$', r'$\bm{\eta_0}$', r'$\bm{\eta_1}$', ] #'fNL', 'f0k0', 'f0k1', 'f0k2',]
scale =     [1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1., 1e-2, 1e-2, 1e-2]

fixedp = [ #['gamma0', 'gamma1', 'eta0', 'eta1'],
           ['A_xi', 'logkmg', 'gamma1', 'eta0', 'eta1'],
           ['A_xi', 'logkmg', 'eta0', 'eta1'],
           ['A_xi', 'logkmg', 'gamma1', 'eta1'],
           ['A_xi', 'logkmg', 'gamma0', 'gamma1'],
           #['A_xi', 'logkmg'],
         ]
tbl = []



################################################################################
# Load low-z galaxy survey Fisher matrix
root = "output/" + "BOSS_mg"

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
#excl = ['Tb', 'f', 'aperp', 'apar', 'DA', 'H', 'N_eff', 'pk*', 'gamma', 'fs8', 'bs8']
excl = exclude_all

F_lowz, lbls_lowz = rf.combined_fisher_matrix( F_list,
                                               expand=zfns, names=pnames,
                                               exclude=excl )
# Relabel galaxy bias from low-z survey
for i in range(len(lbls_lowz)):
    if "b_HI" in lbls_lowz[i]: lbls_lowz[i] = "lowz%s" % lbls_lowz[i]

################################################################################


# Fiducial value and plotting
fig = P.figure()
ax = fig.add_subplot(111)

_k = range(len(names)) #[::-1]
for k in _k:
    root = "output/" + names[k]

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
    excl = exclude_all
    #excl = ['Tb', 'f', 'aperp', 'apar', 'DA', 'H', 'N_eff', 'pk*', 'gamma', 'fs8', 'bs8'] #, 'f_NL']
    F, lbls = rf.combined_fisher_matrix( F_list, expand=zfns, names=pnames,
                                         exclude=excl )
    
    # Relabel galaxy bias from low-z survey and sum current survey + low-z
    F, lbls = rf.add_fisher_matrices(F_lowz, F, lbls_lowz, lbls, expand=True)
    
    # Add Planck prior
    print "*** Using DETF Planck prior ***"
    l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h', 'sigma_8']
    F_detf = euclid.detf_to_rf("DETF_PLANCK_FISHER.txt", cosmo, omegab=False)
    Fpl, lbls = rf.add_fisher_matrices(F, F_detf, lbls, l2, expand=True)
    
    for l, f in zip(lbls, np.diag(Fpl)):
        print "%10s : %3.3e" % (l, f)
    print names[k]
    
    #Fpl[lbls.index('h'), lbls.index('h')] += 1./(0.005*0.7)**2.
    
    # Loop through sets of fixed parameters
    blk = []
    for fixed_params in fixedp:
        print fixed_params
        Fx, lblx = rf.combined_fisher_matrix( [Fpl,], expand=[], 
                     names=lbls, exclude=fixed_params )
        # Invert matrix
        cov = np.linalg.inv(Fx)
        
        print "****", names[k], lblx
        
        clmn = ["%d" % fixedp.index(fixed_params), "%s" % labels[k],]
        for p in paramlist:
            if p not in lblx:
                clmn.append("---")
            else:
                sc = scale[paramlist.index(p)]
                clmn.append("%3.1f" % (np.sqrt(cov[lblx.index(p), lblx.index(p)]) / sc))
        blk.append(clmn)
        print "-"*30
    tbl.append(blk)

# Print table
for i in range(len(paramlist)+2):
    row = []
    for p in range(len(fixedp)):
        for e in range(len(names)):
            row.append(tbl[e][p][i])
    if i >= 2:
        print paramnames[i-2], "&", " & ".join(row), "\\\\"
    else:
        print " ", "&", " & ".join(row), "\\\\"
