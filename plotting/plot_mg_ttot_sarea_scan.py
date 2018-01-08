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
import os, sys, copy

cosmo = rf.experiments.cosmo

# Parameters to always exclude
#exclude_all = ['Tb', 'f', 'aperp', 'apar', 'DA', 'H', 'N_eff', 'pk*', 'gamma', 'fs8', 'bs8', 'b_1',]
exclude_all = ['Tb', 'f', 'DA', 'H', 'pk*', 'gamma', 'fs8', 'bs8',]

# Which MG parameters to exclude for each parametrisation
MGMODE = 1
mgexclude = [ ['gamma0', 'gamma1', 'eta0', 'eta1', 'f0k*'],          # Tessa's MG params
              ['gamma1', 'eta0', 'eta1', 'f0k*', 'A_xi', 'logkmg'],                                      # Basic gamma=const.
              ['eta0', 'eta1', 'alphaxi', 'f0k*', 'A_xi', 'logkmg'],   # Extended gamma(z)
              ['gamma1', 'eta1', 'alphaxi', 'f0k*', 'A_xi', 'logkmg'], # Basic eta=const.
              ['gamma1', 'alphaxi', 'f0k*', 'A_xi', 'logkmg'],         # Extended eta(z)
            ]
BASE_NAMES = [ 'MID_B1_Base', 'MID_B1_Alt', 'MID_B2_Base', 'MID_B2_Upd', 
               'MID_B2_Alt', 'aLOW_Base', 'aLOW_Upd', 'aLOW_Alt' ]

if len(sys.argv) > 1:
    eid = int(sys.argv[1])
    print "-"*50
    print BASE_NAMES[eid]
    print "-"*50
else:
    print "Missing argument. Must specify experiment ID as integer."
    exit()

base_name = BASE_NAMES[eid]
sareas = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12500, 15000, 17500, 20000, 22500, 25000, 27500, 30000]
ttots = [1, 3, 5, 10]

"""
################################################################################
# Load low-z galaxy survey Fisher matrix
root = "output/" + "BOSS_mgD"

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
excl = exclude_all + mgexclude[MGMODE]
F_lowz, lbls_lowz = rf.combined_fisher_matrix( F_list,
                                               expand=zfns, names=pnames,
                                               exclude=excl )
# Relabel galaxy bias from low-z survey
for i in range(len(lbls_lowz)):
    if "b_HI" in lbls_lowz[i]: lbls_lowz[i] = "lowz%s" % lbls_lowz[i]
"""
################################################################################

# Loop over survey areas and times
for ttot in ttots:
    for sarea in sareas:
        root = "output/%s_surv_ttot%dk_%d" % (base_name, ttot, sarea)
        #print base_name, ttot, sarea
        
        # See if this experiment/sarea/ttot exists
        try:
            # Load cosmo fns.
            dat = np.atleast_2d( np.genfromtxt(root+"-cosmofns-zc.dat") ).T
            zc, Hc, dAc, Dc, fc = dat
            zs, Hs, dAs, Ds, fs = np.genfromtxt(root+"-cosmofns-smooth.dat").T
            kc = np.genfromtxt(root+"-fisher-kc.dat").T
        except:
            print "*** NOT FOUND:", root
            continue
        
        # Load Fisher matrices as fn. of z
        Nbins = zc.size
        F_list = [np.genfromtxt(root+"-fisher-full-%d.dat" % i) for i in range(Nbins)]
        
        # EOS FISHER MATRIX
        pnames = rf.load_param_names(root+"-fisher-full-0.dat")
        zfns = ['b_HI',]
        excl = exclude_all + mgexclude[MGMODE]
        F, lbls = rf.combined_fisher_matrix( F_list, expand=zfns, names=pnames,
                                             exclude=excl )
        
        # Relabel galaxy bias from low-z survey and sum current survey + low-z
        #if 'BOSS' not in labels[k]:
        #    F, lbls = rf.add_fisher_matrices(F_lowz, F, lbls_lowz, lbls, expand=True)
        
        # Add Planck prior
        #print "*** Using DETF Planck prior ***"
        l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h', 'sigma8']
        F_detf = euclid.detf_to_rf("DETF_PLANCK_FISHER.txt", cosmo, omegab=False)
        Fpl, lbls = rf.add_fisher_matrices(F, F_detf, lbls, l2, expand=True)
        
        # Decide whether to fix various parameters
        fixed_params = []
        #fixed_params = ['omegak', 'n_s', 'sigma8', 'omega_b',]
        if len(fixed_params) > 0:
            Fpl, lbls = rf.combined_fisher_matrix( [Fpl,], expand=[], 
                         names=lbls, exclude=fixed_params )
        #print lbls
        
        # Invert matrix
        cov_pl = np.linalg.inv(Fpl)
        
        # Print 1D marginals
        print "%s %d %d" % (base_name, ttot*1e3, sarea),
        print np.sqrt(cov_pl[lbls.index('w0'),lbls.index('w0')]), # w0
        print np.sqrt(cov_pl[lbls.index('gamma0'),lbls.index('gamma0')])  # gamma
    
