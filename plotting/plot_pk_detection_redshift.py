#!/usr/bin/python
"""
Plot dP/P as a function of redshift, for a single k bin.
"""
import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
from units import *
from mpi4py import MPI

import os
import euclid

cosmo = rf.experiments.cosmo

names = ['EuclidRef_onebin', 'SKAHI100_onebin30k', 'SKAHI100_onebin', 'iSKA1MIDbase1_onebin', 'SKA1MIDbase1_onebin', 'iSKA1MIDfull1_onebin', 'SKA1MIDfull1_onebin', 'iSKA1MIDfull2_onebin', 'SKA1MIDfull2_onebin']
colours = ['#990A9C', 'c', 'c', '#CC0000', '#CC0000', '#1619A1', '#1619A1', '#5B9C0A', '#5B9C0A'] # DETF/F/M/S
labels = ['Euclid', 'SKA1 HI gal. (30k)', 'SKA1 HI gal. (5k)', 'SKA1-MID (190) Band 1 Int.', 'SKA1-MID (190) Band 1 Auto.', 'SKA1-MID Band 1 Int.', 'SKA1-MID Band 1 Auto.', 'SKA1-MID Band 2 Int.', 'SKA1-MID Band 2 Auto.']
linestyle = [ [1,0], [8, 4], [1,0], [8, 4], [1,0], [8, 4], [1,0], [8, 4], [1,0],]

# Get f_bao(k) function
#cosmo = rf.load_power_spectrum(cosmo, "cache_pk.dat", force_load=True)
#fbao = cosmo['fbao']

# Fiducial value and plotting
P.subplot(111)

for k in range(len(names)):
    root = "../output/" + names[k]

    # Load cosmo fns.
    dat = np.atleast_2d( np.genfromtxt(root+"-cosmofns-zc.dat") ).T
    zc, Hc, dAc, Dc, fc = dat
    z, H, dA, D, f = np.genfromtxt(root+"-cosmofns-smooth.dat").T
    kc = np.genfromtxt(root+"-fisher-kc.dat").T

    # Load Fisher matrices as fn. of z
    Nbins = zc.size
    F_list = [np.genfromtxt(root+"-fisher-full-%d.dat" % i) for i in range(Nbins)]
    
    # EOS FISHER MATRIX
    # Actually, (aperp, apar) are (D_A, H)
    pnames = rf.load_param_names(root+"-fisher-full-0.dat")
    lbls = pnames
    
    dP = []
    for j in range(zc.size):
        cov = [ np.sqrt(1. / np.diag(F_list[j])[lbls.index(lbl)]) 
                  for lbl in lbls if "pk" in lbl ]
        #print np.diag(F_list[j])[lbls.index(lbl)]
        dP.append(cov[0]) # Only one bin
    
    line = P.plot(zc, dP, lw=2.4, color=colours[k], marker='o', label=labels[k])
    line[0].set_dashes(linestyle[k]) # Set custom linestyle


#P.xscale('log')
P.yscale('log')
#P.xlim((1.5e-3, 3e0))
#P.ylim((9e-4, 1e1))
P.legend(loc='upper right', prop={'size':'medium'}, frameon=False, ncol=1)

P.tick_params(axis='both', which='major', labelsize=20, size=8., width=1.5, pad=8.)
P.tick_params(axis='both', which='minor', labelsize=20, size=5., width=1.5)

P.xlabel(r"$z$", fontdict={'fontsize':'xx-large'})
P.ylabel(r"$\Delta P / P$", fontdict={'fontsize':'xx-large'})

P.figtext(0.65, 0.15, "$10^{-4} < k < 10^{1} \,\mathrm{Mpc}^{-1}$", fontsize=18)

P.tight_layout()
# Set size
P.gcf().set_size_inches(8.,6.)
P.savefig('ska-pk-detectability2.pdf', transparent=True) # 100

P.show()
