#!/usr/bin/python
"""
Plot P(k) constraints at a given scale, as a function of redshift (from a given 
list of rf.experiments.scales).
"""
import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
from units import *

import os, sys

cosmo = rf.experiments.cosmo

#names = ['SKA1MIDfull1', 'iSKA1MIDfull1', 'fSKA1SURfull1', 'SKA1MIDfull2', 'fSKA1SURfull2']
#labels = ['SKA1-MID Full B1 Auto.', 'SKA1-MID Full B1 Interferom.', 'SKA1-SUR Full B1 Auto.', 'SKA1-MID Full B2 Auto.', 'SKA1-SUR Full B2 Auto.']

names = ['SKA0MID', 'SKA0SUR', 'SKA1MID900', 'SKA1MID350', 'iSKA1MID900', 
         'iSKA1MID350', 'fSKA1SUR650', 'fSKA1SUR350']
labels = names
colours = ['#1619A1', '#5B9C0A', '#990A9C', '#FFB928', '#CC0000', 'y']
colours = ['#1619A1', '#5B9C0A', '#990A9C', '#FFB928', '#CC0000', 'c', '#ff6600', 'k', 'y']
sarea = [25000, 25000, 25000, 25000, 5000, 5000, 25000, 25000]
try:
    kbin = int(sys.argv[1])
except:
    print "Expects 1 argument: int(k_bin_idx)"
    sys.exit(1)

# Fiducial value and plotting
P.subplot(111)

for k in range(len(names)):
    root = "../output/%s_nofg_%d" % (names[k], sarea[k])

    # Load cosmo fns.
    dat = np.atleast_2d( np.genfromtxt(root+"-cosmofns-zc.dat") ).T
    zc, Hc, dAc, Dc, fc = dat
    z, H, dA, D, f = np.genfromtxt(root+"-cosmofns-smooth.dat").T
    kc = np.genfromtxt(root+"-fisher-kc.dat").T

    # Load Fisher matrices as fn. of z
    Nbins = zc.size
    try:
        F_list = [np.genfromtxt(root+"-fisher-full-%d.dat" % i) for i in range(Nbins)]
    except:
        print "*** FAILED:", root
        continue
    
    # EOS FISHER MATRIX
    # Actually, (aperp, apar) are (D_A, H)
    pnames = rf.load_param_names(root+"-fisher-full-0.dat")
    ppk = rf.indices_for_param_names(pnames, 'pk*')
    
    pk_err = np.zeros(len(F_list))
    for j in range(len(F_list))[::-1]:
        F = F_list[j]
        
        # Just do the simplest thing for P(k) and get 1/sqrt(F)
        cov = np.sqrt(1. / np.diag(F)[ppk])
        #pk = cosmo['pk_nobao'](kc) * (1. + fbao(kc))
        
        # Replace nan/inf values
        cov[np.where(np.isnan(cov))] = 1e10
        cov[np.where(np.isinf(cov))] = 1e10
        
        # Add to correct z bin
        pk_err[j] = cov[kbin]
    
    line = P.plot(zc, pk_err, lw=2., color=colours[k],
                  label="%s (%d deg^2)" % (labels[k], sarea[k]), marker='.' )
    #line[0].set_dashes(linestyle[k])

for i in range(kc.size):
    print "%03d -- %3.3e" % (i, kc[i])

# Title is k value of k bin
P.title("k = %3.3e Mpc$^{-1}$" % kc[kbin])

P.legend(loc='upper right', prop={'size':'medium'}, frameon=False)

P.tick_params(axis='both', which='major', labelsize=20, size=8., width=1.5, pad=8.)
P.tick_params(axis='both', which='minor', labelsize=20, size=5., width=1.5)

P.xlabel(r"$z$", fontdict={'fontsize':'x-large'})
P.ylabel(r"$\Delta P / P$", fontdict={'fontsize':'x-large'})

P.ylim((5e-3, 5e-1))
P.yscale('log')

P.gcf().set_size_inches(8.,6.)
P.tight_layout()
P.savefig("pk_redshift_k%3.3e.pdf" % kc[kbin], transparent=True)
print "Output: pk_redshift_k%3.3e.pdf" % kc[kbin]
#P.show()
