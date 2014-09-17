#!/usr/bin/python
"""
Plot P(k) constraints at a given scale, as a function of redshift.
"""
import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
from units import *

import os, sys

cosmo = rf.experiments.cosmo

#names = ['SKA1MIDfull', 'SKA1MIDfull2', 'iSKA1MIDfull1', 'iSKA1MIDfull2', 'fSKA1SURfull1', 'fSKA1SURfull2']
names = ['SKA0MID', 'SKA0SUR', 'SKA1MID900', 'SKA1MID350', 'iSKA1MID900', 
         'iSKA1MID350', 'fSKA1SUR650', 'fSKA1SUR350']
colours = ['#1619A1', '#5B9C0A', '#990A9C', '#FFB928', '#CC0000', 'c', '#ff6600', 'k', 'y']

# Define which k bin and which band to use
try:
    kbin = int(sys.argv[1])
    ids = [int(sys.argv[i]) for i in range(2, len(sys.argv))]
except:
    print "Expects 1+N arguments: int(k_bin_idx), list_ints(expt_ids)"
    sys.exit(1)

#kbin = 52 #70 #40 #52 # 33
#BAND = 1
sarea = [50, 100, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000]

# Fiducial value and plotting
P.subplot(111)

for k in ids:
    for s in sarea:
        root = "../output/%s_nofg_%d" % (names[k], s)
        
        try:
            # Load cosmo fns.
            dat = np.atleast_2d( np.genfromtxt(root+"-cosmofns-zc.dat") ).T
            zc, Hc, dAc, Dc, fc = dat
            z, H, dA, D, f = np.genfromtxt(root+"-cosmofns-smooth.dat").T
            kc = np.genfromtxt(root+"-fisher-kc.dat").T

            # Load Fisher matrices as fn. of z
            Nbins = zc.size
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
        
        alpha = 0.1 + 0.9*( float(sarea.index(s)) / float(len(sarea)) )
        
        line = P.plot(zc, pk_err, lw=1.5, color=colours[k],
                      label="%s (%d deg^2)" % (names[k], s), alpha=alpha,
                      marker='.' )
        #line[0].set_dashes(linestyle[k])

for i in range(kc.size):
    print "%03d -- %3.3e" % (i, kc[i])

# Title is k value of k bin
P.title("k = %3.3e Mpc$^{-1}$" % kc[kbin])

P.legend(loc='lower right', prop={'size':'xx-small'}, ncol=len(ids), frameon=False)

P.tick_params(axis='both', which='major', labelsize=20, size=8., width=1.5, pad=8.)
P.tick_params(axis='both', which='minor', labelsize=20, size=5., width=1.5)

P.xlabel(r"$z$", fontdict={'fontsize':'xx-large'})
P.ylabel(r"$\Delta P / P$", fontdict={'fontsize':'xx-large'})

P.ylim((1e-3, 1e1))
P.yscale('log')

P.tight_layout()
P.gcf().set_size_inches(20.,16.)
P.savefig("pk_redshift_k%3.3e.png" % kc[kbin], transparent=False)
print "Saved: pk_redshift_k%3.3e.png" % kc[kbin]
#P.show()
