#!/usr/bin/python
"""
Process EOS Fisher matrices and plot P(k).
"""

import numpy as np
import pylab as P
import baofisher
import matplotlib.patches
import matplotlib.cm
from units import *
from mpi4py import MPI
import experiments
import os
import euclid

cosmo = experiments.cosmo

names = ['EuclidRef', 'cexptL', 'iexptM', 'exptS']
colours = ['#CC0000', '#1619A1', '#5B9C0A', '#990A9C'] # DETF/F/M/S
labels = ['DETF IV', 'Facility', 'Mature', 'Snapshot']
linestyle = [[2, 4, 6, 4], [1,0], [8, 4], [3, 4]]

# Get f_bao(k) function
cosmo = baofisher.load_power_spectrum(cosmo, "cache_pk.dat", force_load=True)
fbao = cosmo['fbao']

# Fiducial value and plotting
P.subplot(111)

for k in range(len(names)):
    root = "output/" + names[k]

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
    pnames = ['A', 'b_HI', 'Tb', 'sigma_NL', 'sigma8', 'n_s', 'f', 'aperp', 'apar', 
             'omegak', 'omegaDE', 'w0', 'wa', 'h', 'gamma']
    #if "Euclid" not in names[k]: pnames.append('fNL')
    pnames += ["pk%d" % i for i in range(kc.size)]
    zfns = []; excl = []
    F, lbls = baofisher.combined_fisher_matrix( F_list,
                                                expand=zfns, names=pnames,
                                                exclude=excl )
    
    # Just do the simplest thing for P(k) and get 1/sqrt(F)
    cov = [np.sqrt(1. / np.diag(F)[lbls.index(lbl)]) for lbl in lbls if "pk" in lbl]
    cov = np.array(cov)
    pk = cosmo['pk_nobao'](kc) * (1. + fbao(kc))
    
    # Replace nan/inf values
    cov[np.where(np.isnan(cov))] = 1e10
    cov[np.where(np.isinf(cov))] = 1e10
    
    pw0 = baofisher.indexes_for_sampled_fns(11, zc.size, zfns)
    pwa = baofisher.indexes_for_sampled_fns(12, zc.size, zfns)
    
    print "-"*50
    print names[k]
    #print cov
    print lbls[pw0], 1. / np.sqrt(F[pw0,pw0])
    print lbls[pwa], 1. / np.sqrt(F[pwa,pwa])
    
    """
    if k == 0:
        # Plot shaded region
        P.fill_between(kc, np.ones(kc.size)*1e-10, cov, facecolor='#e1e1e1', edgecolor='none')
    else:
        # Plot errorbars
        P.plot(kc, cov, color=colours[k], label=labels[k], lw=2.2, ls=linestyle[k])
    """
    line = P.plot(kc, cov, color=colours[k], label=labels[k], lw=2.4)
    
    # Set custom linestyle    
    line[0].set_dashes(linestyle[k])


P.xscale('log')
P.yscale('log')
P.xlim((2e-3, 3e0))
P.ylim((9e-4, 1e1))
P.legend(loc='lower left', prop={'size':'large'})

# Resize labels/ticks
fontsize = 18
ax = P.gca()
for tick in ax.xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in ax.yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)

P.xlabel(r"$k \,[\mathrm{Mpc}^{-1}]$", fontdict={'fontsize':'20'})
P.ylabel(r"$\Delta P / P$", fontdict={'fontsize':'20'})

P.tight_layout()
# Set size
P.gcf().set_size_inches(8.,6.)
P.savefig('pub-dlogp.pdf', transparent=True) # 100

P.show()
