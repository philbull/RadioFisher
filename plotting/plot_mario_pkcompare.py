#!/usr/bin/python
"""
Process EOS Fisher matrices and plot P(k).
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

#names = ["GBT", "BINGO", "WSRT", "APERTIF", "JVLA", "ASKAP", "KAT7", "MeerKAT", "SKA1", "SKAMID", "SKAMID_COMP", "iSKAMID", "iSKAMID_COMP", "SKA1_CV"]
#names = ["SKA1", "SKAMID", "SKAMID_COMP", "iSKAMID", "iSKAMID_COMP"]
names = ["SKAMID", "iSKAMID",] # "iSKAMID_COMP", "iSKAMID_BIGZ", "iSKAMID_COMP_BIGZ", "iSKA_CORE", "iSKA_CORE2", "iSKA_CORE3"] #"iSKAMID_COMP_BIGZ", "iSKA_CORE"]

#ls = ['k-', 'r-', 'b--', 'm-', 'c--']
cols = ['b', 'r']
labels = ['SKAMID single-dish', 'SKAMID interferom.']

colours = ['#22AD1A', '#3399FF', '#ED7624']

# Get f_bao(k) function
print "(A) ----------------------------------"
cosmo_fns, cosmo = rf.precompute_for_fisher(rf.experiments.cosmo, "camb/rf_matterpower.dat")
print "(B) ----------------------------------"
fbao = cosmo['fbao']

P.subplot(211)

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
             'omegak', 'omegaDE', 'w0', 'wa', 'h', 'gamma', 'fNL']
    pnames += ["pk%d" % i for i in range(kc.size)]
    zfns = [1,]
    F, lbls = rf.combined_fisher_matrix( F_list,
                                                expand=zfns, names=pnames,
                                                exclude=[2,4,5,6,7,8 ] )
    # Just do the simplest thing for P(k)
    cov = np.sqrt(1. / np.diag(F)[-kc.size:])
    cov[np.where(np.isinf(cov))] = 1e4
    
    #pk = cosmo['pk_nobao'](kc) * (1. + fbao(kc))
    P.plot(kc, cov, lw=1.5, label=labels[k])



P.legend(loc='upper left', prop={'size':'large'})
P.ylim((1e-3, 400.))
P.xlim((3e-3, 1e0))
P.xscale('log')
P.yscale('log')
P.axvline(2.3e-2, ls='dotted', color='k')
P.axvline(3.7e-1, ls='dotted', color='k')

fontsize = 18
for tick in P.gca().xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in P.gca().yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)

P.ylabel("$\Delta P / P$", fontdict={'fontsize':'20'})
#P.xlabel("$k \,[\mathrm{Mpc}^{-1}]$", fontdict={'fontsize':'20'})



P.subplot(212)
kk = np.logspace(-3., 1., 2000)
P.plot(kk, fbao(kk), 'k-', lw=2.5, alpha=0.6)

P.axvline(2.3e-2, ls='dotted', color='k')
P.axvline(3.7e-1, ls='dotted', color='k')
P.xscale('log')
P.xlim((3e-3, 1e0))

P.ylabel("$f_\mathrm{BAO}(k)$", fontdict={'fontsize':'20'})
P.xlabel("$k \,[\mathrm{Mpc}^{-1}]$", fontdict={'fontsize':'20'})

fontsize = 18
for tick in P.gca().xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in P.gca().yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)

P.gcf().set_size_inches(16.5,10.5)
P.savefig('mario-pk-compare.png', dpi=100)

P.show()
exit()




# Hide x labels in upper subplot
for tick in ax1.xaxis.get_major_ticks():
  tick.label1.set_visible(False)

fontsize = 18
for tick in ax2.xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in ax1.yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in ax2.yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)

P.set_xlabel(r"$k \,[\mathrm{Mpc}^{-1}]$", fontdict={'fontsize':'20'})
#ax.set_ylabel(r"$P(k)$", fontdict={'fontsize':'20'})

# Set size
P.gcf().set_size_inches(16.5,10.5)
P.savefig('mario-pk-compare.png', dpi=100)

P.show()
