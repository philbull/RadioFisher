#!/usr/bin/python
"""
Process Fisher matrices for a full experiment and output results in a 
plotting-friendly format.
"""

import numpy as np
import pylab as P
import baofisher
import matplotlib.patches
from units import *
from mpi4py import MPI
import experiments
import os

names = ["SKA1MK_alpha_all", "SKA1MK_alpha1", "SKA1MK_alpha2", "SKA1MK_alpha3", "SKA1MK_alpha4", "SKA1MK_alpha4a", "SKA1MK_alpha4b"]
labels = ["All", "Volume", "RSD angle", "RSD shift", "P(k) shift", "  > BAO shift", "  > P_smooth(k) shift"]
col = ['k', 'r', 'g', 'b', 'y', 'c', 'm']

w_perp = []; w_par = []

for j in range(len(names)):
    # Choose experiment to process
    root = "output/" + names[j]

    # Load cosmo fns.
    zc, Hc, dAc, Dc, fc = np.genfromtxt(root+"-cosmofns-zc.dat").T

    # Load Fisher matrices and P(k) constraints as fn. of z
    F_list = []; F_eos_list = []
    kc = []; pk = []; pkerr = []
    for i in range(zc.size):
        F_list.append( np.genfromtxt(root+"-fisher-alpha-%d.dat" % i) )

    Nbins = zc.size
    F_base = 0; F_a = 0; F_b = 0; F_all = 0
    alpha_perp = []
    alpha_par = []
    for i in range(Nbins):
        
        # Trim params we don't care about here
        # A(z), bHI(z), f(z), sig2, aperp(z), apar(z), [fNL], [omega_k_ng], [omega_DE_ng], pk
        _F = baofisher.fisher_with_excluded_params(F_list[i], [0, 1, 2, 3, 6, 7, 8, 9])
        
        # Only get 1D marginals (some of 2D marginals are degenerate)
        alpha_perp.append(_F[0,0])
        alpha_par.append(_F[1,1])
    w_perp.append(alpha_perp)
    w_par.append(alpha_par)

# Calculate fractional weights
wtot_perp = np.zeros(zc.size)
wtot_par = np.zeros(zc.size)
for j in range(1, len(names)):
    wtot_perp += w_perp[j]
    wtot_par += w_par[j]

# Plot results
P.subplot(211)
P.axhline(1., ls='dotted', color='k')
#P.title(r"$\alpha_\perp$")
sig = np.zeros(zc.size)
for j in range(1, len(names)):
    P.plot(zc, w_perp[j] / wtot_perp, color=col[j], ls='solid', lw=1.5, label=labels[j])
P.yscale('log')
P.ylim((7e-3, 1.5))
P.xlim((np.min(zc), np.max(zc)*1.001))
P.legend(loc='upper right', prop={'size':'medium'}) #prop={'size':'large'})

P.ylabel(r"$w_i / w_\mathrm{tot} \,(\alpha_\perp)$", fontdict={'fontsize':'22'})
P.xlabel("$z$", fontdict={'fontsize':'20'})
fontsize = 18.
for tick in P.gca().yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in P.gca().xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)


P.subplot(212)
P.axhline(1., ls='dotted', color='k')
#P.title(r"$\alpha_\parallel$")
for j in range(1, len(names)):
    P.plot(zc, w_par[j] / wtot_par, color=col[j], ls='solid', lw=1.5, label=labels[j])

P.yscale('log')
P.ylim((7e-3, 1.5))
P.xlim((np.min(zc), np.max(zc)*1.001))

P.ylabel(r"$w_i / w_\mathrm{tot} \,(\alpha_\parallel)$", fontdict={'fontsize':'22'})
P.xlabel("$z$", fontdict={'fontsize':'20'})
fontsize = 18.
for tick in P.gca().yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in P.gca().xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)

P.show()

