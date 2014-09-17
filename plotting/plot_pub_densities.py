#!/usr/bin/python
"""
Process EOS Fisher matrices and overplot results for several rf.experiments.
"""

import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
from units import *
from mpi4py import MPI

import os

cosmo = rf.experiments.cosmo

#names = ["GBT", "BINGO", "WSRT", "APERTIF", "JVLA", "ASKAP", "KAT7", "MeerKAT", "SKA1mid", "SKA1MK", "iSKA1MK", "aSKA1MK", "SKA1MK_A0"]
names = ["MeerKAT", "SKA1mid", "SKA1MK"]
#colours = ['#22AD1A', '#3399FF', '#ED7624']
colours = ['r', 'g', 'b']

k = 0
root = "../output/" + names[k]

# Fiducial value and plotting
x = rf.experiments.cosmo['omega_lambda_0']; y = rf.experiments.cosmo['omega_k_0']
alpha = [1.52, 2.48, 3.44]

# Load cosmo fns.
dat = np.atleast_2d( np.genfromtxt(root+"-cosmofns-zc.dat") ).T
zc, Hc, dAc, Dc, fc = dat
z, H, dA, D, f = np.genfromtxt(root+"-cosmofns-smooth.dat").T

# Load Fisher matrices and P(k) constraints as fn. of z
F_list = []; F_eos_list = []
kc = []; pk = []; pkerr = []
for i in range(zc.size):
    F_list.append( np.genfromtxt(root+"-fisher-%d.dat" % i) )
    F_eos_list.append( np.genfromtxt(root+"-fisher-eos-%d.dat" % i) )
    _kc, _pk, _pkerr = np.genfromtxt(root+"-pk-%d.dat" % i).T
    kc.append(_kc); pk.append(_pk); pkerr.append(_pkerr)

Nbins = zc.size
F_base = 0; F_wa = 0; F_w0wa = 0
for i in range(Nbins):
    
    # Trim params we don't care about here
    # A, b_HI, f(z), sigma_NL, omega_k, omega_DE, w0, wa
    _F = F_eos_list[i]
    
    # Expand fns. of z one-by-one for the current z bin. (Indices of fns. of z 
    # are given in reverse order, to make figuring out where they are in the 
    # updated matrix from the prev. step easier.)
    
    # b_HI(z), f(z)
    zfns_base = [2, ]
    FF = _F
    for idx in zfns_base:
        FF = rf.expand_matrix_for_sampled_fn(FF, idx, Nbins, i)
    F_base += FF
    
    # b_HI(z), f(z), fix w_a
    _F = rf.fisher_with_excluded_params(_F, [7,])
    zfns_fix_wa = [2, ]
    FF = _F
    for idx in zfns_fix_wa:
        FF = rf.expand_matrix_for_sampled_fn(FF, idx, Nbins, i)
    F_wa += FF
    
    # b_HI(z), f(z), fix w_0, w_a
    _F = F_eos_list[i]
    _F = rf.fisher_with_excluded_params(_F, [6, 7])
    zfns_fix_w0wa = [2, ]
    FF = _F
    for idx in zfns_fix_w0wa:
        FF = rf.expand_matrix_for_sampled_fn(FF, idx, Nbins, i)
    F_w0wa += FF
    

ax = P.subplot(111)

# Overlay error ellipses as a fn. of z
p1 = rf.indexes_for_sampled_fns(4, zc.size, zfns_base) # omega_k # y
p2 = rf.indexes_for_sampled_fns(5, zc.size, zfns_base) # omega_de # x

# 1D marginals
cov = np.linalg.inv(F_wa)
print "sigma(ok) =", np.sqrt(cov[p1,p1])
print "sigma(oDE) =", np.sqrt(cov[p2,p2])

y = rf.experiments.cosmo['omega_k_0']
x = rf.experiments.cosmo['omega_lambda_0']

Fs = [F_base, F_wa, F_w0wa]
for i in range(len(Fs)):
    FF = Fs[i]
    a, b, ang = rf.ellipse_for_fisher_params(p1, p2, FF)
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*b, 
                 height=alpha[kk]*a, angle=ang, fc='none', ec=colours[i], lw=2., alpha=0.85) for kk in range(0, 2)]
    for e in ellipses: ax.add_patch(e)
ax.plot(x, y, 'kx')


ax.set_ylabel(r"$\Omega_K$", fontdict={'fontsize':'20'})
ax.set_xlabel(r"$\Omega_\mathrm{DE}$", fontdict={'fontsize':'20'})

fontsize = 16.
for tick in ax.yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in ax.xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
P.tight_layout()
P.show()
