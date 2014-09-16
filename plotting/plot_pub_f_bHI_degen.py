#!/usr/bin/python
"""
Process Fisher matrices for a full experiment and output results in a 
plotting-friendly format.
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

cosmo = experiments.cosmo

# Choose experiment to process
name = "SKA1MK"
root = "output/" + name


################################################################################
# Load data
################################################################################

# Load cosmo fns.
zc, Hc, dAc, Dc, fc = np.genfromtxt(root+"-cosmofns-zc.dat").T
z, H, dA, D, f = np.genfromtxt(root+"-cosmofns-smooth.dat").T

# Load Fisher matrices and P(k) constraints as fn. of z
F_list = []; F_eos_list = []
kc = []; pk = []; pkerr = []
for i in range(zc.size):
    F_list.append( np.genfromtxt(root+"-fisher-%d.dat" % i) )
    F_eos_list.append( np.genfromtxt(root+"-fisher-eos-%d.dat" % i) )
    _kc, _pk, _pkerr = np.genfromtxt(root+"-pk-%d.dat" % i).T
    kc.append(_kc); pk.append(_pk); pkerr.append(_pkerr)

################################################################################
# Load constraints on b_HI(z), f(z), H(z), d_A(z)
################################################################################

Nbins = zc.size
F_b = 0
for i in range(Nbins):
    
    # Trim params we don't care about here
    # A(z), bHI(z), f(z), sig2, dA(z), H(z), [fNL], [omega_k_ng], [omega_DE_ng], pk
    _F = baofisher.fisher_with_excluded_params(F_list[i], [6, 7, 8, 9])
    
    # Expand fns. of z one-by-one for the current z bin. (Indices of fns. of z 
    # are given in reverse order, to make figuring out where they are in the 
    # updated matrix from the prev. step easier.)
    # b_HI(z), f(z), dA(z), H(z)
    zfns_b = [5, 4, 2, 1]
    FF = _F
    for idx in zfns_b:
        FF = baofisher.expand_matrix_for_sampled_fn(FF, idx, Nbins, i)
    F_b += FF


# Overlay error ellipses as a fn. of z
p1 = baofisher.indexes_for_sampled_fns(1, zc.size, zfns_b) # y
p2 = baofisher.indexes_for_sampled_fns(2, zc.size, zfns_b) # x
x = 1.; y = 1.
alpha = [1.52, 2.48, 3.44]

cmap = matplotlib.cm.get_cmap("spectral")
ZMAX = 1.39
print "Max:", np.max(zc)

ax = P.subplot(111)
Nused = np.where(zc <= ZMAX)[0].size # No. of z bins that are actually used
zvals = []
for i in range(len(p1)):
    #if i % 2 == 0: continue
    if zc[i] > ZMAX: continue
    zvals.append(zc[i])
    a, b, ang = baofisher.ellipse_for_fisher_params(p1[i], p2[i], F_b)
    a /= baofisher.bias_HI(zc[i], cosmo)
    b /= fc[i]
    c = i*0.97 / float(Nused - 1) # Colour (must be 0 <= c <= 1)
    print c
    
    # Get 1,2,3-sigma ellipses and plot
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[k]*b, 
                 height=alpha[k]*a, angle=ang, fc='none', ec=cmap(c), lw=1.5) for k in [0,]] #range(0, 2)]
    for e in ellipses: ax.add_patch(e)

P.plot(x, y, 'bx')
P.xlabel(r"Frac. error, $f(z)$", fontdict={'fontsize':'20'})
P.ylabel(r"Frac. error, $b_\mathrm{HI}(z)$", fontdict={'fontsize':'20'})

# Hack to get colorbar on P.plot()
sm = P.cm.ScalarMappable(cmap=cmap)
sm._A = [np.min(zvals), np.max(zvals)/0.97]
cbar = P.colorbar(sm)
cbar.ax.tick_params(labelsize=16)
#cbar.ax.set_title("$z$", fontdict={'fontsize':'20'})

# Display options
fontsize = 18.
for tick in P.gca().yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in P.gca().xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
P.tight_layout()
P.show()
