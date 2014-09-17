#!/usr/bin/python
"""
Plot functions of redshift.
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

# Choose which function of redshift to plot
#fn = 'DA'
#fn = 'H'
fn = 'f'

cosmo = rf.experiments.cosmo

names = ['EuclidRef', 'cexptL', 'iexptM', 'exptS']
colours = ['#CC0000', '#1619A1', '#5B9C0A', '#990A9C'] # DETF/F/M/S
labels = ['DETF IV', 'Facility', 'Mature', 'Snapshot']
linestyle = [[2, 4, 6, 4], [1,0], [8, 4], [3, 4]]

cosmo_fns = rf.background_evolution_splines(cosmo)
#cosmo = rf.load_power_spectrum(cosmo, "cache_pk.dat", force_load=True)

# Fiducial value and plotting
fig = P.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

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
    pnames = ['A', 'b_HI', 'Tb', 'sigma_NL', 'sigma8', 'n_s', 'f', 'aperp', 'apar', 
             'omegak', 'omegaDE', 'w0', 'wa', 'h', 'gamma']
    pnames += ["pk%d" % i for i in range(kc.size)]
    zfns = [0,1,6,7,8]
    excl = [2,4,5,  9,10,11,12,13,14] # Exclude all cosmo params
    excl += [i for i in range(len(pnames)) if "pk" in pnames[i]]
    
    F, lbls = rf.combined_fisher_matrix( F_list,
                                                expand=zfns, names=pnames,
                                                exclude=excl )
    cov = np.linalg.inv(F)
    
    # Get functions of z
    zfns = [0,1,3,4,5] # A, b_HI, f, DA, H
    pA  = rf.indexes_for_sampled_fns(0, zc.size, zfns)
    #pb  = rf.indexes_for_sampled_fns(1, zc.size, zfns)
    pDA = rf.indexes_for_sampled_fns(4, zc.size, zfns)
    pH  = rf.indexes_for_sampled_fns(5, zc.size, zfns)
    pf  = rf.indexes_for_sampled_fns(3, zc.size, zfns)
    
    if k == 0:
        print "DA:", [lbls[j] for j in pDA], "\n"
        print "H:", [lbls[j] for j in pH], "\n"
        print "f:", [lbls[j] for j in pf], "\n"
    
    # Plot errorbars
    errs = np.sqrt(np.diag(cov))
    if fn == 'DA': err = 1e3 * errs[pDA] / dAc
    if fn == 'H': err = 1e2 * errs[pH] / Hc
    if fn == 'f': err = errs[pf] / fc
    
    line = ax1.plot(zc, err, color=colours[k], lw=1.8, marker='o', label=labels[k])
    # Set custom linestyle
    line[0].set_dashes(linestyle[k])
    
    if k == 1:
        if fn == 'DA': ax2.plot(z, dA, color='k', lw=2., alpha=0.4)
        if fn == 'H':  ax2.plot(z, H, color='k', lw=2., alpha=0.4)
        if fn == 'f':  ax2.plot(z, f, color='k', lw=2., alpha=0.4)

# Move subplots
axes = [ax1, ax2]
pos = [ax.get_position().get_points() for ax in axes]
dy = pos[0][0,1] - pos[1][1,1]
l = pos[0][0,0]
w = pos[0][1,0] - pos[0][0,0]
h = pos[0][1,1] - pos[0][0,1]
b = pos[0][0,1]

ax1.set_position([l, b - 0.5*dy - 0.3*h, w, 1.3*h+0.5*dy])
ax2.set_position([l, (b - 0.5*dy - 0.3*h) - (0.6*h+0.5*dy), w, 0.6*h+0.5*dy])

ax1.set_ylim((0., 0.067))
ax1.set_xlim((0.26, 2.2))
ax2.set_xlim((0.26, 2.2)) # 2.4

#ax1.set_ylim((0., 2.52))
if fn == 'DA': ax2.set_ylim((800., 1950.))
if fn == 'H':  ax2.set_ylim((50., 290.))
if fn == 'f':  ax2.set_ylim((0.5, 1.09))


# Resize labels/ticks
fontsize = 18
for i in range(len(axes)):
    ax = axes[i]
    for tick in ax.xaxis.get_major_ticks():
      tick.label1.set_fontsize(fontsize)
      if i == 0: tick.label1.set_visible(False) # Hide x lbls in upper subplots
    for tick in ax.yaxis.get_major_ticks():
      tick.label1.set_fontsize(fontsize)

# y labels
if fn == 'DA':
    ax1.set_ylabel(r"$\sigma_{D_A} / D_A$", fontdict={'fontsize':'20'})
    ax2.set_ylabel(r"$D_A(z)$", fontdict={'fontsize':'20'})
if fn == 'H':
    ax1.set_ylabel(r"$\sigma_H / H$", fontdict={'fontsize':'20'})
    ax2.set_ylabel(r"$H(z)$", fontdict={'fontsize':'20'})
if fn == 'f':
    ax1.set_ylabel(r"$\sigma_f / f$", fontdict={'fontsize':'20'})
    ax2.set_ylabel(r"$f(z)$", fontdict={'fontsize':'20'})

ax2.set_xlabel(r"$z$", fontdict={'fontsize':'20'})
ax1.legend(loc='upper left', prop={'size':'large'})

#P.tight_layout()
# Set size
P.gcf().set_size_inches(8.5, 7.)

#if fn == 'DA': P.savefig('pub-da-z.png', dpi=100)
#if fn == 'H': P.savefig('pub-h-z.png', dpi=100)
#if fn == 'f': P.savefig('pub-f-z.png', dpi=100)

P.show()
