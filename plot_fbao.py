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

names = ["cexptL", "iexptM", "exptS"]

#colours = ['#CC0000', '#ED5F21', '#FAE300', '#5B9C0A', '#1619A1', '#56129F', '#990A9C']
colours = ['#CC0000', '#5B9C0A', '#1619A1']
labels = ['Behemoth', 'Mature', 'Snapshot']

names = ["SKA1MID", "cSKA1MID", "SKA1SUR"]
labels = ["SKA1-MID (Dish)", "SKA1-MID (Comb.)", "SKA1-SUR (Dish)"]

# Get f_bao(k) function
cosmo_fns, cosmo = baofisher.precompute_for_fisher(experiments.cosmo, "camb/baofisher_matterpower.dat")
fbao = cosmo['fbao']

# Fiducial value and plotting
fig = P.figure()
axes = [fig.add_subplot(311), fig.add_subplot(312), fig.add_subplot(313)]

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
    zfns = []; excl = []
    F, lbls = baofisher.combined_fisher_matrix( F_list,
                                                expand=zfns, names=pnames,
                                                exclude=excl )
    
    # Just do the simplest thing for P(k) and get 1/sqrt(F)
    cov = [np.sqrt(1. / np.diag(F)[lbls.index(lbl)]) for lbl in lbls if "pk" in lbl]
    cov = np.array(cov)
    pk = cosmo['pk_nobao'](kc) * (1. + fbao(kc))
    
    # Plot errorbars
    yup, ydn = baofisher.fix_log_plot(pk, cov)
    axes[k].errorbar( kc, fbao(kc), yerr=[ydn, yup], color=colours[k], ls='none', 
                      lw=1.8, capthick=1.8, label=names[k], ms='.' )

    # Plot f_bao(k)
    kk = np.logspace(-3., 1., 2000)
    axes[k].plot(kk, fbao(kk), 'k-', lw=1.8, alpha=0.6)
    
    # Set limits
    axes[k].set_xscale('log')
    axes[k].set_xlim((4e-3, 1e0))
    axes[k].set_ylim((-0.13, 0.13))
    
    axes[k].text( 0.38, 0.09, labels[k], fontsize=14, 
                  bbox={'facecolor':'white', 'alpha':1., 'edgecolor':'white',
                        'pad':15.} )


# Move subplots
# pos = [[x0, y0], [x1, y1]]
pos = [ax.get_position().get_points() for ax in axes]
dy = pos[0][0,1] - pos[1][1,1]
l = pos[0][0,0]
w = pos[0][1,0] - pos[0][0,0]
h = pos[0][1,1] - pos[0][0,1]
b = pos[0][0,1]

axes[0].set_position([l, b - 0.5*dy, w, h+0.5*dy])
axes[1].set_position([l, b - h - dy, w, h+0.5*dy])
axes[2].set_position([l, b - 2.*h - 1.5*dy, w, h+0.5*dy])


# Resize labels/ticks
fontsize = 18
for i in range(len(axes)):
    ax = axes[i]
    for tick in ax.xaxis.get_major_ticks():
      tick.label1.set_fontsize(fontsize)
      if i == 0 or i == 1: tick.label1.set_visible(False) # Hide x lbls in upper subplots
    for tick in ax.yaxis.get_major_ticks():
      tick.label1.set_fontsize(fontsize)

axes[-1].set_xlabel(r"$k \,[\mathrm{Mpc}^{-1}]$", fontdict={'fontsize':'20'})
#ax.set_ylabel(r"$P(k)$", fontdict={'fontsize':'20'})

#P.tight_layout()
# Set size
P.gcf().set_size_inches(8.5,10.)
P.savefig('mario-pub-fbao.png', dpi=200) #100

P.show()
