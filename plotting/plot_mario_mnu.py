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
#names = ["SKAMID_mnu01",] #"iSKAMID_mnu01"]

names = ["SKA1", "SKAMID", "iSKAMID", "iSKAMID_COMP", "iSKAMID_COMP_BIGZ", "iSKAMID_BIGZ"]

#names = ["SKA1", "SKAMID", "iSKAMID", "iSKA_CORE"]


#ls = ['k-', 'r-', 'b--', 'm-', 'c--']
cols = ['r', 'g', 'c']

colours = ['#22AD1A', '#3399FF', '#ED7624', 'y', 'c', 'm', 'k', '#cc33cc']

cosmo_fns, cosmo = rf.precompute_for_fisher(rf.experiments.cosmo, "camb/rf_matterpower.dat")
H, r, D, f = cosmo_fns


# Fiducial value and plotting
fig = P.figure()
ax = fig.add_subplot(111)

for k in range(len(names)):
    root = "output/" + names[k]

    # Load cosmo fns.
    dat = np.atleast_2d( np.genfromtxt(root+"-cosmofns-zc.dat") ).T
    zc, Hc, dAc, Dc, fc = dat
    zs, Hs, dAs, Ds, fs = np.genfromtxt(root+"-cosmofns-smooth.dat").T
    kc = np.genfromtxt(root+"-fisher-kc.dat").T

    # Load Fisher matrices as fn. of z
    Nbins = zc.size
    F_list = [np.genfromtxt(root+"-fisher-full-%d.dat" % i) for i in range(Nbins)]
    
    # EOS FISHER MATRIX
    # Actually, (aperp, apar) are (D_A, H)
    pnames = ['A', 'b_HI', 'Tb', 'sigma_NL', 'sigma8', 'n_s', 'f', 'aperp', 'apar', 
             'omegak', 'omegaDE', 'w0', 'wa', 'h', 'gamma', 'Mnu']
    pnames += ["pk%d" % i for i in range(kc.size)]
    zfns = [] #[1,]
    excl = [2, 6,7,8,  4]
    excl += [i for i in range(len(pnames)) if "pk" in pnames[i]]
    
    F, lbls = rf.combined_fisher_matrix( F_list,
                                                expand=zfns, names=pnames,
                                                exclude=excl )
    
    # Add Planck prior
    Fpl = F.copy()
    # Indices of Planck prior Fisher matrix in our Fisher matrix
    # w0:6, wa:7, omega_DE:5, omega_k:4, w_m, w_b, n_s:3
    _names = ['w0', 'wa', 'omegaDE', 'omegak', 'w_m', 'w_b', 'n_s']
    idxs_in_f = [6, 7, 5, 4, -1, -1, 3]
    for i in range(len(idxs_in_f)):
        if idxs_in_f[i] == -1: continue
        for j in range(len(idxs_in_f)):
            if idxs_in_f[j] == -1: continue
            ii = rf.indexes_for_sampled_fns(idxs_in_f[i], zc.size, zfns)
            jj = rf.indexes_for_sampled_fns(idxs_in_f[j], zc.size, zfns)
            Fpl[ii,jj] += euclid.planck_prior_full[i,j]
            #print ">>>", _names[i], _names[j], lbls[idxs_in_f[i]], lbls[idxs_in_f[j]]
    
    cov = np.linalg.inv(F)
    cov_pl = np.linalg.inv(Fpl)
    
    print names[k]
    for i in range(len(lbls)):
        print "%2d  %10s  %3.4f  %3.4f" % (i, lbls[i], np.sqrt(cov[i,i]), np.sqrt(cov_pl[i,i]))
    print "-"*50
    
    # Indices of fns. of z
    pmnu = rf.indexes_for_sampled_fns(10, zc.size, zfns)
    pns  = rf.indexes_for_sampled_fns(3,  zc.size, zfns)
    
    x = 0.15 #rf.experiments.cosmo['mnu']
    y = rf.experiments.cosmo['n']
    
    """
    # Mnu, n_s
    w, h, ang, alpha = rf.ellipse_for_fisher_params(pmnu, pns, cov, Finv=cov)
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                 height=alpha[kk]*h, angle=ang, fc='none', ec=colours[k], 
                 lw=2.5, alpha=1.) for kk in range(0, 2)]
    for e in ellipses: ax.add_patch(e)
    """
    
    # Mnu, n_s + Planck
    w, h, ang, alpha = rf.ellipse_for_fisher_params(pmnu, pns, None, Finv=cov_pl)
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                 height=alpha[kk]*h, angle=ang, fc='none', ec=colours[k+1], 
                 lw=2.5, alpha=1., ls='solid') for kk in range(0, 2)]
    for e in ellipses: ax.add_patch(e)

"""
w, h, ang, alpha = rf.ellipse_for_fisher_params(0, 1, None, Finv=euclid.cov_mnu_ns_euclid_boss)
ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
             height=alpha[kk]*h, angle=ang, fc='r', ec='r', 
             lw=2.5, alpha=0.2) for kk in range(0, 2)]
for e in ellipses: ax.add_patch(e)
"""

w, h, ang, alpha = rf.ellipse_for_fisher_params(0, 1, None, Finv=euclid.cov_mnu_ns_euclid_boss_planck)
ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
             height=alpha[kk]*h, angle=ang, fc='r', ec='r', 
             lw=2.5, alpha=0.1) for kk in range(0, 2)]
for e in ellipses: ax.add_patch(e)

# Fiducial value
ax.plot(x, y, 'kx', markersize=14)


# Labels
lines = []; labels = []
for i in range(len(names)):
    lines.append( matplotlib.lines.Line2D([0.,], [0.,], lw=2.5, color=colours[i+1]) )
    labels.append( names[i] )


#labels = ["SKAMID + Planck (SD, )", "SKAMID + Planck (int)", "Euclid + Planck (no marg.)"]
#lines = []
#lines.append( matplotlib.lines.Line2D([0.,], [0.,], lw=2.5, color=colours[1]) )
#lines.append( matplotlib.lines.Line2D([0.,], [0.,], lw=2.5, color=colours[2]) )
#lines.append( matplotlib.lines.Line2D([0.,], [0.,], lw=2.5, color=colours[1]) )


lines.append( matplotlib.lines.Line2D([0.,], [0.,], lw=8.5, color='r', alpha=0.3) )
ax.legend((l for l in lines), (lbl for lbl in labels), prop={'size':'x-large'}, loc='upper left')


fontsize = 18
for tick in ax.xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in ax.yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)

ax.set_xlabel(r"$M_\nu \, [\mathrm{eV}]$", fontdict={'fontsize':'20'})
ax.set_ylabel(r"$n_s$", fontdict={'fontsize':'20'})

ax.axvline(0.15 - 0.031, color='k', ls='dotted', lw=1.5)
ax.axvline(0.15 + 0.031, color='k', ls='dotted', lw=1.5)

ax.set_xlim((0., 0.30))
ax.set_ylim((0.95, 0.9701))

# Set size and save
P.gcf().set_size_inches(16.5,10.5)
P.savefig('mario-mnu.png', dpi=100)

P.show()
