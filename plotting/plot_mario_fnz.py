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
names = ["SKAMID",] # "SKA1"] # , "iSKAMID_COMP"]

#ls = ['k-', 'r-', 'b--', 'm-', 'c--']
cols = ['r', 'g', 'c']

colours = ['#22AD1A', '#3399FF', '#ED7624']

cosmo_fns, cosmo = rf.precompute_for_fisher(rf.experiments.cosmo, "camb/rf_matterpower.dat")
H, r, D, f = cosmo_fns


# Fiducial value and plotting
fig = P.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

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
             'omegak', 'omegaDE', 'w0', 'wa', 'h', 'gamma', 'fNL']
    pnames += ["pk%d" % i for i in range(kc.size)]
    zfns = [1,  6,7,8]
    excl = [2,4,5,   9,10,11,12,13,14,15] #6,7,8 ]
    excl += [i for i in range(len(pnames)) if "pk" in pnames[i]]
    
    F, lbls = rf.combined_fisher_matrix( F_list,
                                                expand=zfns, names=pnames,
                                                exclude=excl )
    # Fixed bias evol.
    zfns = [6,7,8]
    excl = [2,4,5,   9,10,11,12,13,14,15] #6,7,8 ]
    excl += [i for i in range(len(pnames)) if "pk" in pnames[i]]
    
    F2, lbls2 = rf.combined_fisher_matrix( F_list,
                                                expand=zfns, names=pnames,
                                                exclude=excl )
    # Invert matrices
    cov = np.linalg.inv(F)
    cov2 = np.linalg.inv(F2)
    
    # Indices of fns. of z
    zfns = [1, 3,4,5] # Update after other params subtracted
    pf  = rf.indexes_for_sampled_fns(3, zc.size, zfns)
    pDA = rf.indexes_for_sampled_fns(4, zc.size, zfns)
    pH  = rf.indexes_for_sampled_fns(5, zc.size, zfns)
    
    # Indices of fns. of z (bias fixed in z)
    zfns = [3,4,5] # Update after other params subtracted
    pf2  = rf.indexes_for_sampled_fns(3, zc.size, zfns)
    pDA2 = rf.indexes_for_sampled_fns(4, zc.size, zfns)
    pH2  = rf.indexes_for_sampled_fns(5, zc.size, zfns)
    
    # SKA marginal errorbars
    err_f = np.sqrt( np.diag(cov)[pf] )
    err_H = np.sqrt( np.diag(cov)[pH] ) * 100.
    err_H2 = np.sqrt( np.diag(cov2)[pH2] ) * 100.
    err_DA = np.sqrt( np.diag(cov)[pDA] ) * 1e3
    err_DA2 = np.sqrt( np.diag(cov2)[pDA2] ) * 1e3
    
    print "-"*50
    for i in range(len(lbls)):
        print "%2d   %10s  %4.4f" % (i, lbls[i], np.sqrt(np.diag(cov)[i]))
    print "-"*50
    for i in range(len(lbls2)):
        print "%2d   %10s  %4.4f" % (i, lbls2[i], np.sqrt(np.diag(cov2)[i]))
    print "-"*50
    
    print err_DA
    print err_H
    
    ax1.errorbar(zc, Hc, yerr=err_H, ls='none', color=cols[k], marker='.', 
                 label=names[k] + " (BAO only)", lw=2.5, capthick=2.)
    #ax1.errorbar(zc+0.01, Hc, yerr=err_H2, ls='none', color='c', marker='.', lw=2.5, capthick=2.)
    
    ax2.errorbar(zc, dAc, yerr=err_DA, ls='none', color=cols[k], marker='.', 
                 lw=2.5, capthick=2.)
    #ax2.errorbar(zc+0.01, dAc, yerr=err_DA2, ls='none', color='c', marker='.', lw=2.5, capthick=2.)
    
    #err_f = np.sqrt( cov[pf, pf] )
    #err_f2 = np.sqrt( cov2[pf2, pf2] )
    #ax1.errorbar(zc, fc, yerr=err_f, ls='none', color=cols[k], lw=1.5, marker='.')
    #P.errorbar(zc, fc, yerr=err_f2, ls='none', color='g', lw=1.5, marker='.')

    
# Euclid f(z) errorbars
#z_euclid, f_euclid, err_f_euclid = euclid.sigma_f
#P.errorbar(z_euclid, f(z_euclid), yerr=err_f_euclid, ls='none', color='b', lw=1.5, marker='.')


# Fiducial functions
z = np.linspace(0.4, 1.5, 400)
ax1.plot(z, H(z), 'k-', lw=2., alpha=0.3)
ax2.plot(z, r(z) / (1. + z), 'k-', lw=2., alpha=0.3)

# Values from Lazkoz et al. 2013 (BAO cosmography)
ze, y, dy, yp, dyp = euclid.bao_scales
DAe = r(ze)/(1.+ze)
sig_H = H(ze) * dyp / yp
sig_DA = DAe * dy / y

# Values from Amendola et al. Euclid review, Fig. 1.21 (non-parametric weak lensing)
_z = [0.3, 0.65, 0.875, 1.15, 2.15]
_dH = np.array([0.23, 0.072, 0.089, 0.064, 0.76])

#ax1.errorbar(_z, H(_z), yerr=_dH*70., color='c', label="Euclid WL (Amendola '12)", ls='none', lw=2.)
##ax1.errorbar(ze, H(ze), yerr=sig_H, color='b', label="Euclid-like BAO (Lazkoz '13)", ls='none', lw=2.5, capthick=2.)

##ax2.errorbar(ze, DAe, yerr=sig_DA, color='b', label="Euclid-like BAO (Lazkoz '13)", ls='none', lw=2.5, capthick=2.)

ax1.set_xlim((0.4, 1.5))
ax2.set_xlim((0.4, 1.5))

ax1.set_ylim((76., 162.))
#ax2.set_ylim((1080., 1990.))

ax2.set_ylim((0., 2200.))

ax1.legend(loc='upper left', prop={'size':'x-large'})

# Move subplots
# pos = [[x0, y0], [x1, y1]]
pos1 = ax1.get_position().get_points()
pos2 = ax2.get_position().get_points()
dy = pos1[0,1] - pos2[1,1]
l = pos1[0,0]
w = pos1[1,0] - pos1[0,0]
h = pos1[1,1] - pos1[0,1]
b = pos1[0,1]

ax1.set_position([l, b - 0.5*dy, w, h+0.5*dy])
ax2.set_position([l, b - h - dy, w, h+0.5*dy])

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

ax2.set_xlabel(r"$z$", fontdict={'fontsize':'20'})
ax1.set_ylabel(r"$H(z)$", fontdict={'fontsize':'20'})
ax2.set_ylabel(r"$D_A(z)$", fontdict={'fontsize':'20'})

# Set size and save
P.gcf().set_size_inches(16.5,10.5)
P.savefig('mario-da-hz.png', dpi=100)

P.show()
