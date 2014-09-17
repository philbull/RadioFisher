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
ax1 = fig.add_subplot(111)

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
    
    """
    FF = np.abs(F_list[8] - F_list[7])
    FF = rf.fisher_with_excluded_params(FF, [i for i in range(16, 16+kc.size)])
    
    rf.plot_corrmat(FF, pnames[:16])
    exit()
    """
    
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
    err_f2 = np.sqrt( np.diag(cov2)[pf2] )
    err_H = np.sqrt( cov[pH, pH] ) * 100.
    err_H2 = np.sqrt( cov2[pH2, pH2] ) * 100.
    err_DA = np.sqrt( cov[pDA, pDA] ) * 1e3
    err_DA2 = np.sqrt( cov2[pDA2, pDA2] ) * 1e3
    
    ax1.errorbar(zc, fc, yerr=err_f, ls='none', color=cols[k], marker='.', 
                 label=names[k] + " (binned bias)", lw=2.5, capthick=2.)
    ax1.errorbar(zc+0.005, fc, yerr=err_f2, ls='none', color='c', marker='.', 
                 label=names[k] + " (constrained bias)", lw=2.5, capthick=2.)
    
# Euclid f(z) errorbars
z_euclid, f_euclid, err_f_euclid = euclid.sigma_f
ax1.errorbar(z_euclid, f(z_euclid), yerr=err_f_euclid, ls='none', color='b', marker='.', lw=2.5, capthick=2., label="Euclid (Amendola '12)")


# Fiducial functions
z = np.linspace(0.4, 1.52, 400)
ax1.plot(z, f(z), 'k-', lw=2., alpha=0.4)


ax1.set_xlim((0.4, 1.52))
ax1.set_ylim((0.68, 0.96))
ax1.legend(loc='upper left', prop={'size':'x-large'})


fontsize = 18
for tick in ax1.xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in ax1.yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)

ax1.set_xlabel(r"$z$", fontdict={'fontsize':'20'})
ax1.set_ylabel(r"$f(z)$", fontdict={'fontsize':'20'})

# Set size and save
P.gcf().set_size_inches(16.5,10.5)
P.savefig('mario-fz.png', dpi=100)

P.show()
