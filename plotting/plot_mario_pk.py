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
names = ["SKAMID", "iSKAMID"] #"iSKAMID_COMP_BIGZ", "iSKA_CORE"]

#ls = ['k-', 'r-', 'b--', 'm-', 'c--']
cols = ['b', 'r']

colours = ['#22AD1A', '#3399FF', '#ED7624']



# Get f_bao(k) function
cosmo_fns, cosmo = rf.precompute_for_fisher(rf.experiments.cosmo, "camb/rf_matterpower.dat")

fbao = cosmo['fbao']

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
    
    # Save P(k) rebinning info
    #np.savetxt(root+"-rebin-Fbase-%d.dat" % i, np.array(binning_info['F_base']) )
    #np.savetxt(root+"-rebin-cumul-%d.dat" % i, np.array(binning_info['cumul']) )
    #np.savetxt(root+"-rebin-kgrid-%d.dat" % i, np.array(binning_info['kgrid']) )
    #np.savetxt(root+"-rebin-Vfac-%d.dat" % i, np.array([binning_info['Vfac'],]) )
    
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
    pk = cosmo['pk_nobao'](kc) * (1. + fbao(kc))
    
    """
    # Remove elements with zero diagonal (completely unconstrained)
    zero_idxs = np.where(np.diag(F) == 0.)[0]
    print "Zero idxs:", zero_idxs
    F = rf.fisher_with_excluded_params(F, excl=zero_idxs)
    lbls = lbls[:-zero_idxs.size]
    """
    
    yup, ydn = rf.fix_log_plot(pk, cov) # cov*pk
    if names[k][0] is not 'i':
        #ax1.errorbar(kc, pk, yerr=[ydn, yup], ls=ls[k], lw=1.5, label=names[k], ms='.')
        ax1.errorbar(kc, fbao(kc), yerr=[ydn, yup], color=cols[k], ls='none', 
                     lw=2.2, capthick=2.2, label=names[k], ms='.')
    else:
        #ax2.errorbar(kc, pk, yerr=[ydn, yup], ls=ls[k], lw=1.5, label=names[k], ms='.')
        ax2.errorbar(kc, fbao(kc), yerr=[ydn, yup], color=cols[k], ls='none', 
                     lw=2.2, capthick=2.2, label=names[k], ms='.')
    
    
    #P.plot(kc, cov, ls[k])
    
    # Print diags.
    #for i in range(diags.size):
    #    #if i < diags2.size:
    #    #    print "%2d   %10s   %3.4f   %3.4f" % (i, lbls[i], diags[i], diags2[i])
    #    #else:
    #    print "%2d   %10s   %3.4f" % (i, lbls[i], diags[i])
    #exit()


kk = np.logspace(-3., 1., 2000)
ax1.plot(kk, fbao(kk), 'k-', lw=2.5, alpha=0.6)
ax2.plot(kk, fbao(kk), 'k-', lw=2.5, alpha=0.6)
#ax1.plot(kk, cosmo['pk_nobao'](kk) * (1. + fbao(kk)), 'k-')
#ax2.plot(kk, cosmo['pk_nobao'](kk) * (1. + fbao(kk)), 'k-')

ax1.set_xscale('log')
#ax1.set_yscale('log')
ax1.set_xlim((4e-3, 1e0))
#ax1.set_ylim((1e1, 1e6))
ax1.set_ylim((-0.11, 0.11))

ax2.set_xscale('log')
#ax2.set_yscale('log')
ax2.set_xlim((4e-3, 1e0))
ax2.set_ylim((-0.11, 0.11))
#ax2.set_ylim((1e1, 1e6))


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

ax2.set_xlabel(r"$k \,[\mathrm{Mpc}^{-1}]$", fontdict={'fontsize':'20'})
#ax.set_ylabel(r"$P(k)$", fontdict={'fontsize':'20'})

# Set size
P.gcf().set_size_inches(16.5,10.5)
P.savefig('mario-pk.png', dpi=100)

P.show()
