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

###names = ["cVcexptL", "cNEWexptL", "cNEW2exptL", "cNEW3exptL"] #"iexptM"] #, "exptS"]
names = ["ctestL", "itestL", "testL", "ctestM", "itestM", "testM"] #["cNEWexptL", "iexptM"] #, "exptS"]

#colours = ['#CC0000', '#ED5F21', '#FAE300', '#5B9C0A', '#1619A1', '#56129F', '#990A9C']
colours = ['#990A9C', '#CC0000', '#5B9C0A', '#1619A1']
linestyle = ['solid', 'solid', 'dashed', 'dashdot']
#labels = ['CV-limited Behemoth', 'Behemoth', 'Mature', 'Snapshot']
labels = ['CV-limited','SKAMREF2COMP', 'SKAMREF2', 'cNEW3exptL'] #'SKAMREF2(old)']

# Get f_bao(k) function
cosmo_fns, cosmo = baofisher.precompute_for_fisher(experiments.cosmo, "camb/baofisher_matterpower.dat")
fbao = cosmo['fbao']

# Fiducial value and plotting
#fig, (ax1, ax2, ax3) = P.subplots(1, 3)
#axes = [ax1, ax2, ax3]

ax = P.subplot(111)
#ax2 = P.subplot(122)
cmaps = [matplotlib.cm.autumn, matplotlib.cm.winter]
cols = ['r-', 'y--', 'm:', 'b-', 'g--', 'c:']

for k in range(len(names)):
    #ax = axes[k]
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
    
    mins = []

    for j in range(len(F_list)):
        F = F_list[j]
        print F.shape
        
        # Just do the simplest thing for P(k) and get 1/sqrt(F)
        cov = [np.sqrt(1. / np.diag(F)[pnames.index(lbl)]) for lbl in pnames if "pk" in lbl]
        cov = np.array(cov)
        pk = cosmo['pk_nobao'](kc) * (1. + fbao(kc))

        # Replace nan/inf values
        cov[np.where(np.isnan(cov))] = 1e10
        cov[np.where(np.isinf(cov))] = 1e10
        
        #cmap = cmaps[k]
        
        frac = float(j) / float(len(F_list))

        # Plot errorbars
        #if "exptS" in names[k]:
        #    ax.plot(kc, cov, color=cmap(frac), lw=2.4, ls='solid')
        #else:
        #    if j % 1 == 0: ax.plot(kc, cov, color=cmap(frac), lw=2.4, ls='solid', label="%d: %3.2f" % (k, zc[j]))
        
        """
        if (zc[j] > 0.8 and zc[j] < 0.9) or \
           (zc[j] > 1.5 and zc[j] < 1.65) or \
           (zc[j] > 2.0 and zc[j] < 2.15):
            #color=cmap(frac)
            ax.plot( kc, cov, color=cols[k], lw=2.4, ls='solid', 
                     label="%d: %3.2f" % (k, zc[j]) )
        mins.append(np.min(cov))
        """
        
        ax.plot( kc, cov, cols[k], lw=2.4, label=names[k]) #label="%d: %3.2f" % (k, zc[j]) )
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        #ax.set_xlim((2e-3, 3e0))
        #ax.set_ylim((9e-4, 1e1))
    
    # Output table of min. dP/P
    print "-"*50
    for j in range(len(mins)):
        print "%2d:  %3.3f  %3.3e" % (j, zc[j], mins[j])
    print "-"*50
    #ax2.plot(zc, mins, lw=1.8, marker='.')
    P.title(str(zc))
    
    P.ylim((1e-3, 1e10))

ax.legend(loc='lower left', prop={'size':'x-small'}, ncol=2)


"""
# Resize labels/ticks
fontsize = 18
ax = P.gca()
for tick in ax.xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in ax.yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)

P.xlabel(r"$k \,[\mathrm{Mpc}^{-1}]$", fontdict={'fontsize':'20'})
P.ylabel(r"$\Delta P / P$", fontdict={'fontsize':'20'})
"""

P.tight_layout()
# Set size
#P.gcf().set_size_inches(8.,6.)
#P.savefig('pub-dlogp.png', dpi=100)

P.show()
