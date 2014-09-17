#!/usr/bin/python
"""
Plot 1D constraints on Mnu
"""
import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
import matplotlib.ticker
from units import *
from mpi4py import MPI

import os
import euclid

cosmo = rf.experiments.cosmo

#fig_name = "pub-mnu-sigma8.pdf"
fig_name = "pub-mnu.pdf"

USE_DETF_PLANCK_PRIOR = True

names = [ 'cexptLmnu005', 'iexptMmnu005', 'exptSmnu005',
          'cexptLmnu010', 'iexptMmnu010', 'exptSmnu010',
          'cexptLmnu020', 'iexptMmnu020', 'exptSmnu020' ]
fid = [0.05, 0.05, 0.05,  0.1, 0.1, 0.1,  0.2, 0.2, 0.2]
labels = ['Facility', 'Mature', 'Snapshot'] #, 'Snapshot']
colours = ['#1619A1', '#5B9C0A', '#990A9C', '#FFB928', '#CC0000']

#colours = ['#BAE484', '#5B9C0A',   '#B1C9FD', '#1619A1',   '#F6ADAD', '#CC0000',
#           '#FFB928', '#FFEA28']

# Fiducial value and plotting
fig = P.figure()
ax = fig.add_subplot(111)

Nexpt = len(names)
m = 0
_k = range(len(names))
for k in _k:
    root = "../output/" + names[k]

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
    
    zfns = [1,]
    excl = [2,  6,7,8,  14]
    #excl = [2, 4,5, 9,10,11,12,13,14]
    excl  += [i for i in range(len(pnames)) if "pk" in pnames[i]]
    
    F, lbls = rf.combined_fisher_matrix( F_list,
                                                expand=zfns, names=pnames,
                                                exclude=excl )
    # Add Planck prior
    if USE_DETF_PLANCK_PRIOR:
        # DETF Planck prior
        print "*** Using DETF Planck prior ***"
        l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h', 'sigma8']
        F_detf = euclid.detf_to_rf("DETF_PLANCK_FISHER.txt", cosmo)
        Fpl, lbls = rf.add_fisher_matrices(F, F_detf, lbls, l2, expand=True)
    else:
        # Euclid Planck prior
        print "*** Using Euclid (Mukherjee) Planck prior ***"
        l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h']
        Fe = euclid.planck_prior_full
        F_eucl = euclid.euclid_to_rf(Fe, cosmo)
        Fpl, lbls = rf.add_fisher_matrices(F, F_eucl, lbls, l2, expand=True)
    
    # Decide whether to fix various parameters
    fixed_params = [] #['sigma8'] #['w0', 'wa'] #['omegak']
    if len(fixed_params) > 0:
        print "REMOVING:", fixed_params
        Fpl, lbls = rf.combined_fisher_matrix( [Fpl,], expand=[], 
                     names=lbls, exclude=[lbls.index(p) for p in fixed_params] )
    
    # Invert matrix
    cov_pl = np.linalg.inv(Fpl)
    
    # Plot errorbars and annotate
    p1 = lbls.index('Mnu')
    y = fid[k]
    err = np.sqrt(np.diag(cov_pl))[p1]
    print "%10s -- %s: %4.2e" % (lbls[p1], names[k], err), "\n"
    
    # Define label
    l = None
    if k < 3: l = labels[k]
    
    ax.errorbar( 0.5*m+0.1*k, y, yerr=err, color=colours[k%3], lw=2., 
                 marker='.', markersize=10., markeredgewidth=2., label=l )
    #ax.annotate( labels[k], xy=(, m+1.0), xytext=(0., -10.), 
    #             fontsize='large', textcoords='offset points', ha='center', va='center' )
    
    if k % 3 == 2: m += 1

fontsize = 16
for tick in ax.xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in ax.yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)


labels = [labels[k] for k in range(len(labels))]
lines = [ matplotlib.lines.Line2D([0.,], [0.,], lw=2.5, color=colours[k], alpha=0.95) for k in range(len(labels))]
P.legend((l for l in lines), (name for name in labels), loc='lower right', prop={'size':'x-large'})

"""
xminorLocator = matplotlib.ticker.MultipleLocator(0.1)
yminorLocator = matplotlib.ticker.MultipleLocator(0.5)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)
"""

#ax.set_xlabel(label1, fontdict={'fontsize':'xx-large'}, labelpad=15.)
ax.set_ylabel(r"$\Sigma m_\nu$", fontdict={'fontsize':'xx-large'}, labelpad=10.)

ax.axhline(0., ls='dashed', lw=1.5, color='k')
ax.set_xlim((-0.5, 2.3))
ax.set_ylim((-0.25, 0.4))

ax.tick_params(axis='both', which='both', labelbottom='off', top='off', bottom='off', length=6., width=1.5)

# Set size and save
P.tight_layout()
P.gcf().set_size_inches(8.,6.)
P.savefig(fig_name, transparent=True)
P.show()
