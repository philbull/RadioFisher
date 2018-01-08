#!/usr/bin/python
"""
Plot functions of redshift for RSDs.
"""
import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
import os

cosmo = rf.experiments.cosmo

fname = 'spherex-Hz.pdf'

names = [ 'gSPHEREx1_mgphotoz', 'gSPHEREx2_mgphotoz', 'BOSS_mg', 'EuclidRef_mg' ]
labels = [ 'SPHEREx 0.003', 'SPHEREx 0.008', 'BOSS spectro-z', 'Euclid spectro-z',]
colours = [ '#80B6D6', '#93C993', '#c8c8c8', '#757575', '#a8a8a8']
linestyle = [[], [], [], [], [], [], [], [], [], [], [],]
marker = ['o', 'o', 'o', 'D', 'D', 'D']
ms = [8., 8., 8., 7., 7., 7.]

# Fiducial value and plotting
P.subplot(111)
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
    pnames = rf.load_param_names(root+"-fisher-full-0.dat")
    
    #zfns = ['A', 'b_HI', 'f', 'H', 'DA', 'aperp', 'apar']
    zfns = ['A', 'bs8', 'fs8', 'H', 'DA', 'aperp', 'apar']
    excl = ['Tb', 'n_s', 'sigma8', 'omegak', 'omegaDE', 'w0', 'wa', 'h',
            'gamma', 'N_eff', 'pk*', 'f', 'b_HI', 
            'gamma0', 'gamma1', 'eta0', 'eta1', 'A_xi', 'logkmg',
            'sigma8tot', 'sigma_8', 'k*']
    
    # Marginalising over b_1
    F, lbls = rf.combined_fisher_matrix( F_list,
                                         expand=zfns, names=pnames,
                                         exclude=excl )
    print lbls
    cov = np.linalg.inv(F)
    errs = np.sqrt(np.diag(cov))
    
    # Identify functions of z
    #errDA = 1e3 * errs[pDA] / dAc
    pH = rf.indices_for_param_names(lbls, 'H*')
    errH = 1e2 * errs[pH] / Hc
    
    # Plot errors as fn. of redshift
    if labels[k] is not None:
        P.plot( zc, errH, color=colours[k], label=labels[k], lw=2.8,
                marker=marker[k], markersize=ms[k], markeredgecolor=colours[k] )
    else:
        P.plot( zc, errH, color=colours[k], label=labels[k], lw=2.8,
                marker=marker[k], markersize=ms[k], markeredgecolor=colours[k],
                dashes=[4,3] )

P.tick_params(axis='both', which='major', labelsize=20, width=1.5, size=8., pad=10)
P.tick_params(axis='both', which='minor', labelsize=20, width=1.5, size=5.)

# Set axis limits
P.xlim((-0.001, 2.1))
P.ylim((2e-3, 3e-1))

#          fontdict={'size':'x-large'}) #, bbox=dict(ec='k', fc='none', lw=1.2))
P.xlabel('$z$', labelpad=7., fontdict={'fontsize':'xx-large'})
P.ylabel('$\sigma_H / H$', labelpad=10., fontdict={'fontsize':'xx-large'})

# Set tick locations
#P.gca().yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
#P.gca().yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.01))
P.gca().xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.5))
P.yscale('log')
    
leg = P.legend(prop={'size':17}, loc='upper center', frameon=True, ncol=2)
leg.get_frame().set_edgecolor('w')
leg.get_frame().set_alpha(0.1)

# Set size
P.tight_layout()
#P.gcf().set_size_inches(8.4, 7.8)
#P.gcf().set_size_inches(9.5, 6.8)
P.savefig(fname, transparent=True)
P.show()
