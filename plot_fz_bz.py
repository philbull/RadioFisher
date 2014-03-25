#!/usr/bin/python
"""
Plot 2D constraints on f(z) and b_HI(z).
"""
import numpy as np
import pylab as P
import baofisher
import matplotlib.patches
import matplotlib.cm
import matplotlib.ticker
from units import *
from mpi4py import MPI
import experiments
import os
import euclid

USE_DETF_PLANCK_PRIOR = True
MARGINALISE_CURVATURE = True # Marginalise over Omega_K
MARGINALISE_INITIAL_PK = True # Marginalise over n_s, sigma_8
MARGINALISE_OMEGAB = True # Marginalise over Omega_baryons

cosmo = experiments.cosmo
#names = ['EuclidRef', 'cexptL', 'iexptM']
#labels = ['DETF IV', 'Facility', 'Mature']

names = ['cexptL',]
labels = ['Facility',]

colours = [ ['#CC0000', '#F09B9B'],
            ['#1619A1', '#B1C9FD'],
            ['#5B9C0A', '#BAE484'],
            ['#FFB928', '#FFEA28'] ]

"""
names = ['cexptL_bao', 'cexptL_bao_rsd', 'cexptL_bao_pkshift', 'cexptL_bao_vol', 'cexptL_bao_allap', 'cexptL_bao_all']
labels = ['BAO only', 'BAO + RSD', 'BAO + P(k) shift', 'BAO + Volume', 'BAO + AP', 'All']
colours = [ ['#CC0000', '#F09B9B'],
            ['#1619A1', '#B1C9FD'],
            ['#5B9C0A', '#BAE484'],
            ['c', '#FFEA28'],
            ['m', '#F09B9B'],
            ['k', '#B1C9FD'] ]
"""

# Fiducial value and plotting
fig = P.figure()
ax = fig.add_subplot(111)

_k = range(len(names))[::-1]
for k in _k:
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
    pnames = baofisher.load_param_names(root+"-fisher-full-0.dat")
    zfns = ['A', 'bs8', 'fs8', 'DA', 'H', 'aperp', 'apar']
    excl = ['Tb', 'sigma8', 'n_s', 'omegak', 'omegaDE', 'w0', 'wa', 'h', 
            'gamma', 'N_eff', 'pk*', 'f', 'b_HI']
    F, lbls = baofisher.combined_fisher_matrix( F_list,
                                                expand=zfns, names=pnames,
                                                exclude=excl )
    # Get indices of f, b_HI
    pf = baofisher.indices_for_param_names(lbls, 'fs8*')
    pb = baofisher.indices_for_param_names(lbls, 'bs8*')
    
    print "-"*50
    print names[k]
    print "-"*50
    
    # Invert matrix
    cov_pl = np.linalg.inv(F)
    
    for jj in range(pf.size):
        #if jj % 2 == 0: continue
        
        print jj, lbls[pb[jj]], lbls[pf[jj]]
        x = baofisher.bias_HI(zc[jj], cosmo) * cosmo['sigma_8']
        y = fc[jj] * cosmo['sigma_8']
        
        # Plot contours for w0, wa; omega_k free
        w, h, ang, alpha = baofisher.ellipse_for_fisher_params(pb[jj], pf[jj], None, Finv=cov_pl)
        ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                    height=alpha[kk]*h, angle=ang, fc='none', #colours[k][kk], 
                    ec=colours[k][0], lw=1.7, alpha=1.) for kk in [0,]]
        for e in ellipses: ax.add_patch(e)
    
        # Centroid
        ax.plot(x, y, 'kx', markersize=4, markeredgewidth=1.2)
        
        if jj == 0:
            ax.annotate( "z = %3.2f"%zc[jj], xy=(x, y), xytext=(10., -25.), 
               fontsize='large', textcoords='offset points', ha='center', va='center' )
        if jj == 11:
            ax.annotate( "z = %3.2f"%zc[jj], xy=(x, y), xytext=(10., -80.), 
               fontsize='large', textcoords='offset points', ha='center', va='center' )


# Axis ticks and labels
ax.tick_params(axis='both', which='major', labelsize=20, size=8., width=1.5, pad=8.)
ax.set_xlabel(r"$b_\mathrm{HI}\sigma_8(z)$", fontdict={'fontsize':'xx-large'}, labelpad=15.)
ax.set_ylabel(r"$f\sigma_8(z)$", fontdict={'fontsize':'xx-large'}, labelpad=15.)
ax.set_xlim((0.56, 1.62))
ax.set_ylim((0.49, 0.95))
ymajorLocator = matplotlib.ticker.MultipleLocator(0.1)
ax.yaxis.set_major_locator(ymajorLocator)


# Set size and save
P.tight_layout()
P.gcf().set_size_inches(8.,6.)
P.savefig("pub-fz-bz.pdf", transparent=True)
P.show()
