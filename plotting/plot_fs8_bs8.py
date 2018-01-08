#!/usr/bin/python
"""
Plot 2D constraints on f(z) and b_HI(z) as fn of redshift (Fig. 10).
"""
import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
import matplotlib.ticker
import os
from radiofisher import euclid

USE_DETF_PLANCK_PRIOR = True
MARGINALISE_CURVATURE = True # Marginalise over Omega_K
MARGINALISE_INITIAL_PK = True # Marginalise over n_s, sigma_8
MARGINALISE_OMEGAB = True # Marginalise over Omega_baryons

cosmo = rf.experiments.cosmo

names = ['exptL_paper',]
labels = ['Facility',]

colours = [ ['#CC0000', '#F09B9B'],
            ['#1619A1', '#B1C9FD'],
            ['#5B9C0A', '#BAE484'],
            ['#FFB928', '#FFEA28'] ]

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
    pnames = rf.load_param_names(root+"-fisher-full-0.dat")
    zfns = ['A', 'bs8', 'fs8', 'DA', 'H', 'aperp', 'apar']
    excl = ['Tb', 'sigma8', 'n_s', 'omegak', 'omegaDE', 'w0', 'wa', 'h', 
            'gamma', 'N_eff', 'pk*', 'f', 'b_HI']
    F, lbls = rf.combined_fisher_matrix( F_list, expand=zfns, names=pnames,
                                         exclude=excl )
    # Get indices of f, b_HI
    pf = rf.indices_for_param_names(lbls, 'fs8*')
    pb = rf.indices_for_param_names(lbls, 'bs8*')
    
    print "-"*50
    print names[k]
    print "-"*50
    
    # Invert matrix
    cov_pl = np.linalg.inv(F)
    
    ll = -1
    for jj in pf:
        ll += 1
        #print jj, lbls[jj], zc[ll], np.sqrt(np.diag(cov_pl))[jj]
        print "%4.3f %10.10f" % (zc[ll], np.sqrt(np.diag(cov_pl))[jj])
    
    for jj in range(pf.size):
        #if jj % 2 == 0: continue
        
        print jj, lbls[pb[jj]], lbls[pf[jj]]
        x = rf.bias_HI(zc[jj], cosmo) * cosmo['sigma_8']
        y = fc[jj] * cosmo['sigma_8']
        
        # Plot contours for w0, wa; omega_k free
        w, h, ang, alpha = rf.ellipse_for_fisher_params(pb[jj], pf[jj], 
                                                        None, Finv=cov_pl)
        transp = [1., 0.85]
        ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                    height=alpha[kk]*h, angle=ang, fc=colours[k][kk], 
                    ec=colours[k][0], lw=1.3, alpha=0.85) for kk in [1,0]]
        for e in ellipses: ax.add_patch(e)
    
        # Centroid
        ax.plot(x, y, 'kx', markersize=4, markeredgewidth=1.2)
        
        if jj == 0:
            ax.annotate( "z = %3.2f"%zc[jj], xy=(x, y), xytext=(10., -25.), 
               fontsize='large', textcoords='offset points', ha='center', va='center' )
        if jj == 11:
            ax.annotate( "z = %3.2f"%zc[jj], xy=(x, y), xytext=(10., -60.), 
               fontsize='large', textcoords='offset points', ha='center', va='center' )


exit()

# Axis ticks and labels
ax.tick_params(axis='both', which='major', labelsize=20, size=8., width=1.5, pad=8.)
ax.tick_params(axis='both', which='minor', labelsize=20, size=5., width=1.5, pad=8.)
ax.set_xlabel(r"$b_\mathrm{HI}\sigma_8(z)$", fontdict={'fontsize':'xx-large'}, labelpad=15.)
ax.set_ylabel(r"$f\sigma_8(z)$", fontdict={'fontsize':'xx-large'}, labelpad=15.)
ax.set_xlim((0.5, 1.3))
ax.set_ylim((0.49, 0.95))
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))

# Set size and save
P.tight_layout()
P.gcf().set_size_inches(8.,6.)
#P.savefig("fig10-fs8-bs8.pdf", transparent=True)
P.show()
