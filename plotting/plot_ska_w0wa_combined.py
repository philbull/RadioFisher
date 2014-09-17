#!/usr/bin/python
"""
Plot 2D constraints on (w0, wa).
"""
import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
import matplotlib.ticker
from units import *
from mpi4py import MPI

import os, copy
import euclid

fig_name = "ska-w0wa-combined.pdf"

USE_DETF_PLANCK_PRIOR = True
MARGINALISE_CURVATURE = True # Marginalise over Omega_K
MARGINALISE_INITIAL_PK = True # Marginalise over n_s, sigma_8
MARGINALISE_OMEGAB = True # Marginalise over Omega_baryons

cosmo = rf.experiments.cosmo

colours = [ ['#FFB928', '#FFEA28'],
            ['#1619A1', '#B1C9FD'], 
            ['#6B6B6B', '#BDBDBD'],
            ['#CC0000', '#F09B9B'], 
            ['#990A9C', '#F4BAF5'], 
            ['#FFB928', '#FFEA28'],
            ['#5B9C0A', '#BAE484'],
            ['#6B6B6B', '#BDBDBD'] ]

names = [ 'gSKASURASKAP_baoonly', 'SKA1MIDfull1_baoonly', 'EuclidRef_baoonly', 'gSKA2_baoonly', ]
labels = ['SKA1-SUR  (gal.)', 'SKA1-MID B1 (IM)', 'Euclid (gal.)', 'Full SKA (gal.)',]


################################################################################
# Load BOSS
################################################################################

root = "../output/" + 'BOSS_baoonly'
zc = np.atleast_2d( np.genfromtxt(root+"-cosmofns-zc.dat") ).T[0]
F_list = [np.genfromtxt(root+"-fisher-full-%d.dat" % i) for i in range(zc.size)]

pnames = rf.load_param_names(root+"-fisher-full-0.dat")
zfns = ['b_HI',]
excl = ['Tb', 'f', 'aperp', 'apar', 'DA', 'H', 'gamma', 'N_eff', 'pk*', 'fs8', 'bs8']
Fboss, lbl_boss = rf.combined_fisher_matrix( F_list,
                                                    expand=zfns, names=pnames,
                                                    exclude=excl )
# Relabel galaxy bias
for i in range(len(lbl_boss)):
    if "b_HI" in lbl_boss[i]: lbl_boss[i] = "gal%s" % lbl_boss[i]

################################################################################

# Fiducial value and plotting
fig = P.figure()
ax = fig.add_subplot(111)

_k = range(len(names)) #[::-1]
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
    pnames = rf.load_param_names(root+"-fisher-full-0.dat")
    zfns = ['b_HI', ]
    #excl = ['Tb', 'f', 'aperp', 'apar', 'DA', 'H', 'N_eff', 'pk*', 'fs8', 'bs8']
    excl = ['Tb', 'f', 'aperp', 'apar', 'DA', 'H', 'gamma', 'N_eff', 'pk*', 'fs8', 'bs8']
    F, lbls = rf.combined_fisher_matrix( F_list,
                                                expand=zfns, names=pnames,
                                                exclude=excl )
    
    # Combine with BOSS
    F, lbls = rf.add_fisher_matrices(F, Fboss, lbls, lbl_boss, expand=True)
    
    # DETF Planck prior
    print "*** Using DETF Planck prior ***"
    l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h', 'sigma8']
    F_detf = euclid.detf_to_rf("DETF_PLANCK_FISHER.txt", cosmo, omegab=False)
    Fpl, lbls = rf.add_fisher_matrices(F, F_detf, lbls, l2, expand=True)
    
    # Decide whether to fix various parameters
    fixed_params = []
    if not MARGINALISE_CURVATURE: fixed_params += ['omegak',]
    if not MARGINALISE_INITIAL_PK: fixed_params += ['n_s', 'sigma8']
    if not MARGINALISE_OMEGAB: fixed_params += ['omega_b',]
    
    if len(fixed_params) > 0:
        Fpl, lbls = rf.combined_fisher_matrix( [Fpl,], expand=[], 
                     names=lbls, exclude=fixed_params )
    
    # Really hopeful H0 prior
    #ph = lbls.index('h')
    #Fpl[ph, ph] += 1./(0.012)**2.
    
    # Get indices of w0, wa
    pw0 = lbls.index('w0'); pwa = lbls.index('wa')
    
    print "-"*50
    print names[k]
    print "-"*50
    
    # Invert matrix
    cov_pl = np.linalg.inv(Fpl)
    
    # Print 1D marginals
    print "1D sigma(w_0) = %3.4f" % np.sqrt(cov_pl[pw0,pw0])
    print "1D sigma(gamma) = %3.4f" % np.sqrt(cov_pl[pwa,pwa])
    print lbls
    
    x = rf.experiments.cosmo['w0']
    y = rf.experiments.cosmo['wa']
    print "FOM:", rf.figure_of_merit(pw0, pwa, Fpl, cov=cov_pl)
    
    # Plot contours for gamma, w0
    transp = [1., 0.85]
    w, h, ang, alpha = rf.ellipse_for_fisher_params(pw0, pwa, None, Finv=cov_pl)
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                height=alpha[kk]*h, angle=ang, fc=colours[k][kk], 
                ec=colours[k][0], lw=1.5, alpha=transp[kk]) for kk in [1,0]]
    for e in ellipses: ax.add_patch(e)
    
# Centroid
ax.plot(x, y, 'ko')


# Report on what options were used
print "-"*50
s1 = "Marginalised over Omega_K" if MARGINALISE_CURVATURE else "Fixed Omega_K"
s2 = "Marginalised over ns, sigma8" if MARGINALISE_INITIAL_PK else "Fixed ns, sigma8"
s3 = "Marginalised over Omega_b" if MARGINALISE_OMEGAB else "Fixed Omega_b"
print "NOTE:", s1
print "NOTE:", s2
print "NOTE:", s3


labels = [labels[k] for k in [0,1,2,3,]]
lines = [ matplotlib.lines.Line2D([0.,], [0.,], lw=8.5, color=colours[k][0], alpha=0.65) for k in [0,1,2,3,]]

P.gcf().legend((l for l in lines), (name for name in labels), prop={'size':'large'}, bbox_to_anchor=[0.96, 0.95], frameon=False)

P.figtext(0.18, 0.23, "(incl. BOSS + Planck)", fontsize=16)

ax.tick_params(axis='both', which='major', labelsize=20, size=8., width=1.5, pad=15.)
xminorLocator = matplotlib.ticker.MultipleLocator(0.1)
yminorLocator = matplotlib.ticker.MultipleLocator(0.5)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)

ax.set_xlabel(r"$w_0$", fontdict={'fontsize':'xx-large'}, labelpad=15.)
ax.set_ylabel(r"$w_a$", fontdict={'fontsize':'xx-large'})

ax.set_xlim((-1.51, -0.49))
ax.set_ylim((-1.3, 1.3))

#ax.set_xlim((-1.31, -0.69))
#ax.set_ylim((-0.8, 0.8))

# Set size and save
P.tight_layout()
P.gcf().set_size_inches(10.,7.)
P.savefig(fig_name, transparent=True)
P.show()
