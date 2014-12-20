#!/usr/bin/python
"""
Plot 2D constraints on (w0, wa), with a low-z survey added in for good measure.
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

print "OBSOLETE"
exit()

fig_name = "ska-w0wa-inclSKA1gal.pdf"

USE_DETF_PLANCK_PRIOR = True
MARGINALISE_CURVATURE = True # Marginalise over Omega_K
MARGINALISE_INITIAL_PK = True # Marginalise over n_s, sigma_8
MARGINALISE_OMEGAB = True # Marginalise over Omega_baryons

cosmo = rf.experiments.cosmo

names = ['EuclidRef', 'SKA1SURfull1', 'cSKA1MIDfull1']
labels = ['Euclid', 'SKA1-SUR (B1)', 'SKA1-MID (B1)']

names = ['EuclidRef', 'cSKA1MIDfull2']
labels = ['Euclid', 'SKA1-MID (B2)']

#names = ["SKAHI73", "EuclidRef", 'SKA1SURfull2', 'SKA1SURfull1']
#labels = ['SKA2 HI gal.', 'Euclid', 'SKA1-SUR (B1)', 'SKA1-SUR (B2)']

names = ["SKAHI73", "EuclidRef", 'SKA1SURfull2', 'SKA1SURfull1', 'cSKA1MIDfull2', 'cSKA1MIDfull1']
labels = ['SKA2 HI gal.', 'Euclid', 'SKA1-SUR (B2)', 'SKA1-SUR (B1)', 'SKA1-MID (B2)', 'SKA1-MID (B1)']

colours = [ ['#5B9C0A', '#BAE484'], 
            ['#990A9C', '#F4BAF5'], 
            ['#1619A1', '#B1C9FD'], 
            ['#CC0000', '#F09B9B'], 
            ['#FFB928', '#FFEA28'],
            ['#6B6B6B', '#BDBDBD'],
            ['#5B9C0A', '#BAE484'],
            ['#6B6B6B', '#BDBDBD'] ]
#            ['#5B9C0A', '#BAE484'],
#            ['#FFB928', '#FFEA28'] ]




#-----------------------------------------------------------
# SKA BAO chapter: BAO-only (w0, wa), incl. BOSS and Planck
#-----------------------------------------------------------
names = ['gSKASURASKAP_baoonly', 'SKA1MIDfull2_baoonly', 'EuclidRef_baoonly', 
         'gSKA2_baoonly']
labels = ['SKA1-SUR (gal.)', 'SKA1-MID B2 (IM)', 'Euclid (gal.)', 'SKA 2 (gal.)']
fig_name = "ska-w0wa-gal.pdf"
legend_order = [0,1,3,2]
colours = [ ['#1619A1', '#B1C9FD'], 
            ['#FFA728', '#F8D24B'],
            ['#6B6B6B', '#BDBDBD'],
            ['#CC0000', '#F09B9B'], ]





################################################################################
# Load low-z galaxy survey Fisher matrix
root = "output/" + "BOSS_baoonly"

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
excl = ['Tb', 'f', 'aperp', 'apar', 'DA', 'H', 'N_eff', 'pk*']
F_lowz, lbls_lowz = rf.combined_fisher_matrix( F_list, expand=zfns, 
                                               names=pnames, exclude=excl )
# Relabel galaxy bias from low-z survey
for i in range(len(lbls_lowz)):
    if "b_HI" in lbls_lowz[i]: lbls_lowz[i] = "lowz%s" % lbls_lowz[i]

print lbls_lowz

################################################################################

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
    zfns = ['b_HI', ]
    excl = ['Tb', 'f', 'aperp', 'apar', 'DA', 'H', 'N_eff', 'pk*']
    F, lbls = rf.combined_fisher_matrix( F_list, expand=zfns, names=pnames,
                                         exclude=excl )
    
    # Relabel galaxy bias from low-z survey and sum current survey + low-z
    F, lbls = rf.add_fisher_matrices(F_lowz, F, lbls_lowz, lbls, expand=True)
    print lbls
    
    # Add Planck prior
    print "*** Using DETF Planck prior ***"
    l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h', 'sigma8']
    F_detf = euclid.detf_to_rf("DETF_PLANCK_FISHER.txt", cosmo, omegab=False)
    Fpl, lbls = rf.add_fisher_matrices(F, F_detf, lbls, l2, expand=True)
    
    # Add BOSS 
    Fpl, lbls = rf.add_fisher_matrices(F, F_detf, lbls, l2, expand=True)
    
    # Decide whether to fix various parameters
    fixed_params = []
    if not MARGINALISE_CURVATURE: fixed_params += ['omegak',]
    if not MARGINALISE_INITIAL_PK: fixed_params += ['n_s', 'sigma8']
    if not MARGINALISE_OMEGAB: fixed_params += ['omega_b',]
    
    if len(fixed_params) > 0:
        Fpl, lbls = rf.combined_fisher_matrix( [Fpl,], expand=[], 
                     names=lbls, exclude=fixed_params )
    
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
    
    x = rf.experiments.cosmo['w0']
    y = rf.experiments.cosmo['wa']
    
    # Plot contours for gamma, w0
    transp = [1., 0.85]
    w, h, ang, alpha = rf.ellipse_for_fisher_params(pw0, pwa, None, Finv=cov_pl)
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                height=alpha[kk]*h, angle=ang, fc=colours[k][kk], 
                ec=colours[k][0], lw=1.5, alpha=transp[kk]) for kk in [1,0]]
    for e in ellipses: ax.add_patch(e)
    
    # Centroid
    ax.plot(x, y, 'ko')

P.figtext(0.18, 0.22, "Combined w. Planck + SKA1 gal.", fontsize=15)

# Report on what options were used
print "-"*50
s1 = "Marginalised over Omega_K" if MARGINALISE_CURVATURE else "Fixed Omega_K"
s2 = "Marginalised over ns, sigma8" if MARGINALISE_INITIAL_PK else "Fixed ns, sigma8"
s3 = "Marginalised over Omega_b" if MARGINALISE_OMEGAB else "Fixed Omega_b"
print "NOTE:", s1
print "NOTE:", s2
print "NOTE:", s3

# Legend
labels = [labels[k] for k in range(len(labels))]
lines = [ matplotlib.lines.Line2D([0.,], [0.,], lw=8.5, color=colours[k][0], alpha=0.65) for k in range(len(labels))]
P.gcf().legend((l for l in lines), (name for name in labels), prop={'size':'large'}, bbox_to_anchor=[0.96, 0.95], frameon=False)

ax.tick_params(axis='both', which='major', labelsize=20, size=8., width=1.5, pad=15.)
xminorLocator = matplotlib.ticker.MultipleLocator(0.1)
yminorLocator = matplotlib.ticker.MultipleLocator(0.5)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)

ax.set_xlabel(r"$w_0$", fontdict={'fontsize':'xx-large'}, labelpad=15.)
ax.set_ylabel(r"$w_a$", fontdict={'fontsize':'xx-large'})

ax.set_xlim((-1.21, -0.79))
ax.set_ylim((-0.6, 0.6))

# Set size and save
P.tight_layout()
P.gcf().set_size_inches(8.,6.)

P.savefig(fig_name, transparent=True)
#P.savefig(fig_name, transparent=False)
P.show()
