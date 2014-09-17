#!/usr/bin/python
"""
Plot 2D constraints on a pair of parameters.
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

#fig_name = "pub-w0omegaDE.pdf"
#fig_name = "pub-w0wa-okmarg.pdf"
nsig = 4. # No. of sigma to plot out to
aspect = 1. #1.7 # Aspect ratio of range (w = aspect * h)

param1 = "w0"
label1 = "$w_0$"
fid1 = cosmo['w0']

param2 = "wa"
label2 = "$w_a$"
fid2 = cosmo['wa']

#param2 = "omegaDE"
#label2 = "$\Omega_\mathrm{DE}$"
#fid2 = cosmo['omega_lambda_0']


USE_DETF_PLANCK_PRIOR = False # FIXME True
MARGINALISE_CURVATURE = True # Marginalise over Omega_K
MARGINALISE_INITIAL_PK = True # Marginalise over n_s, sigma_8
MARGINALISE_OMEGAB = True # Marginalise over Omega_baryons

names = ['EuclidRef', 'cexptL', 'iexptM'] #, 'exptS']
labels = ['DETF IV', 'Facility', 'Mature'] #, 'Snapshot']
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
    # Actually, (aperp, apar) are (D_A, H)
    pnames = ['A', 'b_HI', 'Tb', 'sigma_NL', 'sigma8', 'n_s', 'f', 'aperp', 'apar', 
             'omegak', 'omegaDE', 'w0', 'wa', 'h', 'gamma']
    pnames += ["pk%d" % i for i in range(kc.size)]
    
    zfns = [1,]
    excl = [2,  6,7,8,  14] # omega_k free 4,5
    excl  += [i for i in range(len(pnames)) if "pk" in pnames[i]]
    
    F, lbls = rf.combined_fisher_matrix( F_list,
                                                expand=zfns, names=pnames,
                                                exclude=excl )
    # Add Planck prior
    #Fpl = euclid.add_detf_planck_prior(F, lbls, info=False)
    #Fpl = euclid.add_planck_prior(F, lbls, info=False)
    if USE_DETF_PLANCK_PRIOR:
        # DETF Planck prior
        print "*** Using DETF Planck prior ***"
        l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h']
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
    fixed_params = []
    if not MARGINALISE_CURVATURE: fixed_params += ['omegak',]
    if not MARGINALISE_INITIAL_PK: fixed_params += ['n_s', 'sigma8']
    if not MARGINALISE_OMEGAB: fixed_params += ['omega_b',]
    
    if len(fixed_params) > 0:
        Fpl, lbls = rf.combined_fisher_matrix( [Fpl,], expand=[], 
                     names=lbls, exclude=[lbls.index(p) for p in fixed_params] )
    
    # Really hopeful H0 prior
    #ph = lbls.index('h')
    #Fpl[ph, ph] += 1./(0.001)**2.
    
    # Get indices of params
    p1 = lbls.index(param1)
    p2 = lbls.index(param2)
    
    print "-"*50
    print names[k]
    print "-"*50
    
    # Invert matrix
    cov_pl = np.linalg.inv(Fpl)
    
    # Calculate FOM
    fom = rf.figure_of_merit(p1, p2, None, cov=cov_pl)
    print "%s: FOM = %3.2f" % (names[k], fom)
    print "1D sigma(p1) = %3.4f" % np.sqrt(cov_pl[p1,p1])
    print "1D sigma(p2) = %3.4f" % np.sqrt(cov_pl[p2,p2])
    
    x = fid1
    y = fid2
    
    # Plot 2D contours for params
    w, h, ang, alpha = rf.ellipse_for_fisher_params(p1, p2, None, Finv=cov_pl)
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                height=alpha[kk]*h, angle=ang, fc=colours[k][kk], 
                ec=colours[k][0], lw=1.5, alpha=1.) for kk in [1,0]]
    for e in ellipses: ax.add_patch(e)
    
    # Use 4-sigma out of middle experiment to decide on scale of plot
    if k == 1:
        dx = aspect * nsig * np.sqrt(cov_pl[p1,p1])
        dy = nsig * np.sqrt(cov_pl[p2,p2])
    
    # Centroid
    ax.plot(x, y, 'kx')


# Report on what options were used
print "-"*50
s1 = "Marginalised over Omega_K" if MARGINALISE_CURVATURE else "Fixed Omega_K"
s2 = "Marginalised over ns, sigma8" if MARGINALISE_INITIAL_PK else "Fixed ns, sigma8"
s3 = "Marginalised over Omega_b" if MARGINALISE_OMEGAB else "Fixed Omega_b"
print "NOTE:", s1
print "NOTE:", s2
print "NOTE:", s3

# Legend
labels = [labels[k] + " + Planck" for k in range(len(labels))]
lines = [ matplotlib.lines.Line2D([0.,], [0.,], lw=8.5, color=colours[k][0], alpha=0.65) for k in range(len(labels))]

P.gcf().legend((l for l in lines), (name for name in labels), loc='upper right', prop={'size':'large'})

fontsize = 20
for tick in ax.xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in ax.yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)

xminorLocator = matplotlib.ticker.MultipleLocator(0.1)
yminorLocator = matplotlib.ticker.MultipleLocator(0.5)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)

ax.set_xlabel(label1, fontdict={'fontsize':'xx-large'}, labelpad=15.)
ax.set_ylabel(label2, fontdict={'fontsize':'xx-large'}, labelpad=15.)

ax.set_xlim((x-dx, x+dx))
ax.set_ylim((y-dy, y+dy))

# Set size and save
P.tight_layout()
P.gcf().set_size_inches(8.,6.)
#P.savefig(fig_name, transparent=True)
P.show()
