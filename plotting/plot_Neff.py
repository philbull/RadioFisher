#!/usr/bin/python
"""
Plot 1D constraints on Neff.
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

cosmo = experiments.cosmo

fig_name = "pub-Neff.pdf"

param1 = "N_eff"
label1 = "$N_\mathrm{eff}$"
fid1 = cosmo['N_eff']

USE_DETF_PLANCK_PRIOR = True
MARGINALISE_CURVATURE = True    # Marginalise over Omega_K
MARGINALISE_INITIAL_PK = True   # Marginalise over (n_s, sigma_8)
MARGINALISE_OMEGAB = True      # Marginalise over Omega_baryons
MARGINALISE_W0WA = True         # Marginalise over (w0, wa)

#names = ['EuclidRef', 'cexptL', 'iexptM'] #, 'exptS']
#labels = ['DETF IV', 'Facility', 'Mature'] #, 'Snapshot']
names = ['cexptL', 'iexptM', 'exptS']
labels = ['Facility', 'Mature', 'Snapshot']

colours = [ ['#CC0000', '#F09B9B'],
            ['#1619A1', '#B1C9FD'],
            ['#5B9C0A', '#BAE484'],
            ['#FFB928', '#FFEA28'] ]

# Fiducial value and plotting
fig = P.figure()
ax = fig.add_subplot(111)

Nexpt = len(names)
m = 0
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
    pnames = baofisher.load_param_names(root+"-fisher-full-0.dat")
    zfns = [1,]
    excl = [2,  6,7,8,  14]
    excl  += [i for i in range(len(pnames)) if "pk" in pnames[i]]
    
    F, lbls = baofisher.combined_fisher_matrix( F_list, expand=zfns, 
                                                names=pnames, exclude=excl )
    # Add Planck prior
    #Fpl = euclid.add_detf_planck_prior(F, lbls, info=False)
    #Fpl = euclid.add_planck_prior(F, lbls, info=False)
    if USE_DETF_PLANCK_PRIOR:
        # DETF Planck prior
        print "*** Using DETF Planck prior ***"
        l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h']
        F_detf = euclid.detf_to_baofisher("DETF_PLANCK_FISHER.txt", cosmo, omegab=False)
        Fpl, lbls = baofisher.add_fisher_matrices(F, F_detf, lbls, l2, expand=True)
    else:
        # Euclid Planck prior
        print "*** Using Euclid (Mukherjee) Planck prior ***"
        l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h']
        Fe = euclid.planck_prior_full
        F_eucl = euclid.euclid_to_baofisher(Fe, cosmo)
        Fpl, lbls = baofisher.add_fisher_matrices(F, F_eucl, lbls, l2, expand=True)
    
    # Decide whether to fix various parameters
    fixed_params = []
    if not MARGINALISE_CURVATURE: fixed_params += ['omegak',]
    if not MARGINALISE_INITIAL_PK: fixed_params += ['n_s', 'sigma8']
    if not MARGINALISE_OMEGAB: fixed_params += ['omega_b',]
    if not MARGINALISE_W0WA: fixed_params += ['w0', 'wa']
    
    if len(fixed_params) > 0:
        print "REMOVING:", fixed_params
        Fpl, lbls = baofisher.combined_fisher_matrix( [Fpl,], expand=[], 
                     names=lbls, exclude=[lbls.index(p) for p in fixed_params] )
    
    # Get indices
    pNeff = lbls.index('N_eff'); psig8 = lbls.index('sigma8')
    
    # Invert matrix
    cov_pl = np.linalg.inv(Fpl)
    
    x = experiments.cosmo['N_eff']
    y = experiments.cosmo['sigma_8']
    
    # Plot contours for N_eff, sigma_8
    w, h, ang, alpha = baofisher.ellipse_for_fisher_params(pNeff, psig8, None, Finv=cov_pl)
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                height=alpha[kk]*h, angle=ang, fc=colours[k][kk], 
                ec=colours[k][0], lw=1.5, alpha=1.) for kk in [1,0]]
    for e in ellipses: ax.add_patch(e)
    
    # Centroid
    ax.plot(x, y, 'kx', markersize=8., markeredgewidth=1.5)


# Report on what options were used
print "-"*50
s1 = "Marginalised over Omega_K" if MARGINALISE_CURVATURE else "Fixed Omega_K"
s2 = "Marginalised over ns, sigma8" if MARGINALISE_INITIAL_PK else "Fixed ns, sigma8"
s3 = "Marginalised over Omega_b" if MARGINALISE_OMEGAB else "Fixed Omega_b"
s4 = "Marginalised over w0, wa" if MARGINALISE_W0WA else "Fixed w0, wa"
print "NOTE:", s1
print "NOTE:", s2
print "NOTE:", s3
print "NOTE:", s4

fontsize = 18
for tick in ax.xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in ax.yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)

xminorLocator = matplotlib.ticker.MultipleLocator(0.5)
yminorLocator = matplotlib.ticker.MultipleLocator(0.1)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)

# Legend
labels = [labels[k] + " + Planck" for k in range(len(labels))]
lines = [ matplotlib.lines.Line2D([0.,], [0.,], lw=8.5, color=colours[k][0], alpha=0.65) for k in range(len(labels))]

P.gcf().legend((l for l in lines), (name for name in labels), prop={'size':'large'}, bbox_to_anchor=[0.52,0.95])

ax.set_xlabel("$\mathrm{N}_\mathrm{eff}$", fontdict={'fontsize':'xx-large'}, labelpad=15.)
ax.set_ylabel("$\sigma_8$", fontdict={'fontsize':'xx-large'}, labelpad=15.)

ax.set_xlim((1.8, 4.3))
ax.set_ylim((0.67, 1.02))

ax.tick_params(axis='both', which='both', length=6., width=1.5)

# Set size and save
P.tight_layout()
P.gcf().set_size_inches(8.,6.)
P.savefig(fig_name, transparent=True)
P.show()
