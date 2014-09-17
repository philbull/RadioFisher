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

#fig_name = "ska-w0gamma.pdf"
fig_name = "ska-w0gamma-both.pdf"
fig_name = "ska-w0gamma-combined-SUR.pdf"
fig_name = "ska-w0gamma-combined-MID.pdf"
fig_name = "ska-w0gamma-combined-MID.pdf"
#fig_name = "BINGO-w0gamma.pdf"
fig_name = "ska-rsd-w0gamma.pdf"

USE_DETF_PLANCK_PRIOR = True
MARGINALISE_CURVATURE = True # Marginalise over Omega_K
MARGINALISE_INITIAL_PK = True # Marginalise over n_s, sigma_8
MARGINALISE_OMEGAB = True # Marginalise over Omega_baryons

cosmo = rf.experiments.cosmo

names = ['EuclidRef', 'cSKA1MIDfull1'] #'SKA1SURfull1',]# 'cSKA1MIDfull1']
labels = ['Euclid', 'SKA1-MID (B1)'] #'SKA1-SUR (B1)',]# 'SKA1-MID (B1)']

#names = ['BINGO',]
#labels = ['BINGO',]

colours = [ ['#CC0000', '#F09B9B'],
            ['#1619A1', '#B1C9FD'],
            ['#6B6B6B', '#BDBDBD'] ]
#            ['#5B9C0A', '#BAE484'],
#            ['#FFB928', '#FFEA28'] ]

names = ["SKAHI73", 'cSKA1MIDfull2', "EuclidRef", 'cSKA1MIDfull1'] # 'SKA1SURfull1', 'cSKA1MIDfull1']
labels = ['SKA2 HI gal.', 'SKA1-MID (B2)', 'Euclid', 'SKA1-MID (B1)'] # 'SKA1-SUR (B1)', 'SKA1-MID (B1)']

colours = [ ['#5B9C0A', '#BAE484'], 
            ['#1619A1', '#B1C9FD'], 
            ['#990A9C', '#F4BAF5'], 
            ['#CC0000', '#F09B9B'], 
            ['#FFB928', '#FFEA28'],
            ['#6B6B6B', '#BDBDBD'],
            ['#5B9C0A', '#BAE484'],
            ['#6B6B6B', '#BDBDBD'] ]

colours = [ ['#1619A1', '#B1C9FD'], 
            ['#FFA728', '#F8D24B'],
            ['#6B6B6B', '#BDBDBD'],
            ['#CC0000', '#F09B9B'], ]

names = ['SKA1MIDfull1', 'fSKA1SURfull1', 'EuclidRef', 'gSKA2']
labels = ['SKA1-MID B1 (IM)', 'SKA1-SUR B1 (IM)', 'Euclid (gal.)', 'Full SKA (gal.)']


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
    excl = ['Tb', 'f', 'aperp', 'apar', 'DA', 'H', 'N_eff', 'pk*', 'fs8', 'bs8']
    F, lbls = rf.combined_fisher_matrix( F_list,
                                                expand=zfns, names=pnames,
                                                exclude=excl )
    if 'Euclid' in names[k]:
        F1 = F; lbl1 = copy.deepcopy(lbls)
    else:
        F2 = F; lbl2 = copy.deepcopy(lbls)
    
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
    
    #fixed_params += ['wa', 'gamma', 'omegak']
    
    if len(fixed_params) > 0:
        Fpl, lbls = rf.combined_fisher_matrix( [Fpl,], expand=[], 
                     names=lbls, exclude=fixed_params )
    
    # Really hopeful H0 prior
    #ph = lbls.index('h')
    #Fpl[ph, ph] += 1./(0.012)**2.
    
    # Get indices of w0, wa
    pw0 = lbls.index('w0'); pgam = lbls.index('gamma')
    print lbls
    
    print "-"*50
    print names[k]
    print "-"*50
    
    # Invert matrix
    cov_pl = np.linalg.inv(Fpl)
    
    # Print 1D marginals
    print "1D sigma(w_0) = %3.4f" % np.sqrt(cov_pl[pw0,pw0])
    print lbls[pw0]
    print "1D sigma(gamma) = %3.4f" % np.sqrt(cov_pl[pgam,pgam])
    
    x = rf.experiments.cosmo['gamma']
    y = rf.experiments.cosmo['w0']
    
    print "cov(w0, gam):", cov_pl[pw0, pw0], cov_pl[pw0, pgam], cov_pl[pgam, pgam]
    
    # Plot contours for gamma, w0
    transp = [1., 0.85]
    w, h, ang, alpha = rf.ellipse_for_fisher_params(pgam, pw0, None, Finv=cov_pl)
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                height=alpha[kk]*h, angle=ang, fc=colours[k][kk], 
                ec=colours[k][0], lw=1.5, alpha=transp[kk]) for kk in [1,0]]
    for e in ellipses: ax.add_patch(e)
    
    # Centroid
    ax.plot(x, y, 'ko')
    



"""
################################################################################
# Add combined constraint for Facility + Euclid

# Relabel galaxy bias from Euclid and sum Facility + Euclid
for i in range(len(lbl1)):
    if "b_HI" in lbl1[i]: lbl1[i] = "gal%s" % lbl1[i]
Fc, lbls = rf.add_fisher_matrices(F1, F2, lbl1, lbl2, expand=True)

# Add Planck prior
l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h', 'sigma8']
F_detf = euclid.detf_to_rf("DETF_PLANCK_FISHER.txt", cosmo, omegab=False)
Fc, lbls = rf.add_fisher_matrices(Fc, F_detf, lbls, l2, expand=True)
cov_pl = np.linalg.inv(Fc)

# Plot contours for gamma, w0
transp = [1., 0.95]
w, h, ang, alpha = rf.ellipse_for_fisher_params(pgam, pw0, None, Finv=cov_pl)
ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
            height=alpha[kk]*h, angle=ang, fc=colours[-1][kk], 
            ec=colours[-1][0], lw=1.5, alpha=transp[kk]) for kk in [1,0]]
for e in ellipses: ax.add_patch(e)
labels += ['Combined']

print "\nCOMBINED"
pw0 = lbls.index('w0'); pgam = lbls.index('gamma')
print "1D sigma(w_0) = %3.4f" % np.sqrt(cov_pl[pw0,pw0])
print "1D sigma(gamma) = %3.4f" % np.sqrt(cov_pl[pgam,pgam])
################################################################################
"""

"""
# Plot datapoints for other theories
ax.plot(0.68, -0.8, 'kD') # DGP
ax.plot(0.4, -0.99, 'kD') # f(r)
ax.plot(0.48, -1.22, 'kD') # Minimal massive bigravity, arXiv:1404.4061

P.annotate("DGP", xy=(0.68, -0.8), xytext=(0., -20.), fontsize='large', 
                       textcoords='offset points', ha='center', va='center')

P.annotate("f(R)", xy=(0.4, -0.99), xytext=(0., -20.), fontsize='large', 
                       textcoords='offset points', ha='center', va='center')

P.annotate("Mass. Bigrav.", xy=(0.48, -1.22), xytext=(0., -20.), fontsize='large', 
                       textcoords='offset points', ha='center', va='center')
"""
################################################################################

# Report on what options were used
print "-"*50
s1 = "Marginalised over Omega_K" if MARGINALISE_CURVATURE else "Fixed Omega_K"
s2 = "Marginalised over ns, sigma8" if MARGINALISE_INITIAL_PK else "Fixed ns, sigma8"
s3 = "Marginalised over Omega_b" if MARGINALISE_OMEGAB else "Fixed Omega_b"
print "NOTE:", s1
print "NOTE:", s2
print "NOTE:", s3

# Legend
#labels = [labels[k] for k in range(len(labels))]
#lines = [ matplotlib.lines.Line2D([0.,], [0.,], lw=8.5, color=colours[k][0], alpha=0.65) for k in range(len(labels))]
#P.gcf().legend((l for l in lines), (name for name in labels), prop={'size':'large'}, bbox_to_anchor=[0.96, 0.95], frameon=False)

labels = [labels[k] for k in [0,1,3,2]]
lines = [ matplotlib.lines.Line2D([0.,], [0.,], lw=8.5, color=colours[k][0], alpha=0.65) for k in [0,1,3,2]]
P.gcf().legend((l for l in lines), (name for name in labels), prop={'size':'large'}, bbox_to_anchor=[0.96, 0.95], frameon=False)

ax.tick_params(axis='both', which='major', labelsize=20, size=8., width=1.5, pad=10.)
xminorLocator = matplotlib.ticker.MultipleLocator(0.1)
yminorLocator = matplotlib.ticker.MultipleLocator(0.5)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)

ax.set_xlabel(r"$\gamma$", fontdict={'fontsize':'xx-large'}, labelpad=10.)
ax.set_ylabel(r"$w_0$", fontdict={'fontsize':'xx-large'}, labelpad=10.)

ax.set_xlim((0.44, 0.7))
#ax.set_xlim((0.37, 0.72))
#ax.set_xlim((0.17, 0.92)) # BINGO
#ax.set_ylim((-1.22, -0.73))

ax.set_ylim((-1.3, -0.7))
#ax.set_ylim((-1.36, -0.54)) # Original
#ax.set_ylim((-1.29, -0.74))
#ax.set_ylim((-2.36, 0.54)) # BINGO

# Set size and save
P.tight_layout()
P.gcf().set_size_inches(8.,6.)
P.savefig(fig_name, transparent=True)
P.show()
