#!/usr/bin/python
"""
Output marginalised Euclid covariance matrix.
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
import os, copy
import euclid

FILENAME = "euclid_covmat.dat"

cosmo = experiments.cosmo
names = ['EuclidRef',]
labels = ['DETF IV + Planck',]

#colours = [ ['#CC0000', '#F09B9B'],
#            ['#1619A1', '#B1C9FD'],
#            ['#6B6B6B', '#BDBDBD'] ]
#            ['#5B9C0A', '#BAE484'],
#            ['#FFB928', '#FFEA28'] ]

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
    zfns = ['b_HI', ]
    excl = ['Tb', 'f', 'aperp', 'apar', 'DA', 'H', 'N_eff', 'pk*', 'fs8', 'bs8']
    F, lbls = baofisher.combined_fisher_matrix( F_list,
                                                expand=zfns, names=pnames,
                                                exclude=excl )
    F1 = F; lbl1 = copy.deepcopy(lbls)
    
    # Add DETF Planck prior
    print "*** Using DETF Planck prior ***"
    l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h', 'sigma8']
    F_detf = euclid.detf_to_baofisher("DETF_PLANCK_FISHER.txt", cosmo, omegab=False)
    Fpl, lbls = baofisher.add_fisher_matrices(F, F_detf, lbls, l2, expand=True)
    
    # Decide whether to fix various parameters
    fixed_params = ['omegak',]
    Fpl, lbls = baofisher.combined_fisher_matrix( [Fpl,], expand=[], 
                 names=lbls, exclude=fixed_params )
    
    # Get indices of w0, wa
    #pw0 = lbls.index('w0'); pwa = lbls.index('wa')
    
    # Invert matrix
    cov_pl = np.linalg.inv(Fpl)
    
    # Save marginalised Euclid + Planck Fisher matrix
    np.savetxt(FILENAME, cov_pl, header=",".join(lbls))
    
    exit()
    
    # Print 1D marginals
    #fom = baofisher.figure_of_merit(pw0, pwa, None, cov=cov_pl)
    #print "%s: FOM = %3.2f" % (names[k], fom)
    #print "1D sigma(w_0) = %3.4f" % np.sqrt(cov_pl[pw0,pw0])
    #print "1D sigma(w_a) = %3.4f" % np.sqrt(cov_pl[pwa,pwa])
    
    #x = experiments.cosmo['w0']
    #y = experiments.cosmo['wa']
    
    # Plot contours for w0, wa
    #transp = [1., 0.85]
    #w, h, ang, alpha = baofisher.ellipse_for_fisher_params(pw0, pwa, None, Finv=cov_pl)
    #ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
    #            height=alpha[kk]*h, angle=ang, fc=colours[k][kk], 
    #            ec=colours[k][0], lw=1.5, alpha=transp[kk]) for kk in [1,0]]
    for e in ellipses: ax.add_patch(e)
    
    # Centroid
    ax.plot(x, y, 'ko')

################################################################################
# Add combined constraint for Facility + Euclid

# Relabel galaxy bias from Euclid and sum Facility + Euclid
for i in range(len(lbl1)):
    if "b_HI" in lbl1[i]: lbl1[i] = "gal%s" % lbl1[i]
Fc, lbls = baofisher.add_fisher_matrices(F1, F2, lbl1, lbl2, expand=True)

# Add Planck prior
l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h', 'sigma8']
F_detf = euclid.detf_to_baofisher("DETF_PLANCK_FISHER.txt", cosmo, omegab=False)
Fc, lbls = baofisher.add_fisher_matrices(Fc, F_detf, lbls, l2, expand=True)
cov_pl = np.linalg.inv(Fc)

# Plot contours for gamma, w0
transp = [1., 0.95]
w, h, ang, alpha = baofisher.ellipse_for_fisher_params(pw0, pwa, None, Finv=cov_pl)
ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
            height=alpha[kk]*h, angle=ang, fc=colours[-1][kk], 
            ec=colours[-1][0], lw=1.5, alpha=transp[kk]) for kk in [1,0]]
for e in ellipses: ax.add_patch(e)
labels += ['Combined']

print "\nCOMBINED"
pw0 = lbls.index('w0'); pwa = lbls.index('wa')
fom = baofisher.figure_of_merit(pw0, pwa, None, cov=cov_pl)
print "%s: FOM = %3.2f" % ("Combined", fom)
print "1D sigma(w_0) = %3.4f" % np.sqrt(cov_pl[pw0,pw0])
print "1D sigma(gamma) = %3.4f" % np.sqrt(cov_pl[pwa,pwa])
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
labels = [labels[k] for k in range(len(labels))]
lines = [ matplotlib.lines.Line2D([0.,], [0.,], lw=8.5, color=colours[k][0], alpha=0.65) for k in range(len(labels))]

P.gcf().legend((l for l in lines), (name for name in labels), prop={'size':'large'}, bbox_to_anchor=[0.93, 0.95])

ax.tick_params(axis='both', which='major', labelsize=20, size=8., width=1.5, pad=8.)
xminorLocator = matplotlib.ticker.MultipleLocator(0.1)
yminorLocator = matplotlib.ticker.MultipleLocator(0.5)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)

ax.set_xlabel(r"$w_0$", fontdict={'fontsize':'xx-large'}, labelpad=15.)
ax.set_ylabel(r"$w_a$", fontdict={'fontsize':'xx-large'})

ax.set_xlim((-1.21, -0.79))
ax.set_ylim((-0.5, 0.5))

# Set size and save
P.tight_layout()
P.gcf().set_size_inches(8.,6.)
P.savefig(fig_name, transparent=True)
P.show()
