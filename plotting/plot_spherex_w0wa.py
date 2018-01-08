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
from radiofisher.units import *
import radiofisher.euclid as euclid
import os, copy

USE_DETF_PLANCK_PRIOR = True
MARGINALISE_CURVATURE = True # Marginalise over Omega_K
MARGINALISE_INITIAL_PK = True # Marginalise over n_s, sigma_8
MARGINALISE_OMEGAB = True # Marginalise over Omega_baryons

cosmo = rf.experiments.cosmo

exclude_all = ['Tb', 'f', 'aperp', 'apar', 'DA', 'H', 'N_eff', 'pk*', 'fs8', 'b_1', 'sigma8tot', 'k*', 'gamma']
exclude_all += ['bs8',]
#exclude_all += ['b_HI',]

# Which MG parameters to exclude for each parametrisation
MGMODE = 1

W0GAMMA = False

param1 = 'w0'
plabel1 = r"$w_0$"
x = -1.

if W0GAMMA:
    param2 = 'gamma0'
    plabel2 = r"$\gamma$"
    y = 0.55
    fig_name = "spherex-w0gamma-%d.pdf" % MGMODE
else:
    param2 = 'wa'
    plabel2 = r"$w_a$"
    y = 0.
    fig_name = "spherex-w0wa-%d.pdf" % MGMODE

mgexclude = [ ['gamma0', 'gamma1', 'eta0', 'eta1', 'f0k*'],          # Tessa's MG params
              ['gamma1', 'eta0', 'eta1', 'alphaxi', 'f0k*', 
               'A_xi', 'logkmg'],                                      # Basic gamma=const.
              ['eta0', 'eta1', 'alphaxi', 'f0k*', 'A_xi', 'logkmg'],   # Extended gamma(z)
              ['gamma1', 'eta1', 'alphaxi', 'f0k*', 'A_xi', 'logkmg'], # Basic eta=const.
              ['gamma1', 'alphaxi', 'f0k*', 'A_xi', 'logkmg'],         # Extended eta(z)
            ]
mgnames = [r"Standard $(A_\xi, k_\xi)$", "Basic $\gamma = \mathrm{const.}$", 
           "Extended $\gamma(z)$", "Basic $\eta = \mathrm{const.}$", 
           "Extended $\eta(z)$",]

names = [ 'BOSS_mg', 'gSPHEREx1_mgphotoz', 'EuclidRef_mg' ] # 'gSPHEREx2_mgphotoz',
labels = [ 'BOSS spectro-z', 'SPHEREx 0.003', 'Euclid spectro-z',] # 'SPHEREx 0.008'

colours = [ ['#C9C9C9', '#E6E6E6'], # Grey, BOSS
            ['#5C90AE', '#80B6D6'], # SPHEREx 1
            #['#68AB68', '#93C993'], # SPHEREx 2
            ['#6B6B6B', '#BDBDBD'], # Grey, Euclid
            ['#5B9C0A', '#BAE484'], # Green, LOW
            ['#FFB928', '#FFEA28'], # Yellow, MID B1
            ['#CC0000', '#F09B9B'], # Red, MID B2
            ['#1619A1', '#B1C9FD'], # Blue, SKA2
            ['#990A9C', '#F4BAF5'], # Purple, LOW
          ]

"""
################################################################################
# Load low-z galaxy survey Fisher matrix
root = "output/" + "BOSS_mg"

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
zfns = ['b_HI',]
excl = exclude_all + mgexclude[MGMODE]
F_lowz, lbls_lowz = rf.combined_fisher_matrix( F_list,
                                               expand=zfns, names=pnames,
                                               exclude=excl )
# Relabel galaxy bias from low-z survey
for i in range(len(lbls_lowz)):
    if "b_HI" in lbls_lowz[i]: lbls_lowz[i] = "lowz%s" % lbls_lowz[i]

################################################################################
"""

# Fiducial value and plotting
fig = P.figure()
ax = fig.add_subplot(111)

_k = range(len(names)) #[::-1]
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
    
    zfns = ['b_HI',]
    #zfns = ['bs8',]
    excl = exclude_all + mgexclude[MGMODE]
    F, lbls = rf.combined_fisher_matrix( F_list, expand=zfns, names=pnames,
                                         exclude=excl )
    
    # Relabel galaxy bias from low-z survey and sum current survey + low-z
    #if 'BOSS' not in labels[k]:
    #    F, lbls = rf.add_fisher_matrices(F_lowz, F, lbls_lowz, lbls, expand=True)
    #print lbls
    
    # Add Planck prior
    print "*** Using DETF Planck prior ***"
    l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h', 'sigma_8']
    F_detf = euclid.detf_to_rf("DETF_PLANCK_FISHER.txt", cosmo, omegab=False)
    Fpl, lbls = rf.add_fisher_matrices(F, F_detf, lbls, l2, expand=True)
    
    # Decide whether to fix various parameters
    fixed_params = []
    if not MARGINALISE_CURVATURE: fixed_params += ['omegak',]
    if not MARGINALISE_INITIAL_PK: fixed_params += ['n_s', 'sigma8']
    if not MARGINALISE_OMEGAB: fixed_params += ['omega_b',]
    
    fixed_params += ['omegak', 'gamma0']
    
    if len(fixed_params) > 0:
        Fpl, lbls = rf.combined_fisher_matrix( [Fpl,], expand=[], 
                     names=lbls, exclude=fixed_params )
    print lbls
    #rf.plot_corrmat(Fpl, lbls)
    
    for l, f in zip(lbls, np.diag(Fpl)):
        print "%10s %3.3e" % (l, f)
    
    # Get indices of selected parameters
    p1 = lbls.index(param1)
    p2 = lbls.index(param2)
    
    #print Fpl[p1,p1], Fpl[p1,p2], Fpl[p2,p2]
    
    print "-"*50
    print names[k]
    print "-"*50
    
    # Invert matrix
    cov_pl = np.linalg.inv(Fpl)
    
    # Print 1D marginals
    print "1D sigma(%s) = %3.4f" % (param1, np.sqrt(cov_pl[p1,p1]))
    #print "1D sigma(%s) = %3.4f" % (param2, np.sqrt(cov_pl[p2,p2]))
    
    # Plot contours for gamma, w0
    #transp = [1., 0.85]
    transp = [0.98, 0.8]
    w, h, ang, alpha = rf.ellipse_for_fisher_params(p1, p2, None, Finv=cov_pl)
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                height=alpha[kk]*h, angle=ang, fc=colours[k][kk], 
                ec=colours[k][0], lw=1.5, alpha=transp[kk]) for kk in [1,0]]
    for e in ellipses: ax.add_patch(e)
    
    # Centroid
    if k == 1: ax.plot(x, y, 'ko')

#P.figtext(0.18, 0.22, "Combined w. Planck + SKA1 gal.", fontsize=15)

# Report on what options were used
print "-"*50
s1 = "Marginalised over Omega_K" if MARGINALISE_CURVATURE else "Fixed Omega_K"
s2 = "Marginalised over ns, sigma8" if MARGINALISE_INITIAL_PK else "Fixed ns, sigma8"
s3 = "Marginalised over Omega_b" if MARGINALISE_OMEGAB else "Fixed Omega_b"
print "NOTE:", s1
print "NOTE:", s2
print "NOTE:", s3

# Legend
if not W0GAMMA:
    order = (1,0,2,)
    #order = (0,1,)
    #order = range(len(labels))
    labels = [labels[k] for k in order]
    lines = [ matplotlib.lines.Line2D([0.,], [0.,], lw=8.5, color=colours[k][0], alpha=0.65) for k in order]
    #leg = P.legend((l for l in lines), (name for name in labels), prop={'size':'large'}, frameon=True, ncol=1, loc='upper right') # bbox_to_anchor=[0.94, 0.95]
    leg = P.legend((l for l in lines), (name for name in labels), prop={'size':'large'}, frameon=False, ncol=1, loc='upper right') #loc='upper center')
    leg.get_frame().set_edgecolor('w')
    leg.get_frame().set_alpha(0.8)

ax.tick_params(axis='both', which='major', labelsize=20, size=8., width=1.5, pad=15.)
ax.tick_params(axis='both', which='minor', labelsize=20, size=5., width=1.5)

ax.set_xlabel(plabel1, fontdict={'fontsize':'xx-large'}, labelpad=15.)
ax.set_ylabel(plabel2, fontdict={'fontsize':'xx-large'})

if W0GAMMA:
    # w0, gamma
    ax.set_xlim((-1.52, -0.48))
    ax.set_ylim((0.4, 0.7))
    #ax.set_ylim((0.4, 0.77))
else:
    # w0, wa
    ax.set_xlim((-1.5, -0.5))
    ax.set_ylim((-1.4, 1.4))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))

# Set size and save
P.tight_layout()
P.gcf().set_size_inches(8.,6.)

P.savefig(fig_name, transparent=True)
P.show()
