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

exclude_all = ['Tb', 'f', 'aperp', 'apar', 'DA', 'H', 'N_eff', 'pk*', 'gamma', 'fs8', 'bs8'] # 'b_1']

# Which MG parameters to exclude for each parametrisation
MGMODE = 0
fig_name = "mg-parametrisation-%d.pdf" % MGMODE

param1 = 'A_xi'
plabel1 = r"$A_\xi$"
x = rf.experiments.cosmo['A_xi']

param2 = 'k_mg' #'w0'
plabel2 = r"$k_\xi$" #"$w_0$"
y = rf.experiments.cosmo['k_mg']

mgexclude = [ ['gamma0', 'gamma1', 'eta0', 'eta1', 'f0k*'],          # Tessa's MG params
              ['gamma1', 'eta0', 'eta1', 'alphaxi', 'f0k*', 
               'A_xi', 'k_mg'],                                      # Basic gamma=const.
              ['eta0', 'eta1', 'alphaxi', 'f0k*', 'A_xi', 'k_mg'],   # Extended gamma(z)
              ['gamma1', 'eta1', 'alphaxi', 'f0k*', 'A_xi', 'k_mg'], # Basic eta=const.
              ['gamma1', 'alphaxi', 'f0k*', 'A_xi', 'k_mg'],         # Extended eta(z)
            ]
mgnames = [r"Standard $(A_\xi, k_\xi)$", "Basic $\gamma = \mathrm{const.}$", 
           "Extended $\gamma(z)$", "Basic $\eta = \mathrm{const.}$", 
           "Extended $\eta(z)$",]

#names = ['SKA1MID350_mg_Dz_kmg0.01', 'SKA1MID900_mg_Dz_kmg0.01',
#         'fSKA1SUR350_mg_Dz_kmg0.01', 'fSKA1SUR650_mg_Dz_kmg0.01',
#         'gSKA2_mg_Dz_kmg0.01']
#labels = ['SKA1-MID 350 (IM)', 'SKA1-MID 900 (IM)',
#          'SKA1-SUR 350 (IM)', 'SKA1-SUR 650 (IM)',
#          'SKA2 (gal.)',]

names = ['EuclidRef_mg_Dz_kmg0.01', 'SKA1MID350_mg_Dz_kmg0.01', 'gSKA2_mg_Dz_kmg0.01']
labels = ['Euclid (gal.)', 'SKA1-MID 350 (IM)', 'Full SKA (gal.)',]

colours = [ ['#6B6B6B', '#BDBDBD'], # Grey, Euclid
            ['#1619A1', '#B1C9FD'], # Blue, MID
            ['#CC0000', '#F09B9B'], # Red, SKA2
            ['#990A9C', '#F4BAF5'], 
            ['#FFB928', '#FFEA28'], 
            ['#5B9C0A', '#BAE484'], 
            ['#6B6B6B', '#BDBDBD'],
            ['#5B9C0A', '#BAE484'],
            ['#6B6B6B', '#BDBDBD'] ]
#            ['#5B9C0A', '#BAE484'],
#            ['#FFB928', '#FFEA28'] ]


################################################################################
# Load low-z galaxy survey Fisher matrix
root = "output/" + "BOSS_mg_Dz_kmg0.01"

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
    
    for n in range(len(F_list)):
        print "%5.5e  %5.5e" % (F_list[n][pnames.index("A_xi"), pnames.index("A_xi")],
                                F_list[n][pnames.index("k_mg"), pnames.index("k_mg")] )
    
    zfns = ['b_HI',]
    excl = exclude_all + mgexclude[MGMODE]
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
    
    # Decide whether to fix various parameters
    fixed_params = []
    if not MARGINALISE_CURVATURE: fixed_params += ['omegak',]
    if not MARGINALISE_INITIAL_PK: fixed_params += ['n_s', 'sigma8']
    if not MARGINALISE_OMEGAB: fixed_params += ['omega_b',]
    
    #fixed_params += ['omegak', 'wa', 'b1']
    
    if len(fixed_params) > 0:
        Fpl, lbls = rf.combined_fisher_matrix( [Fpl,], expand=[], 
                     names=lbls, exclude=fixed_params )
    print lbls
    #rf.plot_corrmat(Fpl, lbls)
    
    # DEBUG: Output correlation coeffs
    F11 = Fpl[lbls.index("A_xi"), lbls.index("A_xi")]
    F22 = Fpl[lbls.index("k_mg"), lbls.index("k_mg")]
    F12 = Fpl[lbls.index("A_xi"), lbls.index("k_mg")]
    F13 = Fpl[lbls.index("A_xi"), lbls.index("w0")]
    F33 = Fpl[lbls.index("w0"), lbls.index("w0")]
    print "A_xi - A_xi: %3.3e  %3.3f" % (F11, 1.)
    print "A_xi - k_mg: %3.3e  %3.3f" % (F12, F12 / np.sqrt(F11 * F22))
    print "k_mg - k_mg: %3.3e  %3.3f" % (F22, 1.)
    print "A_xi - w_0:  %3.3e  %3.3f" % (F13, F13 / np.sqrt(F11 * F33))
    
    # Get indices of selected parameters
    p1 = lbls.index(param1)
    p2 = lbls.index(param2)
    
    print Fpl[p1,p1], Fpl[p1,p2], Fpl[p2,p2]
    
    print "-"*50
    print names[k]
    print "-"*50
    
    # Invert matrix
    cov_pl = np.linalg.inv(Fpl)
    
    # Print 1D marginals
    print "1D sigma(%s) = %3.4f" % (param1, np.sqrt(cov_pl[p1,p1]))
    print "1D sigma(%s) = %3.4f" % (param2, np.sqrt(cov_pl[p2,p2]))
    
    # Plot contours for gamma, w0
    transp = [1., 0.85]
    w, h, ang, alpha = rf.ellipse_for_fisher_params(p1, p2, None, Finv=cov_pl)
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                height=alpha[kk]*h, angle=ang, fc=colours[k][kk], 
                ec=colours[k][0], lw=1.5, alpha=transp[kk]) for kk in [1,0]]
    for e in ellipses: ax.add_patch(e)
    
    # Centroid
    if k == 1: ax.plot(x, y, 'ko')

#P.figtext(0.18, 0.22, "Combined w. Planck + SKA1 gal.", fontsize=15)

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
w, h, ang, alpha = rf.ellipse_for_fisher_params(pw0, pwa, None, Finv=cov_pl)
ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
            height=alpha[kk]*h, angle=ang, fc=colours[-1][kk], 
            ec=colours[-1][0], lw=1.5, alpha=transp[kk]) for kk in [1,0]]
for e in ellipses: ax.add_patch(e)
labels += ['Combined']

print "\nCOMBINED"
pw0 = lbls.index('w0'); pwa = lbls.index('wa')
print "1D sigma(w_0) = %3.4f" % np.sqrt(cov_pl[pw0,pw0])
print "1D sigma(gamma) = %3.4f" % np.sqrt(cov_pl[pwa,pwa])
################################################################################
"""

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
leg = P.gcf().legend((l for l in lines), (name for name in labels), prop={'size':'medium'}, bbox_to_anchor=[0.94, 0.95], frameon=True)
leg.get_frame().set_edgecolor('w')
leg.get_frame().set_alpha(0.8)

ax.tick_params(axis='both', which='major', labelsize=20, size=8., width=1.5, pad=15.)
xminorLocator = matplotlib.ticker.MultipleLocator(0.1)
yminorLocator = matplotlib.ticker.MultipleLocator(0.5)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)

ax.set_xlabel(plabel1, fontdict={'fontsize':'xx-large'}, labelpad=15.)
ax.set_ylabel(plabel2, fontdict={'fontsize':'xx-large'})

#ax.set_xlim((-1.21, -0.79))
#ax.set_ylim((-0.6, 0.6))

# Set size and save
P.tight_layout()
P.gcf().set_size_inches(8.,6.)

P.savefig(fig_name, transparent=True)
P.show()
