#!/usr/bin/python
"""
Plot 2D constraints on (w0, wa).
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

fig_name = "pub-w0wa-okmarg.pdf"
#fig_name = "pub-w0wa-okfixed.pdf"

USE_DETF_PLANCK_PRIOR = True
MARGINALISE_CURVATURE = True # Marginalise over Omega_K
MARGINALISE_INITIAL_PK = True # Marginalise over n_s, sigma_8
MARGINALISE_OMEGAB = True # Marginalise over Omega_baryons

cosmo = experiments.cosmo
names = ['EuclidRef', 'cexptL', 'yCHIME'] #'iexptM'] #, 'exptS']
labels = ['DETF IV + Planck', 'Facility + Planck', 'Pathfinder + Planck'] #, 'Snapshot']


names = ['yCHIME', 'yCHIME_nocut',] #'iexptM'] #, 'exptS']
labels = ['CHIME', 'CHIME Nocut',] #, 'Snapshot']

names = ['cSKA1MIDfull1', 'cSKA1MIDfull2', 'fSKA1SURfull1', 'fSKA1SURfull2', 'BOSS']
labels = ['SKA1-MID B1', 'SKA1-MID B2', 'SKA1-SUR B1', 'SKA1-SUR B2', 'BOSS']

names = ['EuclidRef_full', 'EuclidRef_fix', 'gSKA2_full', 'SKA1MIDfull2_full']
labels = ['Euclid', 'Euclid_nsfix', 'SKA2', 'SKA1-MID B2']

#names = ['EuclidRef', 'iexptO', 'cexptL', 'iexptM']
#labels = ['DETF IV', 'Optimal', 'Facility', 'Pathfinder']

#names = ['cSKA1MID', 'SKAMIDdishonly', 'SKAMIDionly5k'] #, 'exptS'] # 'EuclidRef',
#labels = ['SKA1-MID (190 dish) Combined', 'SKA1-MID (190 dish) Dish-only', 'SKA1-MID (190 dish) Interferom.-only'] #, 'Snapshot'] # 'DETF IV'

#names = ['EuclidRef', 'EuclidRefLINEAR', 'EuclidRefLINEAR2'] #, 'exptS']
#labels = ['DETF IV', 'DETF IV L', 'DETF IV L2'] #, 'Snapshot']
#names = ['cexptL_Sarea2k', 'cexptL_Sarea5k', 'cexptL_Sarea10k', 'cexptL_Sarea15k', 'cexptL_Sarea20k', 'cexptL_Sarea30k', 'cexptL_Sarea25k', 'cexptL_Sarea1k']

colours = [ ['#CC0000', '#F09B9B'],
            ['#1619A1', '#B1C9FD'],
            ['#5B9C0A', '#BAE484'],
            ['#FFB928', '#FFEA28'],
            ['#FFB928', '#FFEA28'],
            ['#FFB928', '#FFEA28'],
            ['#FFB928', '#FFEA28'], ]

# Fiducial value and plotting
fig = P.figure()
ax = fig.add_subplot(111)

_k = range(len(names))[::-1]
for k in _k:
    root = "output/" + names[k]
    
    print ">"*50
    print "We're doing", names[k]

    # Load cosmo fns.
    dat = np.atleast_2d( np.genfromtxt(root+"-cosmofns-zc.dat") ).T
    zc, Hc, dAc, Dc, fc = dat
    zs, Hs, dAs, Ds, fs = np.genfromtxt(root+"-cosmofns-smooth.dat").T
    kc = np.genfromtxt(root+"-fisher-kc.dat").T
    
    # Load Fisher matrices as fn. of z
    Nbins = zc.size
    F_list = [np.genfromtxt(root+"-fisher-full-%d.dat" % i) for i in range(Nbins)]
    #F_list = [np.genfromtxt(root+"-fisher-full-%d.dat" % i) for i in np.where(zc>0.5)[0]]
    #Nbins = np.where(zc > 0.5)[0].size
    #if Nbins == 0: continue
    
    # EOS FISHER MATRIX
    pnames = baofisher.load_param_names(root+"-fisher-full-0.dat")
    zfns = ['b_HI',]
    #excl = ['Tb', 'f', 'aperp', 'apar', 'DA', 'H', 'gamma', 'N_eff', 'pk*', 'fs8', 'bs8']
    excl = ['Tb', 'f', 'aperp', 'apar', 'DA', 'H', 'N_eff', 'pk*', 'fs8', 'bs8']
    F, lbls = baofisher.combined_fisher_matrix( F_list,
                                                expand=zfns, names=pnames,
                                                exclude=excl )
    #np.savetxt("fisher_%s_gammaz.dat"%names[k], F, header=" ".join(lbls))
    #np.savetxt("fisher_%s_zcut05.dat"%names[k], F, header=" ".join(lbls))
    #np.savetxt("fisher_%s_zbins.dat"%names[k], zc[zc > 0.5])
    
    pok = lbls.index('omegak')
    pns = lbls.index('n_s')
    print "\t*** ok:", F[pok,pok], 1./np.sqrt(F[pok,pok])
    print "\t*** ns:", F[pns,pns], 1./np.sqrt(F[pns,pns])
    print "\t*** ok-ns:", F[pns,pok]
    
    # Add Planck prior
    #Fpl = euclid.add_detf_planck_prior(F, lbls, info=False)
    #Fpl = euclid.add_planck_prior(F, lbls, info=False)
    if USE_DETF_PLANCK_PRIOR:
        # DETF Planck prior
        print "*** Using DETF Planck prior ***"
        l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h', 'sigma8']
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
    
    if len(fixed_params) > 0:
        Fpl, lbls = baofisher.combined_fisher_matrix( [Fpl,], expand=[], 
                     names=lbls, exclude=fixed_params )
    
    # Really hopeful H0 prior
    #ph = lbls.index('h')
    #Fpl[ph, ph] += 1./(0.012)**2.
    
    # Get indices of w0, wa
    pw0 = lbls.index('w0'); pwa = lbls.index('wa'); pA = lbls.index('A')
    
    print "-"*50
    print names[k]
    print "-"*50
    
    # Invert matrix
    cov_pl = np.linalg.inv(Fpl)
    
    # Calculate FOM
    fom = baofisher.figure_of_merit(pw0, pwa, None, cov=cov_pl)
    print "%s: FOM = %3.2f, sig(A) = %3.3f" % (names[k], fom, np.sqrt(cov_pl[pA,pA]))
    print "1D sigma(w_0) = %3.4f" % np.sqrt(cov_pl[pw0,pw0])
    print "1D sigma(w_a) = %3.4f" % np.sqrt(cov_pl[pwa,pwa])
    pnl = lbls.index('sigma_NL')
    print "1D sigma(sigma_NL) = %3.4f" % np.sqrt(cov_pl[pnl,pnl])
    
    x = experiments.cosmo['w0']
    y = experiments.cosmo['wa']
    
    # Plot contours for w0, wa; omega_k free
    transp = [1., 0.85]
    w, h, ang, alpha = baofisher.ellipse_for_fisher_params(pw0, pwa, None, Finv=cov_pl)
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                height=alpha[kk]*h, angle=ang, fc=colours[k][kk], 
                ec=colours[k][0], lw=1.5, alpha=transp[kk]) for kk in [1,0]]
    for e in ellipses: ax.add_patch(e)
    
    # Centroid
    ax.plot(x, y, 'kx')
    print "\nDONE\n"


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

P.gcf().legend((l for l in lines), (name for name in labels), prop={'size':'x-large'}, bbox_to_anchor=[0.93, 0.95], frameon=False)

ax.tick_params(axis='both', which='major', labelsize=20, size=8., width=1.5, pad=8.)
xminorLocator = matplotlib.ticker.MultipleLocator(0.1)
yminorLocator = matplotlib.ticker.MultipleLocator(0.5)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)

ax.set_xlabel(r"$w_0$", fontdict={'fontsize':'xx-large'}, labelpad=15.)
ax.set_ylabel(r"$w_a$", fontdict={'fontsize':'xx-large'})


ax.set_xlim((-1.75, -0.25))
ax.set_ylim((-2.1, 2.1))

if MARGINALISE_CURVATURE:
    ax.set_xlim((-1.25, -0.75))
    ax.set_ylim((-0.9, 0.9))
else:
    ax.set_xlim((-1.25, -0.75))
    ax.set_ylim((-0.9, 0.9))


# FIXME
ax.set_xlim((-1.15, -0.85))
ax.set_ylim((-0.5, 0.5))


# Set size and save
P.tight_layout()
P.gcf().set_size_inches(8.,6.)
##P.savefig(fig_name, transparent=True)
#P.savefig("mario-w0wa-SKAMID.pdf", transparent=True)
#P.savefig("ska-w0wa.pdf", transparent=True)
P.show()
