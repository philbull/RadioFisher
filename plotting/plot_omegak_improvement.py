#!/usr/bin/python
"""
Plot improvement in Omega_K or gamma as a function of z.
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

nsig = 1.5 # No. of sigma to plot out to
aspect = 1. #1.7 # Aspect ratio of range (w = aspect * h)

#TYPE = 'omegak'
TYPE = 'gamma'
#TYPE = 'FOM'

if TYPE == 'omegak':
    param1 = "omegak"
    label1 = "\Omega_K"
    fid1 = 0.
    fig_name = "pub-ok-improvement.pdf"
elif TYPE == 'gamma':
    param1 = "gamma"
    label1 = "\gamma"
    fid1 = cosmo['gamma']
    fig_name = "pub-gamma-improvement.pdf"
else:
    param1 = "fom"
    fig_name = "pub-fom-improvement.pdf"

USE_DETF_PLANCK_PRIOR = True
MARGINALISE_GAMMA = True # Marginalise over gamma
MARGINALISE_CURVATURE = True # Marginalise over Omega_K
MARGINALISE_INITIAL_PK = True # Marginalise over n_s, sigma_8
MARGINALISE_OMEGAB = True # Marginalise over Omega_baryons
MARGINALISE_W0WA = True # Marginalise over (w0, wa)

names = ['EuclidRef', 'cexptL', 'iexptM'] #, 'exptS']
labels = ['DETF IV', 'Facility', 'Stage II'] #, 'Stage I']
colours = ['#CC0000', '#1619A1', '#5B9C0A', '#FFB928']
linestyle = [[2, 4, 6, 4], [1,0], [8, 4], [3, 4]]

# Fiducial value and plotting
fig = P.figure()
ax = fig.add_subplot(111)

_k = range(len(names))
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
    
    # Add each redshift bin in turn, and recalculate FOM.
    zmax = []; fom = []
    #for l in range(1, len(F_list)):
    for l in range(1, len(F_list)+1):
        
        # EOS FISHER MATRIX
        # Actually, (aperp, apar) are (D_A, H)
        pnames = rf.load_param_names(root+"-fisher-full-0.dat")
        zfns = ['b_HI',]
        excl = ['Tb', 'f', 'H', 'DA', 'apar', 'aperp', 'pk*', 'N_eff', 'fs8', 'bs8']
        F, lbls = rf.combined_fisher_matrix( F_list[:l],
                                                    expand=zfns, names=pnames,
                                                    exclude=excl )
        # DETF Planck prior
        print "*** Using DETF Planck prior ***"
        l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h', 'sigma8']
        F_detf = euclid.detf_to_rf("DETF_PLANCK_FISHER.txt", cosmo, omegab=False)
        Fpl, lbls = rf.add_fisher_matrices(F, F_detf, lbls, l2, expand=True)
        
        # Decide whether to fix various parameters
        fixed_params = []
        if not MARGINALISE_GAMMA: fixed_params += ['gamma',]
        if not MARGINALISE_CURVATURE: fixed_params += ['omegak',]
        if not MARGINALISE_INITIAL_PK: fixed_params += ['n_s', 'sigma8']
        if not MARGINALISE_OMEGAB: fixed_params += ['omega_b',]
        if not MARGINALISE_W0WA: fixed_params += ['w0', 'wa']
        
        if len(fixed_params) > 0:
            Fpl, lbls = rf.combined_fisher_matrix( [Fpl,], expand=[], 
                         names=lbls, exclude=fixed_params )
        
        # Really hopeful H0 prior
        #ph = lbls.index('h')
        #Fpl[ph, ph] += 1./(0.012)**2.
        
        # Invert matrix
        cov_pl = np.linalg.inv(Fpl)
        
        # Plot improvement in chosen parameter
        if param1 == 'fom':
            p1 = lbls.index('w0'); p2 = lbls.index('wa')
            _fom = rf.figure_of_merit(p1, p2, None, cov=cov_pl)
        else:
            pp = lbls.index(param1)
            _fom = 1. / np.sqrt(cov_pl[pp, pp])
        zmax.append(zc[:l][-1])
        fom.append(_fom)
    
    # Plot curve for this experiment
    line = ax.plot(zmax, fom, color=colours[k], ls='solid', lw=1.8, 
                   marker='o', label=labels[k])
    line[0].set_dashes(linestyle[k])


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

# Axis ticks and labels
ax.legend(prop={'size':'x-large'}, bbox_to_anchor=[0.96, 0.30], frameon=False)
ax.tick_params(axis='both', which='major', labelsize=20, size=8., width=1.5, pad=8.)
ax.set_xlim((0.25, 2.55))
ax.set_xlabel("$z_\mathrm{max}$", fontdict={'fontsize':'xx-large'}, labelpad=15.)
if param1 == 'fom':
    ax.set_ylabel("$\mathrm{FOM}$", fontdict={'fontsize':'xx-large'}, labelpad=15.)
    ax.set_ylim((-0.5, 460.))
elif param1=='omegak':
    ax.set_ylabel("$[\sigma({%s})]^{-1}$" % label1, fontdict={'fontsize':'xx-large'}, 
                  labelpad=15.)
    ax.set_ylim((-10., 1200.))
else:
    ax.set_ylabel("$[\sigma({%s})]^{-1}$" % label1, fontdict={'fontsize':'xx-large'}, 
                  labelpad=15.)
    ax.set_ylim((-0.5, 50.))

# Set size and save
P.tight_layout()
P.gcf().set_size_inches(8.,6.)
P.savefig(fig_name, transparent=True)
P.show()
