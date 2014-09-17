#!/usr/bin/python
"""
Plot improvement in w0, wa as a function of z (nested ellipses).
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

print "*** OBSOLETE ***"
exit()

fig_name = "pub-w0wa-improvement.pdf"
#fig_name = "pub-ok-improvement.pdf"
#fig_name = "pub-ok-improvement-reverse.pdf"
nsig = 1.5 # No. of sigma to plot out to
aspect = 1. #1.7 # Aspect ratio of range (w = aspect * h)

param1 = "w0"
label1 = "$w_0$"
fid1 = cosmo['w0']

param2 = "wa"
label2 = "$w_a$"
fid2 = cosmo['wa']

USE_DETF_PLANCK_PRIOR = True
MARGINALISE_CURVATURE = True # Marginalise over Omega_K
MARGINALISE_INITIAL_PK = True # Marginalise over n_s, sigma_8
MARGINALISE_OMEGAB = True # Marginalise over Omega_baryons

names = ['EuclidRef', 'cexptL', 'iexptM'] #, 'exptS']
labels = ['DETF IV', 'Facility', 'Mature'] #, 'Snapshot']
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
    for l in range(1, len(F_list)):
        
        # EOS FISHER MATRIX
        # Actually, (aperp, apar) are (D_A, H)
        pnames = ['A', 'b_HI', 'Tb', 'sigma_NL', 'sigma8', 'n_s', 'f', 'aperp', 'apar', 
                 'omegak', 'omegaDE', 'w0', 'wa', 'h', 'gamma']
        pnames += ["pk%d" % i for i in range(kc.size)]
        
        zfns = [1,]
        excl = [2,  6,7,8,  14] # omega_k free 4,5
        excl  += [i for i in range(len(pnames)) if "pk" in pnames[i]]
        
        F, lbls = rf.combined_fisher_matrix( F_list[:l],
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
        
        # Invert matrix
        cov_pl = np.linalg.inv(Fpl)
        
        # Calculate FOM
        _fom = rf.figure_of_merit(p1, p2, None, cov=cov_pl)
        print "%s: FOM = %3.2f, Nbins = %d" % (zc[l], _fom, len(F_list[:l]))
        
        # FIXME
        # Plot improvement in Omega_K, rather than FOM
        #pok = lbls.index('omegak')
        #_fom = 1. / np.sqrt(cov_pl[pok, pok])
        
        zmax.append(zc[:l][-1])
        fom.append(_fom)
    
    # Plot curve for this experiment
    line = ax.plot(zmax, fom, color=colours[k], ls='solid', lw=1.8, marker='o', label=labels[k])
    line[0].set_dashes(linestyle[k])


# Report on what options were used
print "-"*50
s1 = "Marginalised over Omega_K" if MARGINALISE_CURVATURE else "Fixed Omega_K"
s2 = "Marginalised over ns, sigma8" if MARGINALISE_INITIAL_PK else "Fixed ns, sigma8"
s3 = "Marginalised over Omega_b" if MARGINALISE_OMEGAB else "Fixed Omega_b"
print "NOTE:", s1
print "NOTE:", s2
print "NOTE:", s3

fontsize = 20
for tick in ax.xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in ax.yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)

ax.set_xlabel("$z_\mathrm{max}$", fontdict={'fontsize':'xx-large'}, labelpad=15.)
#ax.set_ylabel("$(\sigma_K)^{-1}$", fontdict={'fontsize':'xx-large'}, labelpad=15.)
ax.set_ylabel("FOM", fontdict={'fontsize':'xx-large'}, labelpad=15.)

ax.legend(loc='upper left', prop={'size':'x-large'})

ax.set_xlim((0.25, 2.55))
#ax.set_ylim((y-dy, y+dy))

# Set size and save
P.tight_layout()
P.gcf().set_size_inches(8.,6.)
P.savefig(fig_name, transparent=True)
P.show()
