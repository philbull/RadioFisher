#!/usr/bin/python
"""
Process EOS Fisher matrices and plot P(k).
"""
import numpy as np
import pylab as P
import baofisher
import matplotlib.patches
import matplotlib.cm
from units import *
from mpi4py import MPI
import experiments
import os
import euclid

cosmo = experiments.cosmo

names = ['SKA1MIDfull1', 'SKA1MIDfull2', 'fSKA1SURfull1', 'fSKA1SURfull2', 'EuclidRef']
colours = ['#1619A1', '#1619A1', '#5B9C0A', '#5B9C0A', '#990A9C', '#FFB928', '#CC0000']
labels = ['SKA1-MID B1 IM', 'SKA1-MID B2 IM', 'SKA1-SUR B1 IM', 'SKA1-SUR B2 IM',
          'Euclid']
linestyle = [[], [8, 4], [], [8, 4], [], [], [2, 4, 6, 4], []]
marker = ['o', 'D', 'o', 'D', 'o', 'D', 'o', 'D']

#names = ['EuclidRef', 'cexptLx', 'cexptLy', 'iexptOpt']
#labels = ['Euclid', 'Fac. quadrature', 'Fac. min.', 'MEGA']

#names = ['EuclidRef', 'EuclidRef_BAOonly', 'SKA1SURfull2',] # 'BINGO', 'yCHIME', 'yTIANLAI', 'yTIANLAIband2', 'yTIANLAIpath']
#labels = ['Euclid', 'Euclid_BAO', 'SKA1-SUR Full (B2)', 'BINGO', 'CHIME', 'TIANLAI (550 - 950)', 'TIANLAI (1000 - 1420)', 'TIANLAI Pathfinder']
#colours = ['#1619A1', '#5B9C0A', '#990A9C', '#FFB928', '#CC0000', 'c', 'm']
#linestyle = [[1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0]]

# Get f_bao(k) function
cosmo = baofisher.load_power_spectrum(cosmo, "cache_pk.dat", force_load=True)
fbao = cosmo['fbao']

# Fiducial value and plotting
P.subplot(111)

for k in range(len(names)):
    root = "output/" + names[k]

    # Load cosmo fns.
    dat = np.atleast_2d( np.genfromtxt(root+"-cosmofns-zc.dat") ).T
    zc, Hc, dAc, Dc, fc = dat
    z, H, dA, D, f = np.genfromtxt(root+"-cosmofns-smooth.dat").T
    kc = np.genfromtxt(root+"-fisher-kc.dat").T

    # Load Fisher matrices as fn. of z
    Nbins = zc.size
    F_list = [np.genfromtxt(root+"-fisher-full-%d.dat" % i) for i in range(Nbins)]
    
    # EOS FISHER MATRIX
    # Actually, (aperp, apar) are (D_A, H)
    pnames = baofisher.load_param_names(root+"-fisher-full-0.dat")
    zfns = []; excl = []
    F, lbls = baofisher.combined_fisher_matrix( F_list,
                                                expand=zfns, names=pnames,
                                                exclude=excl )
    
    # Just do the simplest thing for P(k) and get 1/sqrt(F)
    cov = [np.sqrt(1. / np.diag(F)[lbls.index(lbl)]) for lbl in lbls if "pk" in lbl]
    cov = np.array(cov)
    pk = cosmo['pk_nobao'](kc) * (1. + fbao(kc))
    
    # Replace nan/inf values
    cov[np.where(np.isnan(cov))] = 1e10
    cov[np.where(np.isinf(cov))] = 1e10
    
    pw0 = baofisher.indexes_for_sampled_fns(11, zc.size, zfns)
    pwa = baofisher.indexes_for_sampled_fns(12, zc.size, zfns)
    
    print "-"*50
    print names[k]
    #print cov
    print lbls[pw0], 1. / np.sqrt(F[pw0,pw0])
    print lbls[pwa], 1. / np.sqrt(F[pwa,pwa])
    
    """
    if k == 0:
        # Plot shaded region
        P.fill_between(kc, np.ones(kc.size)*1e-10, cov, facecolor='#e1e1e1', edgecolor='none')
    else:
        # Plot errorbars
        P.plot(kc, cov, color=colours[k], label=labels[k], lw=2.2, ls=linestyle[k])
    """
    line = P.plot(kc, cov, color=colours[k], label=labels[k], lw=2.4)
    
    # Set custom linestyle    
    line[0].set_dashes(linestyle[k])


P.xscale('log')
P.yscale('log')
P.xlim((1.5e-3, 3e0))
P.ylim((9e-4, 1e1))
P.legend(loc='lower left', prop={'size':'large'}, frameon=False)

P.tick_params(axis='both', which='major', labelsize=20, size=8., width=1.5, pad=8.)
P.tick_params(axis='both', which='minor', labelsize=20, size=5., width=1.5)

P.xlabel(r"$k \,[\mathrm{Mpc}^{-1}]$", fontdict={'fontsize':'xx-large'})
P.ylabel(r"$\Delta P / P$", fontdict={'fontsize':'xx-large'})

P.tight_layout()
# Set size
P.gcf().set_size_inches(8.,6.)
#P.savefig('ska-dlogp.pdf', transparent=True) # 100

P.show()
