#!/usr/bin/python
"""
Process EOS Fisher matrices and plot P(k).
"""
import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
import os

cosmo = rf.experiments.cosmo

names = ['EuclidRef_paper', 'exptL_paper', 'aexptM_paper', 'yCHIME_paper'] #'exptS_paper']
colours = ['#CC0000', '#1619A1', '#5B9C0A', '#990A9C', 'c', 'm'] # DETF/F/M/S
labels = ['DETF IV', 'Facility', 'Stage II', 'Stage I']
linestyle = [[2, 4, 6, 4], [], [8, 4], [3, 4], [], [], [], []]


names = ['yCHIME_paper', 'yCHIME_nocut_paper', 'yCHIME_avglow_paper', 'EuclidRef_paper']
labels = names

"""
#names = ['EuclidRef', 'cexptLx', 'cexptLy', 'iexptOpt']
#labels = ['Euclid', 'Fac. quadrature', 'Fac. min.', 'MEGA']

#names = ['yCHIME', 'yCHIME_nocut']
#labels = ['CHIME', 'CHIME nocut']

names = ['testSKA1SURfull1', 'ftestSKA1SURfull1_fixedfov']
labels = ['SKA1-SUR old', 'SKA1-SUR fixed FOV']

names = ['fSKA1SURfull1', 'fSKA1SURfull2', 'SKA1SURfull2', 'SKA1MIDfull1', 'SKA1MIDfull2', 'BOSS']
labels = ['SKA1-SUR Full B1', 'SKA1-SUR Full B2 Fixed FOV', 'SKA1-SUR Full B2', 'SKA1-MID Full B1', 'SKA1-MID Full B2', 'BOSS']

names = ['FAST', 'FAST4yr', 'fSKA1SURfull1', 'EuclidRef', 'yCHIME']
labels = ['FAST 10k hrs', 'FAST 4yr', 'SKA1-SUR Full B1', 'Euclid', 'CHIME']
linestyle = [[], [], [], [], [], [], []]

names = ['SKA1MID350XXX_25000', 'fSKA1SUR350XXX_25000']
labels = ['SKA1MID350', 'fSKA1SUR350']
"""

# Get f_bao(k) function
cosmo = rf.load_power_spectrum(cosmo, "cache_pk.dat", force_load=True)
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
    pnames = rf.load_param_names(root+"-fisher-full-0.dat")
    zfns = []; excl = []
    F, lbls = rf.combined_fisher_matrix( F_list, expand=zfns, names=pnames,
                                         exclude=excl )
    
    # Just do the simplest thing for P(k) and get 1/sqrt(F)
    cov = [np.sqrt(1. / np.diag(F)[lbls.index(lbl)]) for lbl in lbls if "pk" in lbl]
    cov = np.array(cov)
    pk = cosmo['pk_nobao'](kc) * (1. + fbao(kc))
    
    # Replace nan/inf values
    cov[np.where(np.isnan(cov))] = 1e10
    cov[np.where(np.isinf(cov))] = 1e10
    
    pw0 = rf.indexes_for_sampled_fns(11, zc.size, zfns)
    pwa = rf.indexes_for_sampled_fns(12, zc.size, zfns)
    
    #for jj in range(kc.size):
    #    print "%5.5e %5.5e" % (kc[jj], cov[jj])
    
    print "-"*50
    print names[k]
    print lbls[pw0], 1. / np.sqrt(F[pw0,pw0])
    print lbls[pwa], 1. / np.sqrt(F[pwa,pwa])
    
    """
    # Output dP/P
    print "-"*50
    print names[k]
    for jj in range(zc.size):
        if zc[jj] == 1.:
            cov = [np.sqrt(1. / np.diag(F_list[jj])[lbls.index(lbl)]) for lbl in lbls if "pk" in lbl]
            for _k in range(kc.size):
                print "%5.5e %5.5e" % (kc[_k], cov[_k])
    """
    """
    if k == 0:
        # Plot shaded region
        P.fill_between(kc, np.ones(kc.size)*1e-10, cov, facecolor='#e1e1e1', edgecolor='none')
    else:
        # Plot errorbars
        P.plot(kc, cov, color=colours[k], label=labels[k], lw=2.2, ls=linestyle[k])
    """
    line = P.plot(kc, cov, color=colours[k], label=labels[k], lw=2.4, marker='None')
    #line = P.plot(kc/0.7, cov, color=colours[k], label=labels[k], lw=1.8, marker='.')#lw=2.4)
    
    # Set custom linestyle
    print linestyle[k]
    line[0].set_dashes(linestyle[k])

#P.axhline(1., ls='dashed', color='k', lw=1.5)
#P.axvline(1e-2, ls='dashed', color='k', lw=1.5)

P.xscale('log')
P.yscale('log')
P.xlim((1.3e-3, 1.2e0))
P.ylim((9e-4, 1e1))
P.legend(loc='lower left', prop={'size':'large'}, frameon=False)

P.tick_params(axis='both', which='major', labelsize=20, size=8., width=1.5, pad=8.)
P.tick_params(axis='both', which='minor', labelsize=20, size=5., width=1.5)

P.xlabel(r"$k \,[\mathrm{Mpc}^{-1}]$", fontdict={'fontsize':'xx-large'})
P.ylabel(r"$\Delta P / P$", fontdict={'fontsize':'xx-large'})

P.tight_layout()
# Set size
P.gcf().set_size_inches(8.,6.)
#P.savefig("fig04-dlogp.pdf", transparent=True) # 100

P.show()
