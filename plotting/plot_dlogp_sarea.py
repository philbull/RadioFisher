#!/usr/bin/python
"""
Plot dP/P as a function of redshift for a given Sarea.
"""
import numpy as np
import pylab as P
import baofisher
import matplotlib.patches
import matplotlib.cm
from units import *
import experiments
import os, sys

try:
    sarea = int(sys.argv[1])
except:
    print "Error: Need to specify S_area, in whole degrees: int(sarea)"
    sys.exit(1)

cosmo = experiments.cosmo

#names = ['SKA1MIDfull1', 'iSKA1MIDfull1', 'fSKA1SURfull1'] #, 'fSKA1SURfull2',]
#colours = ['#1619A1', '#5B9C0A', '#990A9C', '#FFB928', '#CC0000']
#labels = ['SKA1-MID B1 SD', 'SKA1-MID B1 Int.', 'SKA1-SUR B1 PAF']

names = ['SKA0MID', 'SKA0SUR', 'SKA1MID900', 'SKA1MID350', 'iSKA1MID900', 
         'iSKA1MID350', 'fSKA1SUR650', 'fSKA1SUR350']
colours = ['#1619A1', '#5B9C0A', '#990A9C', '#FFB928', '#CC0000', 'c', '#ff6600', 'k', 'y']
labels = names

# Get f_bao(k) function
#cosmo = baofisher.load_power_spectrum(cosmo, "cache_pk.dat", force_load=True)
#fbao = cosmo['fbao']

# Fiducial value and plotting
P.subplot(111)

for k in [2,]: #range(len(names)):
    root = "output/%s_nofg_%d" % (names[k], sarea)

    # Load cosmo fns.
    dat = np.atleast_2d( np.genfromtxt(root+"-cosmofns-zc.dat") ).T
    zc, Hc, dAc, Dc, fc = dat
    z, H, dA, D, f = np.genfromtxt(root+"-cosmofns-smooth.dat").T
    kc = np.genfromtxt(root+"-fisher-kc.dat").T

    # Load Fisher matrices as fn. of z
    try:
        Nbins = zc.size
        F_list = [np.genfromtxt(root+"-fisher-full-%d.dat" % i) for i in range(Nbins)]
    except:
        print "ERROR: Couldn't find", root+"-fisher-full-??.dat"
        sys.exit()
    
    # EOS FISHER MATRIX
    # Actually, (aperp, apar) are (D_A, H)
    pnames = baofisher.load_param_names(root+"-fisher-full-0.dat")
    ppk = baofisher.indices_for_param_names(pnames, 'pk*')
    
    cmap = matplotlib.cm.Blues_r
    for j in range(len(F_list)):
        F = F_list[j]
        
        # Just do the simplest thing for P(k) and get 1/sqrt(F)
        cov = np.sqrt(1. / np.diag(F)[ppk])
        #pk = cosmo['pk_nobao'](kc) * (1. + fbao(kc))
        
        # Replace nan/inf values
        cov[np.where(np.isnan(cov))] = 1e10
        cov[np.where(np.isinf(cov))] = 1e10
        
        # Line with fading colour
        col = cmap(0.8*j / len(F_list))
        line = P.plot(kc, cov, color=col, lw=2., alpha=1., label="z=%3.3f"%zc[j])
    
    # Plot the summed constraint (over all z)
    F, lbls = baofisher.combined_fisher_matrix(F_list, expand=[], names=pnames, exclude=[])
    cov = np.sqrt(1. / np.diag(F)[ppk])
    #pk = cosmo['pk_nobao'](kc) * (1. + fbao(kc))
    
    # Replace nan/inf values
    cov[np.where(np.isnan(cov))] = 1e10
    cov[np.where(np.isinf(cov))] = 1e10
    
    # Bold line
    line = P.plot(kc, cov, color='k', lw=3.)
    
    # Set custom linestyle
    #line[0].set_dashes(linestyle[k])


P.xscale('log')
P.yscale('log')
P.xlim((2e-3, 1e1))
P.ylim((3e-3, 1e1))
P.legend(loc='lower left', prop={'size':'x-small'}, frameon=False, ncol=2)

P.title("%s, %d deg^2" % (names[k], sarea))

P.tick_params(axis='both', which='major', labelsize=20, size=8., width=1.5, pad=8.)
P.tick_params(axis='both', which='minor', labelsize=20, size=5., width=1.2)
P.xlabel(r"$k \,[\mathrm{Mpc}^{-1}]$", fontdict={'fontsize':'xx-large'}, labelpad=10.)
P.ylabel(r"$\Delta P / P$", fontdict={'fontsize':'xx-large'}, labelpad=10.)

P.tight_layout()
P.gcf().set_size_inches(8.,6.)
P.savefig("pkz_%s_%05d.png" % (names[k], sarea), transparent=False)
print "output: pkz_%s_%05d.png" % (names[k], sarea)
#P.show()
