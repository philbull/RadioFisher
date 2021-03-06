#!/usr/bin/python
"""
Plot P(k) as a function of redshift (Fig. 7).
"""
import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
import os
from radiofisher import euclid

cosmo = rf.experiments.cosmo

names = ['EuclidRef_paper', 'exptL_paper', 'aexptM_paper', 'exptS_paper']
colours = ['#CC0000', '#1619A1', '#5B9C0A', '#990A9C'] # DETF/F/M/S
labels = ['DETF IV', 'Facility', 'Mature', 'Snapshot']
linestyle = [[2, 4, 6, 4], [1,0], [8, 4], [3, 4]]


#names = ['SKA1MIDfull2', 'fSKA1SURfull2', 'fSKA1LOW'] #, 'fSKA1SURfull2',]
#colours = ['#1619A1', '#5B9C0A', '#990A9C', '#FFB928', '#CC0000']
#labels = ['SKA1-MID B2 SD', 'SKA1-SUR B2 PAF', 'SKA1-LOW']


# Get f_bao(k) function
cosmo = rf.load_power_spectrum(cosmo, "cache_pk.dat", force_load=True)
fbao = cosmo['fbao']

# Fiducial value and plotting
P.subplot(111)

for k in [1,]: #range(len(names)):
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
    ppk = rf.indices_for_param_names(pnames, 'pk*')
    
    cmap = matplotlib.cm.Blues_r
    for j in range(len(F_list))[::-1]:
        F = F_list[j]
        
        # Just do the simplest thing for P(k) and get 1/sqrt(F)
        cov = np.sqrt(1. / np.diag(F)[ppk])
        pk = cosmo['pk_nobao'](kc) * (1. + fbao(kc))
        
        # Replace nan/inf values
        cov[np.where(np.isnan(cov))] = 1e10
        cov[np.where(np.isinf(cov))] = 1e10
        
        # Line with fading colour
        col = cmap(0.8*j / len(F_list))
        line = P.plot(kc, cov, color=col, lw=2., alpha=1.)
        
        # Label for min/max redshifts
        N = kc.size
        if j == 0:
            P.annotate("z = %3.2f" % zc[j], xy=(kc[43], cov[43]), 
                       xytext=(50., -45.), fontsize='large', 
                       textcoords='offset points', ha='center', va='center', 
                       arrowprops={'width':1.8, 'color':'#1619A1', 'shrink':0.05} )
        if j == len(F_list) - 1:
            P.annotate("z = %3.2f" % zc[j], xy=(kc[35], cov[35]), 
                       xytext=(-65., 60.), fontsize='large', 
                       textcoords='offset points', ha='center', va='center',
                       arrowprops={'width':1.8, 'color':'#1619A1', 'shrink':0.07} )
    
    # Plot the summed constraint (over all z)
    F, lbls = rf.combined_fisher_matrix(F_list, expand=[], names=pnames, exclude=[])
    cov = np.sqrt(1. / np.diag(F)[ppk])
    pk = cosmo['pk_nobao'](kc) * (1. + fbao(kc))
    
    # Replace nan/inf values
    cov[np.where(np.isnan(cov))] = 1e10
    cov[np.where(np.isinf(cov))] = 1e10
    
    # Line with fading colour
    line = P.plot(kc, cov, color='k', lw=3.)
    
    # Set custom linestyle
    #line[0].set_dashes(linestyle[k])


P.xscale('log')
P.yscale('log')
P.xlim((2e-3, 7e-1))
P.ylim((2e-3, 1e1))
#P.legend(loc='lower left', prop={'size':'large'})

P.tick_params(axis='both', which='major', labelsize=20, size=8., width=1.5, pad=8.)
P.tick_params(axis='both', which='minor', labelsize=20, size=5., width=1.2)
P.xlabel(r"$k \,[\mathrm{Mpc}^{-1}]$", fontdict={'fontsize':'xx-large'}, labelpad=10.)
P.ylabel(r"$\Delta P / P$", fontdict={'fontsize':'xx-large'}, labelpad=10.)

P.tight_layout()
# Set size
P.gcf().set_size_inches(8.,6.)
P.savefig('fig07-dlogp-fnz.pdf', transparent=True) # 100
#P.savefig("pkz_%s.pdf" % names[k], transparent=True)

P.show()
