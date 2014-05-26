#!/usr/bin/python
"""
Plot dilation distance, D_V(z).
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

names = ['cSKA1MIDfull1', 'cSKA1MIDfull2', 'SKA1SURfull1', 'SKA1SURfull2',
         'SKAHI100', 'SKAHI73', 'EuclidRef', 'LSST']
colours = ['#1619A1', '#1619A1', '#5B9C0A', '#5B9C0A', '#990A9C', '#FFB928', '#CC0000', '#19BCBC']
labels = ['SKA1-MID B1 IM', 'SKA1-MID B2 IM', 'SKA1-SUR B1 IM', 'SKA1-SUR B2 IM',
          'SKA1 HI gal.', 'SKA2 HI gal.', 'Euclid', 'LSST']
linestyle = [[1,0], [8, 4], [1,0], [8, 4], [1,0], [1, 0], [2, 4, 6, 4], [8, 4]]
marker = ['o', 'D', 'o', 'D', 'o', 'D', 'o', 'D']

"""
# FIXME
names = ["SKAHI100_BAOonly", "SKAHI73_BAOonly", 'EuclidRef_BAOonly']
labels = ['SKA1 HI gal.', 'SKA2 HI gal.', 'Euclid']
colours = ['#1619A1', '#CC0000', '#FFB928', '#5B9C0A', '#990A9C', '#FFB928', '#CC0000']
linestyle = [[1,0], [1, 0], [1, 0],]
marker = ['o', 'D', 's',]
"""

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
    #pnames = ['A', 'b_HI', 'Tb', 'sigma_NL', 'sigma8', 'n_s', 'f', 'aperp', 'apar', 
    #         'omegak', 'omegaDE', 'w0', 'wa', 'h', 'gamma']
    #pnames += ["pk%d" % i for i in range(kc.size)]
    #zfns = [0,1,6,7,8]
    #excl = [2,4,5,  9,10,11,12,13,14] # Exclude all cosmo params
    pnames = baofisher.load_param_names(root+"-fisher-full-0.dat")
    
    # Transform from D_A and H to D_V and F
    F_list_lss = []
    for i in range(Nbins):
        Fnew, pnames_new = baofisher.transform_to_lss_distances(
                              zc[i], F_list[i], pnames, DA=dAc[i], H=Hc[i], 
                              rescale_da=1e3, rescale_h=1e2)
        F_list_lss.append(Fnew)
    pnames = pnames_new
    F_list = F_list_lss
    
    #zfns = ['A', 'b_HI', 'f', 'DV', 'F']
    zfns = ['A', 'bs8', 'fs8', 'DV', 'F']
    excl = ['Tb', 'sigma8', 'n_s', 'omegak', 'omegaDE', 'w0', 'wa', 'h', 
            'gamma', 'N_eff', 'pk*', 'f', 'b_HI'] #'fs8', 'bs8']
    F, lbls = baofisher.combined_fisher_matrix( F_list,
                                                expand=zfns, names=pnames,
                                                exclude=excl )
    cov = np.linalg.inv(F)
    errs = np.sqrt(np.diag(cov))
    
    # Identify functions of z
    pDV = baofisher.indices_for_param_names(lbls, 'DV*')
    pFF = baofisher.indices_for_param_names(lbls, 'F*')
    
    DV = ((1.+zc)**2. * dAc**2. * C*zc / Hc)**(1./3.)
    Fz = (1.+zc) * dAc * Hc / C
    
    # Plot errors as fn. of redshift
    err = errs[pDV] / DV
    line = P.plot( zc, err, color=colours[k], lw=1.8, marker=marker[k], 
                          label=labels[k] )
    line[0].set_dashes(linestyle[k])
    

# Subplot labels
P.gca().tick_params(axis='both', which='major', labelsize=14, width=1.5, size=8., pad=7.)
P.gca().tick_params(axis='both', which='minor', labelsize=14, width=1.5, size=8.)

# Set axis limits
P.xlim((-0.05, 2.5))
#P.ylim((0., 0.065))
P.ylim((0., 0.045))

P.xlabel('$z$', labelpad=15., fontdict={'fontsize':'xx-large'})
P.ylabel("$\sigma_{D_V}/D_V$", labelpad=15., fontdict={'fontsize':'xx-large'})
    
## Set tick locations
#ymajorLocator = matplotlib.ticker.MultipleLocator(0.02)
#yminorLocator = matplotlib.ticker.MultipleLocator(0.01)
P.gca().yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.01))
P.gca().yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.005))

#P.legend(loc='upper center', prop={'size':'medium'}, frameon=True, ncol=2)
P.legend(loc='upper right', prop={'size':'large'}, frameon=False, ncol=1)

# Set size
#P.gcf().set_size_inches(8.4, 7.8)
P.tight_layout()
P.savefig('ska-dv.pdf', transparent=True)
#P.savefig('ska-dv-gal-bao-only.pdf', transparent=True)
#P.savefig('ska-dv-gal-bao-only-skaonly.pdf', transparent=True)
P.show()
