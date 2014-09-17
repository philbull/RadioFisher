#!/usr/bin/python
"""
Plot functions of redshift for RSDs.
"""
import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
from units import *
from mpi4py import MPI

import os
import euclid

cosmo = rf.experiments.cosmo

fname = 'ska-rsd-fsigma8.pdf'

names = ['BOSS', 'EuclidRef', 'SKAHI73',] # 'cSKA1MIDfull1', 'cSKA1MIDfull2', 'SKA1SURfull1', 'SKA1SURfull2']
colours = ['#CC0000', 'c', '#1619A1', '#5B9C0A', '#990A9C', '#FFB928'] # DETF/F/M/S
labels = ['BOSS', 'Euclid', 'SKA2 HI gal.', 'SKA1-MID (B1)', 'SKA1-MID (B2)', 'SKA1-SUR (B1)', 'SKA1-SUR (B2)']
linestyle = [[2, 4, 6, 4], [], [], [8, 4], [], [3, 4], []]

names = ['cSKA1MIDfull1', 'cSKA1MIDfull2', 'fSKA1SURfull1', 'fSKA1SURfull2', 'BOSS']
labels = ['SKA1-MID B1', 'SKA1-MID B2', 'SKA1-SUR B1', 'SKA1-SUR B2', 'BOSS']


names = ['SKA1MIDfull1', 'SKA1MIDfull2', 'fSKA1SURfull1', 'fSKA1SURfull2', 
         'gSKAMIDMKB2', 'gSKASURASKAP', 'gSKA2', 'EuclidRef']
labels = ['SKA1-MID B1 (IM)', 'SKA1-MID B2 (IM)', 'SKA1-SUR B1 (IM)', 
          'SKA1-SUR B2 (IM)', 'SKA1-MID (gal.)', 'SKA1-SUR (gal.)', 
          'Full SKA (gal.)', 'Euclid (gal.)']

colours = ['#8082FF', '#1619A1', '#FFB928', '#ff6600', '#95CD6D', '#007A10', '#CC0000', 
           '#000000', '#858585', '#c1c1c1', 'y']
#           '#FF8080',  '#990A9C', '#FFB928',   '#990A9C', '#FFB928', '#CC0000', '#19BCBC']
linestyle = [[], [], [], [], [], [], [], [], []]
marker = ['D', 'D', 'D', 'D', 's', 's', 'o', 'o', 'o']
ms = [6., 6., 6., 6., 6., 6., 5., 5., 5.]

# Fiducial value and plotting
P.subplot(111)

for k in range(len(names)):
    root = "../output/" + names[k]

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
    #zfns = ['A', 'b_HI', 'f', 'H', 'DA', 'aperp', 'apar']
    zfns = ['A', 'bs8', 'fs8', 'H', 'DA', 'aperp', 'apar']
    excl = ['Tb', 'n_s', 'sigma8', 'omegak', 'omegaDE', 'w0', 'wa', 'h',
            'gamma', 'N_eff', 'pk*', 'f', 'b_HI']
    F, lbls = rf.combined_fisher_matrix( F_list,
                                                expand=zfns, names=pnames,
                                                exclude=excl )
    cov = np.linalg.inv(F)
    errs = np.sqrt(np.diag(cov))
    
    #print lbls
    #np.savetxt("fisher_%s_cosmofns.dat"%names[k], F, header=" ".join(lbls))
    
    # Identify functions of z
    pA = rf.indices_for_param_names(lbls, 'A*')
    pDA = rf.indices_for_param_names(lbls, 'DA*')
    pH = rf.indices_for_param_names(lbls, 'H*')
    pfs8 = rf.indices_for_param_names(lbls, 'fs8*')
    
    #fn_vals = [dAc/1e3, 1., Hc/1e2, fc]
    
    print ""
    print "#", names[k]
    print "# z, fsigma8, sigma(fsigma8)"
    for j in range(zc.size):
        print "%4.4f %5.5e %5.5e" % (zc[j], errs[pfs8][j], (cosmo['sigma_8']*fc*Dc)[j])
    
    # Plot errors as fn. of redshift
    err = errs[pfs8] / (cosmo['sigma_8']*fc*Dc)
    line = P.plot( zc, err, color=colours[k], lw=1.8, label=labels[k], 
                   marker=marker[k], markersize=ms[k], markeredgecolor=colours[k] )
    line[0].set_dashes(linestyle[k])
    

P.tick_params(axis='both', which='major', labelsize=20, width=1.5, size=8., pad=5)
P.tick_params(axis='both', which='minor', labelsize=20, width=1.5, size=8.)
    
# Set axis limits
P.xlim((-0.05, 2.2))
P.ylim((0., 0.045))

# Add label to panel
#P.figtext(l0 + 0.02, b0 + hh*(0.86+i), ax_lbls[i], 
#          fontdict={'size':'x-large'}) #, bbox=dict(ec='k', fc='none', lw=1.2))

P.xlabel('$z$', labelpad=10., fontdict={'fontsize':'xx-large'})
P.ylabel('$\sigma(f \sigma_8) / (f \sigma_8)$', labelpad=15., fontdict={'fontsize':'xx-large'})

# Set tick locations
P.gca().yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.01))
P.gca().yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.01))
    
leg = P.legend(prop={'size':'large'}, loc='upper right', frameon=True, ncol=2)
leg.get_frame().set_edgecolor('w')
leg.get_frame().set_alpha(0.8)

# Set size
P.tight_layout()
#P.gcf().set_size_inches(8.4, 7.8)
P.gcf().set_size_inches(9.5, 6.8)
P.savefig(fname, transparent=True)
P.show()
