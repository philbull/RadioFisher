#!/usr/bin/python
"""
Plot functions of redshift for RSDs.
"""
import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
import os

cosmo = rf.experiments.cosmo

#red1 = '#fb9a99' # IM Band 1
#red2 = '#e31a1c' # IM Band 2

#fname = 'RedBook-DA.pdf'

#names = ['MID_B1_SKAonly_RedBook_25000', 'MID_B1_RedBook_25000', 'MID_B2_RedBook_5000',]
#labels = ['MID only (Band 1)', 'MID + MeerKAT (Band 1)', 'MID + MeerKAT (Band 2)',]
#colours = [red1, red1, red2]

#linestyle = [[5,3], [], []]
#marker = ['s', '.', '.']
#ms = [6., 15., 15.]

#-------------------------------------------------------------------------------
# Define colours
#-------------------------------------------------------------------------------
red1 = '#fb9a99' # IM Band 1
red2 = '#e31a1c' # IM Band 2

orange1 = '#fdbf6f' # LOW (lower band)
orange2 = '#FFD025' # LOW (upper band)

green1 = '#b2df8a' # WL/Continuum Band 1
green2 = '#33a02c' # WL/Continuum Band 2

blue1 = '#a6cee3' # HI Galaxies Band 1
blue2 = '#1f78b4' # HI Galaxies Band 2

black1 = '#232323' # External Survey 1
black2 = '#707070' # External Survey 2
black3 = '#A9A9A9' # External Survey 3


fname = 'RedBook-DA-v2.pdf'

names = ['MID_B1_MK_RedBook_20000', 'MID_B2_MK_RedBook_5000',
         'gMIDMK_B2_Rebase', 'EuclidRef', 'HETDEXdz03', ]
labels = ['SKA Wide (IM, Band 1)', 'SKA Medium-Deep (IM, Band 2)',
          'SKA Medium-Deep (GS, Band 2)', 'Euclid-like (GS)', 'HETDEX (GS)']
colours = [red1, red2, blue2, black1, black2]

linestyle = [[], [], [], [], [], []]
marker = ['s', 's', '.', '.', '.', '.']
ms = [6., 6., 15., 15., 15., 15.]

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
    
    #zfns = ['A', 'b_HI', 'f', 'H', 'DA', 'aperp', 'apar']
    zfns = ['bs8', 'fs8', 'H', 'DA',]
    excl = ['Tb', 'n_s', 'sigma8', 'omegak', 'omegaDE', 'w0', 'wa', 'h',
            'gamma', 'N_eff', 'pk*', 'f', 'b_HI', 
            'gamma0', 'gamma1', 'eta0', 'eta1', 'A_xi', 'logkmg',
            'sigma8tot', 'sigma_8', 'k*', 'A', 'aperp', 'apar']
    
    # Marginalising over b_1
    F, lbls = rf.combined_fisher_matrix( F_list,
                                         expand=zfns, names=pnames,
                                         exclude=excl )
    print lbls
    cov = np.linalg.inv(F)
    errs = np.sqrt(np.diag(cov))
    
    # Identify functions of z
    pDA = rf.indices_for_param_names(lbls, 'DA*')
    errDA = 1e3 * errs[pDA] / dAc
    
    # Plot errors as fn. of redshift
    P.plot( zc, errDA, color=colours[k], label=labels[k], lw=2.2,
            marker=marker[k], markersize=ms[k], markeredgecolor=colours[k],
            dashes=linestyle[k] )
    

P.tick_params(axis='both', which='major', labelsize=18, width=1.5, size=8., pad=10)
P.tick_params(axis='both', which='minor', labelsize=18, width=1.5, size=5.)

# Set axis limits
P.xlim((-0.001, 3.15))
P.ylim((1e-3, 0.5))
#P.xscale('log')


P.xlabel('$z$', labelpad=7., fontdict={'fontsize':'x-large'})
P.ylabel('$\sigma_{D_A} / D_A$', labelpad=10., fontdict={'fontsize':'x-large'})

# Set tick locations
#P.gca().yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
#P.gca().yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.01))
P.gca().xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.5))
P.yscale('log')
    
#leg = P.legend(prop={'size':14}, loc='upper right', frameon=True, ncol=2)
#leg.get_frame().set_edgecolor('w')
#leg.get_frame().set_alpha(0.1)

# Set size
P.tight_layout()
P.savefig(fname, transparent=True)
P.show()
