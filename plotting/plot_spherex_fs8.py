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

fname = 'spherex-fsig8.pdf'
#fname = 'mg-fsigma8-scaledep.pdf'

#names = [ 'gSPHEREx1_mg', 'gSPHEREx2_mg', 'EuclidRef_mg' ]
names = [ 'gSPHEREx1_mgphotoz', 'gSPHEREx2_mgphotoz', 'BOSS_mg', 'EuclidRef_mg' ]
#         'gSPHEREx3_mg', 'gSPHEREx4_mg', 'gSPHEREx5_mg', 'EuclidRef_mg', 'BOSS_mg', ]
labels = [ 'SPHEREx 0.003', 'SPHEREx 0.008', 'BOSS spectro-z', 'Euclid spectro-z', 'DESI spectro-z']
#          'SPHEREx 0.025', 'SPHEREx 0.07', 'SPHEREx 0.16' ]
colours = [ '#80B6D6', '#93C993', '#c8c8c8', '#757575', '#a8a8a8']
#colours = [ '#80B6D6', '#93C993', '#AB9C6D', '#EE5E3E', '#FDBF6F', '#c8c8c8',]

linestyle = [[], [], [], [], [], [], [], [], [], [], [],]
marker = ['o', 'o', 'o', 'D', 'D', 'D']
ms = [8., 8., 8., 7., 7., 7.]

# Plotting
P.subplot(111)

for k in range(len(names)):
    root = "output/" + names[k]
    print root

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
            'gamma', 'N_eff', 'pk*', 'f', 'b_HI', 
            'gamma0', 'gamma1', 'eta0', 'eta1', 'A_xi', 'logkmg',
            'sigma8tot', 'sigma_8', 'k*', 'b1']
    
    F2, lbls2 = rf.combined_fisher_matrix( F_list,
                                         expand=zfns, names=pnames,
                                         exclude=excl )
    cov2 = np.linalg.inv(F2)
    errs2 = np.sqrt(np.diag(cov2))
    #for d, l in zip(np.diag(F2), lbls2):
    #    print "%10s %3.1e" % (l, d)
    
    # Identify functions of z
    pfs8_2 = rf.indices_for_param_names(lbls2, 'fs8*')
    
    # Plot errors as fn. of redshift (b_1 fixed)
    err2 = errs2[pfs8_2] / (cosmo['sigma_8']*fc*Dc)
    
    print err2
    
    # Remove some Euclid bins
    if labels[k] is not None and 'Euclid' in labels[k]:
        idxs = np.where(np.logical_and(zc >= 1.1, zc <= 1.9))
        zc = zc[idxs]
        err2 = err2[idxs]
    
    if labels[k] is not None:
        P.plot( zc, err2, color=colours[k], label=labels[k], lw=2.8,
                marker=marker[k], markersize=ms[k], markeredgecolor=colours[k] )
    else:
        P.plot( zc, err2, color=colours[k], label=labels[k], lw=2.8,
                marker=marker[k], markersize=ms[k], markeredgecolor=colours[k],
                dashes=[4,3] )


# Plot DESI
# DESI fractional errors of f.sigma_8, taken directly from Table 2.3 of 
# http://desi.lbl.gov/wp-content/uploads/2014/04/tdr-science-biblatex2.pdf
desi_zc = [0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 
           1.55, 1.65, 1.75, 1.85]
#desi_fsig8 = [3.31, 2.1, 2.12, 2.09, 2.23, 2.25, 2.25, 2.9, 3.06, 
#              3.53, 5.1, 8.91, 9.25] # kmax=0.1 Mpc/h
desi_fsig8 = [1.57, 1.01, 1.01, 0.99, 1.11, 1.14, 1.16, 1.73, 1.87, 
              2.27, 3.61, 6.81, 7.07] # kmax=0.2 Mpc/h

P.plot( desi_zc, np.array(desi_fsig8)/100., color=colours[-1], 
        label='DESI spectro-z', lw=2.8, marker='o', markersize=7.,
        markeredgecolor=colours[-1] )

# Numbers for sigma(f)/f from Table 1.4 of Euclid theory paper (1206.1225v2)
#euclid_zc = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
#euclid_fsig8 = [0.0105, 0.0103, 0.01136, 0.0112, 0.0121, 0.0132, 0.0152,
#                0.0151, 0.0183]
#P.plot( euclid_zc, euclid_fsig8, color=colours[-1], 
#        label='Euclid upd spectro-z', lw=2.8, marker='o', markersize=7.,
#        markeredgecolor='r' )


# Load actual f.sigma_8 data and plot it
dat = np.genfromtxt("fsigma8_data.dat").T
#P.plot(dat[1], dat[3], 'kD')
P.plot(dat[1], dat[3]/dat[2], 'ko', mfc='none', mew=1.8, ms=8.)

P.tick_params(axis='both', which='major', labelsize=20, width=1.5, size=8., pad=10)
P.tick_params(axis='both', which='minor', labelsize=20, width=1.5, size=5.)

# Set axis limits and labels
P.xlim((-0.001, 2.1))
P.ylim((3e-3, 1e0))

P.xlabel('$z$', labelpad=10., fontdict={'fontsize':'xx-large'})
P.ylabel('$\sigma(f \sigma_8) / (f \sigma_8)$', labelpad=15., fontdict={'fontsize':'xx-large'})

# Set tick locations
P.gca().xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
P.gca().xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
P.yscale('log')
    
#leg = P.legend(prop={'size':16}, loc='upper right', frameon=True, ncol=1)
#leg.get_frame().set_edgecolor('w')
#leg.get_frame().set_alpha(0.1)

# Custom legend order
order = (0,1,2,4,3) #3,5,4)
labels = [labels[k] for k in order]
lines = [ matplotlib.lines.Line2D([0.,], [0.,], lw=4., color=colours[k]) 
          for k in order ]
leg = P.legend((l for l in lines), (name for name in labels), prop={'size':17}, frameon=False, ncol=2, loc='upper center')
leg.get_frame().set_edgecolor('w')
leg.get_frame().set_alpha(0.1)

# Set size
P.tight_layout()
#P.gcf().set_size_inches(9.5, 6.8)
P.savefig(fname, transparent=True)
P.show()
