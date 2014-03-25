#!/usr/bin/python
"""
Plot functions of redshift.
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

names = ['cexptL_bao', 'cexptL_bao_rsd', 'cexptL_bao_pkshift', 'cexptL_bao_vol', 'cexptL_bao_all']
labels = ['BAO only', 'BAO + RSD', 'BAO + $P(k)$ shift', 'BAO + Volume', 'All']
colours = ['#CC0000', '#1619A1', '#5B9C0A', '#FFB928', 'k']
linestyle = [[1,0], [8,4], [6,4,3,4], [3,4], [1,0]]

cosmo_fns = baofisher.background_evolution_splines(cosmo)

# Fiducial value and plotting
fig = P.figure()
axes = [fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)]

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
    zfns = ['A', 'bs8', 'fs8', 'H', 'DA', 'aperp', 'apar']
    excl = ['Tb', 'n_s', 'sigma8', 'omegak', 'omegaDE', 'w0', 'wa', 'h',
            'gamma', 'N_eff', 'pk*', 'f', 'b_HI']
    F, lbls = baofisher.combined_fisher_matrix( F_list,
                                                expand=zfns, names=pnames,
                                                exclude=excl )
    cov = np.linalg.inv(F)
    errs = np.sqrt(np.diag(cov))
    
    # Identify functions of z
    # Identify functions of z
    pA = baofisher.indices_for_param_names(lbls, 'A*')
    pDA = baofisher.indices_for_param_names(lbls, 'DA*')
    pH = baofisher.indices_for_param_names(lbls, 'H*')
    pf = baofisher.indices_for_param_names(lbls, 'fs8*')
    #pf = baofisher.indices_for_param_names(lbls, 'f*')
    
    indexes = [pf, pDA, pH]
    fn_vals = [cosmo['sigma_8']*fc*Dc, dAc/1e3, Hc/1e2]
    
    # Plot errors as fn. of redshift
    for jj in range(len(axes)):
        err = errs[indexes[jj]] / fn_vals[jj]
        line = axes[jj].plot( zc, err, color=colours[k], lw=1.8, label=labels[k] )
        line[0].set_dashes(linestyle[k])
        axes[jj].plot( zc, err, color=colours[k], marker='o', ls='none')
        #axes[jj].set_ylabel(ax_lbls[jj], fontdict={'fontsize':'20'}, labelpad=10.)


# Subplot labels
ax_lbls = ["$\sigma_{f \sigma_8}/f\sigma_8$", "$\sigma_{D_A}/D_A$", "$\sigma_H/H$"]
ymax = [0.159, 0.159, 0.159]

# Move subplots
# pos = [[x0, y0], [x1, y1]]
l0 = 0.1
b0 = 0.1
ww = 0.8
hh = 0.85 / 3.
j = 0

for i in range(len(axes)):
    axes[i].set_position([l0, b0 + hh*i, ww, hh])
    
    axes[i].tick_params(axis='both', which='major', labelsize=14, size=4., width=1.5)
    
    # Set axis limits
    axes[i].set_xlim((0.25, 2.2))
    axes[i].set_ylim((0., ymax[i]))
    #axes[i].set_ylabel(ax_lbls[i], fontdict={'fontsize':'xx-large'}, labelpad=15.)
    
    # Add label to panel
    P.figtext(l0 + 0.02, b0 + hh*(0.86+i), ax_lbls[i], 
              fontdict={'size':'x-large'}) #, bbox=dict(ec='k', fc='none', lw=1.2))
    
    # Remove labels on the bottom
    if i == 0:
        axes[i].tick_params(axis='both', which='both', labelbottom='on')
        axes[i].set_xlabel("$z$", fontdict={'fontsize':'xx-large'}, labelpad=15.)
    else:
        axes[i].tick_params(axis='both', which='both', labelbottom='off')
    
    j = 0
    for l in axes[i].yaxis.get_major_ticks():
        if j % 2 == 1: l.label1.set_visible(False)
        j += 1

# Manually add shared x label
#P.figtext(0.5, 0.02, "$z$", fontdict={'size':'xx-large'})

P.legend(loc='upper right', ncol=2, prop={'size':'medium'})

# Set size
P.gcf().set_size_inches(8., 10.)
P.savefig('pub-zfns-distance-measures.pdf', transparent=True)
P.show()
