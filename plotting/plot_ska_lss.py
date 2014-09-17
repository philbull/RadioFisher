#!/usr/bin/python
"""
Plot functions of redshift.
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

names = ["SKAHI100", "SKAHI73", 'EuclidRef']
labels = ['SKA1 HI gal.', 'SKA2 HI gal.', 'Euclid']
colours = ['#1619A1', '#CC0000', '#FFB928', '#5B9C0A', '#990A9C', '#FFB928', '#CC0000']
linestyle = [[1,0], [1, 0], [1, 0],]
marker = ['o', 'D', 's',]

# Fiducial value and plotting
fig = P.figure()
axes = [fig.add_subplot(121), fig.add_subplot(122)]

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
    #pnames = ['A', 'b_HI', 'Tb', 'sigma_NL', 'sigma8', 'n_s', 'f', 'aperp', 'apar', 
    #         'omegak', 'omegaDE', 'w0', 'wa', 'h', 'gamma']
    #pnames += ["pk%d" % i for i in range(kc.size)]
    #zfns = [0,1,6,7,8]
    #excl = [2,4,5,  9,10,11,12,13,14] # Exclude all cosmo params
    pnames = rf.load_param_names(root+"-fisher-full-0.dat")
    
    # Transform from D_A and H to D_V and F
    F_list_lss = []
    for i in range(Nbins):
        Fnew, pnames_new = rf.transform_to_lss_distances(
                              zc[i], F_list[i], pnames, DA=dAc[i], H=Hc[i], 
                              rescale_da=1e3, rescale_h=1e2)
        F_list_lss.append(Fnew)
    pnames = pnames_new
    F_list = F_list_lss
    
    #zfns = ['A', 'b_HI', 'f', 'DV', 'F']
    zfns = ['A', 'bs8', 'fs8', 'DV', 'F']
    excl = ['Tb', 'sigma8', 'n_s', 'omegak', 'omegaDE', 'w0', 'wa', 'h', 
            'gamma', 'N_eff', 'pk*', 'f', 'b_HI'] #'fs8', 'bs8']
    F, lbls = rf.combined_fisher_matrix( F_list,
                                                expand=zfns, names=pnames,
                                                exclude=excl )
    cov = np.linalg.inv(F)
    errs = np.sqrt(np.diag(cov))
    
    # Identify functions of z
    pDV = rf.indices_for_param_names(lbls, 'DV*')
    pFF = rf.indices_for_param_names(lbls, 'F*')
    pfs8 = rf.indices_for_param_names(lbls, 'fs8*')
    
    DV = ((1.+zc)**2. * dAc**2. * C*zc / Hc)**(1./3.)
    Fz = (1.+zc) * dAc * Hc / C
    fs8 = cosmo['sigma_8']*fc*Dc
    
    indexes = [pfs8, pFF]
    fn_vals = [fs8, Fz]
    
    # Plot errors as fn. of redshift
    for jj in range(len(axes)):
        err = errs[indexes[jj]] / fn_vals[jj]
        line = axes[jj].plot( zc, err, color=colours[k], lw=1.8, marker=marker[k], 
                              label=labels[k] )
        line[0].set_dashes(linestyle[k])
        #axes[jj].set_ylabel(ax_lbls[jj], fontdict={'fontsize':'20'}, labelpad=10.)
    
    #if fn == 'DA': err = 1e3 * errs[pDA] / dAc
    
    # Set custom linestyle
    #line = ax1.plot(zc, err, color=colours[k], lw=1.8, marker='o', label=labels[k])
    

# Subplot labels
ax_lbls = ["$\sigma_{f \sigma_8}/ f\sigma_8$", "$\sigma_F/F$"]
ymax = [0.055, 0.055]

# Move subplots
# pos = [[x0, y0], [x1, y1]]
l0 = 0.15
b0 = 0.1
ww = 0.8
hh = 0.88 / 2.
j = 0
for i in range(len(axes)):
    axes[i].set_position([l0, b0 + hh*i, ww, hh])
    
    axes[i].tick_params(axis='both', which='major', labelsize=14, width=1.5, size=8.)
    axes[i].tick_params(axis='both', which='minor', labelsize=14, width=1.5, size=8.)
    
    if i == 1: axes[i].tick_params(axis='both', which='major', labelbottom='off')
    axes[i].tick_params(axis='both', which='major', labelsize=20.)
    
    # Set axis limits
    axes[i].set_xlim((-0.1, 2.1))
    axes[i].set_ylim((0., ymax[i]))
    
    # Add label to panel
    #P.figtext(l0 + 0.02, b0 + hh*(0.86+i), ax_lbls[i], 
    #          fontdict={'size':'x-large'}) #, bbox=dict(ec='k', fc='none', lw=1.2))
    
    if i == 0: axes[i].set_xlabel('$z$', labelpad=15., fontdict={'fontsize':'xx-large'})
    axes[i].set_ylabel(ax_lbls[i], labelpad=15., fontdict={'fontsize':'xx-large'})
    
    # Set tick locations
    axes[i].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
    axes[i].yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.01))

# Manually add shared x label
#P.figtext(0.5, 0.02, "$z$", fontdict={'size':'xx-large'})

P.legend(prop={'size':'large'}, bbox_to_anchor=[0.68, 0.98], frameon=False)

# Set size
P.gcf().set_size_inches(8.4, 7.8)
P.savefig('ska-gal-rsd.pdf', transparent=True)
P.show()
