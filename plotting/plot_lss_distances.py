#!/usr/bin/python
"""
Plot BAO/AP functions D_V(z) and F(z) as fns. of redshift (Fig. 9).
"""
import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
import os
from radiofisher import euclid
from radiofisher.units import *

cosmo = rf.experiments.cosmo

names = ['EuclidRef_paper', 'exptL_paper', 'aexptM_paper', 'exptS_paper']
colours = ['#CC0000', '#1619A1', '#5B9C0A', '#990A9C'] # DETF/F/M/S
labels = ['DETF IV', 'Facility', 'Stage II', 'Stage I']
linestyle = [[2, 4, 6, 4], [], [8, 4], [3, 4]]

# Fiducial value and plotting
fig = P.figure()
axes = [fig.add_subplot(121), fig.add_subplot(122)]

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
    F, lbls = rf.combined_fisher_matrix( F_list, expand=zfns, names=pnames,
                                         exclude=excl )
    cov = np.linalg.inv(F)
    errs = np.sqrt(np.diag(cov))
    
    # Identify functions of z
    pDV = rf.indices_for_param_names(lbls, 'DV*')
    pFF = rf.indices_for_param_names(lbls, 'F*')
    
    DV = ((1.+zc)**2. * dAc**2. * C*zc / Hc)**(1./3.)
    Fz = (1.+zc) * dAc * Hc / C
    
    indexes = [pFF, pDV]
    fn_vals = [Fz, DV]
    
    # Plot errors as fn. of redshift
    for jj in range(len(axes)):
        err = errs[indexes[jj]] / fn_vals[jj]
        line = axes[jj].plot( zc, err, color=colours[k], lw=1.8, marker='o', 
                              label=labels[k] )
        line[0].set_dashes(linestyle[k])

# Subplot labels
ax_lbls = ["$\sigma_F/F$", "$\sigma_{D_V}/D_V$"]
ymax = [0.057, 0.057]

# Move subplots
# pos = [[x0, y0], [x1, y1]]
l0 = 0.15
b0 = 0.1
ww = 0.8
hh = 0.88 / 2.
j = 0
for i in range(len(axes)):
    axes[i].set_position([l0, b0 + hh*i, ww, hh])
    
    axes[i].tick_params(axis='both', which='major', labelsize=14, width=1.5, size=8., pad=8.)
    axes[i].tick_params(axis='both', which='minor', labelsize=14, width=1.5, size=8.)
    
    if i == 1: axes[i].tick_params(axis='both', which='major', labelbottom='off')
    axes[i].tick_params(axis='both', which='major', labelsize=20.)
    
    # Set axis limits
    #axes[i].set_xlim((0.25, 2.2))
    axes[i].set_xlim((0., 2.5))
    axes[i].set_ylim((0., ymax[i]))
    
    # Add label to panel
    #P.figtext(l0 + 0.02, b0 + hh*(0.86+i), ax_lbls[i], 
    #          fontdict={'size':'x-large'}) #, bbox=dict(ec='k', fc='none', lw=1.2))
    
    if i == 0: axes[i].set_xlabel('$z$', labelpad=10., fontdict={'fontsize':'xx-large'})
    axes[i].set_ylabel(ax_lbls[i], labelpad=15., fontdict={'fontsize':'xx-large'})
    
    # Set tick locations
    ymajorLocator = matplotlib.ticker.MultipleLocator(0.02)
    yminorLocator = matplotlib.ticker.MultipleLocator(0.01)
    axes[i].yaxis.set_major_locator(ymajorLocator)
    axes[i].yaxis.set_minor_locator(yminorLocator)

# Manually add shared x label
#P.figtext(0.5, 0.02, "$z$", fontdict={'size':'xx-large'})

P.legend(prop={'size':'large'}, bbox_to_anchor=[0.88, 0.98], frameon=False)

# Set size
P.gcf().set_size_inches(8.4, 7.8)
P.savefig('fig09-lss-distances.pdf', transparent=True)
P.show()
