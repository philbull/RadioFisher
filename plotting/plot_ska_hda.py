#!/usr/bin/python
"""
Plot functions of redshift.
"""
import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm

cosmo = rf.experiments.cosmo

"""
#-------------------------------------------------------------------------------
# SKA BAO chapter D_V constraints
#-------------------------------------------------------------------------------
filename = "ska-bao-hda.pdf"
names = [ 'fSKA1SUR350_baoonly_25000', 'fSKA1SUR650_baoonly_25000',
          'SKA1MID350_baoonly_25000', 'SKA1MID900_baoonly_25000',
          'gSKASURASKAP_baoonly', 'gSKAMIDMKB2_baoonly', 'gSKA2_baoonly',
          'EuclidRef_baoonly',]
labels = [ 'SKA1-SUR B1 (IM)', 'SKA1-SUR B2 (IM)', 
           'SKA1-MID B1 (IM)', 'SKA1-MID B2 (IM)',
           'SKA1-SUR (gal.)', 'SKA1-MID (gal.)', 'SKA2 (gal.)',
           'Euclid (gal.)',]
colours = ['#8082FF', '#8082FF', '#1619A1', '#1619A1', '#FFB928', '#ff6600', '#CC0000', 
           '#858585', '#c1c1c1', '#4D4D4D', '#858585', '#c1c1c1']
linestyle = [[], [4,2], [], [4,2], [], [], [], [], [], [], [], []]
marker = ['D', 'D', 's', 's', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
ms = np.ones(10)*4.5
ymax = [0.075, 0.075]
"""

#-------------------------------------------------------------------------------
# SKA galaxy survey chapter H, D_A constraints
#-------------------------------------------------------------------------------
filename = "ska-galaxy-hda.pdf"
names = [ 'gSKAMIDMKB2', 'gSKASURASKAP', 'gSKA2', 'BOSS', 'EuclidRef', 'WFIRST']
labels = [ 'SKA1-MID+MeerKAT', 'SKA1-SUR+ASKAP', 'SKA 2', 'BOSS', 'Euclid', 'WFIRST']
colours = ['#1619A1', '#FFB928', '#CC0000', 
           '#3D3D3D', '#858585', '#c1c1c1']
linestyle = [[], [], [], [], [], [], [], [], [], [], [], []]
marker = ['D', 'D', 's', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
ms = np.ones(10)*4.5
ymax = [0.045, 0.045]


################################################################################
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
    pnames = rf.load_param_names(root+"-fisher-full-0.dat")
    
    #zfns = ['A', 'b_HI', 'f', 'DV', 'F']
    zfns = ['A', 'bs8', 'fs8', 'DA', 'H']
    excl = ['Tb', 'sigma8', 'n_s', 'omegak', 'omegaDE', 'w0', 'wa', 'h', 
            'gamma', 'N_eff', 'pk*', 'f', 'b_HI']
    F, lbls = rf.combined_fisher_matrix( F_list, expand=zfns, names=pnames,
                                         exclude=excl )
    cov = np.linalg.inv(F)
    errs = np.sqrt(np.diag(cov))
    print lbls
    
    # Identify functions of z
    pDA = rf.indices_for_param_names(lbls, 'DA*')
    pH = rf.indices_for_param_names(lbls, 'H*')
    
    for i in pH:
        print i, lbls[i]
    
    errDA = 1e3 * errs[pDA] / dAc
    errH = 1e2 * errs[pH] / Hc
    indexes = [pDA, pH]
    err = [errDA, errH]
    
    # Plot errors as fn. of redshift
    for jj in range(len(axes)):
        line = axes[jj].plot( zc, err[jj], color=colours[k], lw=1.8, 
                              marker=marker[k], ms=ms[k], label=labels[k] )
        line[0].set_dashes(linestyle[k])
    

# Subplot labels
ax_lbls = ["$\sigma_{D_A}/ D_A$", "$\sigma_H/H$"]


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
    axes[i].set_xlim((-0.1, 2.5))
    axes[i].set_ylim((0., ymax[i]))
    
    if i == 0: axes[i].set_xlabel('$z$', labelpad=10., fontdict={'fontsize':'xx-large'})
    axes[i].set_ylabel(ax_lbls[i], labelpad=15., fontdict={'fontsize':'xx-large'})
    
    # Set tick locations
    axes[i].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.01))
    axes[i].yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.005))

leg = P.legend(prop={'size':'medium'}, loc='upper right', frameon=True, ncol=2)
leg.get_frame().set_alpha(0.8)
leg.get_frame().set_edgecolor('w')

# Set size
P.gcf().set_size_inches(8.4, 7.8)
P.savefig(filename, transparent=True)
P.show()
