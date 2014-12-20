#!/usr/bin/python
"""
Plot functions of redshift (Fig. 6).
"""
import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
import os
from radiofisher import euclid

PLOT_DIFFERENT_MEASURES = False

cosmo = rf.experiments.cosmo

if not PLOT_DIFFERENT_MEASURES:
    names = ['EuclidRef_paper', 'exptL_paper', 'aexptM_paper', 'exptS_paper']
    colours = ['#CC0000', '#1619A1', '#5B9C0A', '#990A9C'] # DETF/F/M/S
    labels = ['DETF IV', 'Facility', 'Stage II', 'Stage I']
    linestyle = [[2, 4, 6, 4], [], [8, 4], [3, 4], [1,1]]
else:
    names = ['cexptL_bao', 'cexptL_bao_rsd', 'cexptL_bao_pkshift', 
             'cexptL_bao_vol', 'cexptL_bao_allap', 'cexptL_bao_all']
    labels = ['BAO only', 'BAO + RSD', 'BAO + P(k) shift', 'BAO + Volume', 
              'BAO + AP', 'All']
    colours = ['#1619A1', '#CC0000', '#5B9C0A', 'y', '#990A9C', 'c', 'm']
    linestyle = [[], [], [8, 4], [2, 4],  [], [8, 4], [3, 4]]

cosmo_fns = rf.background_evolution_splines(cosmo)
#cosmo = rf.load_power_spectrum(cosmo, "cache_pk.dat", force_load=True)

# Fiducial value and plotting
fig = P.figure()
axes = [fig.add_subplot(221), fig.add_subplot(222), fig.add_subplot(223), 
        fig.add_subplot(224)]

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
    zfns = ['A', 'bs8', 'fs8', 'H', 'DA',]
    excl = ['Tb', 'n_s', 'sigma8', 'omegak', 'omegaDE', 'w0', 'wa', 'h',
            'gamma', 'N_eff', 'pk*', 'f', 'b_HI', 'aperp', 'apar']
    
    # FIXME
    #zfns = ['H', 'DA',]
    #excl = ['Tb', 'n_s', 'sigma8', 'omegak', 'omegaDE', 'w0', 'wa', 'h',
    #        'gamma', 'N_eff', 'pk*', 'f', 'b_HI', 'aperp', 'apar', 'fs8', 'bs8']
    F, lbls = rf.combined_fisher_matrix( F_list, expand=zfns, names=pnames,
                                         exclude=excl )
    cov = np.linalg.inv(F)
    errs = np.sqrt(np.diag(cov))
    print lbls
    
    # Identify functions of z
    pA = rf.indices_for_param_names(lbls, 'A*')
    pDA = rf.indices_for_param_names(lbls, 'DA*')
    pH = rf.indices_for_param_names(lbls, 'H*')
    pf = rf.indices_for_param_names(lbls, 'fs8*')
    #pf = rf.indices_for_param_names(lbls, 'f*')
    
    indexes = [pDA, pA, pH, pf]
    fn_vals = [dAc/1e3, 1., Hc/1e2, cosmo['sigma_8']*fc*Dc]
    #fn_vals = [dAc/1e3, 1., Hc/1e2, fc]
    
    # Output data
    #print names[k]
    #err = errs[indexes[0]] / fn_vals[0]
    #for _i in range(err.size):
    #    print "%3.3f %6.4e %6.4e" % (zc[_i], err[_i], fn_vals[0][_i]*1e3)
    
    # Plot errors as fn. of redshift
    for jj in range(len(axes)):
        err = errs[indexes[jj]] / fn_vals[jj]
        
        line = axes[jj].plot( zc, err, color=colours[k], lw=1.8, marker='o', 
                              label=labels[k] )
        line[0].set_dashes(linestyle[k])
    
# Subplot labels
ax_lbls = ["$\sigma_{D_A}/D_A$", "$\sigma_A/A$", "$\sigma_H/H$", 
           "$\sigma_{f\sigma_8}/f\sigma_8$"]

if PLOT_DIFFERENT_MEASURES:
    ymax = [0.18, 0.85, 0.11, 0.065]
else:
    ymax = [0.07, 0.85, 0.07, 0.05]

# Move subplots
# pos = [[x0, y0], [x1, y1]]
l0 = 0.1
b0 = 0.1
ww = 0.4
hh = 0.8 / 2.
j = 0
for i in range(len(axes)):
    axes[i].set_position([l0 + ww*j, b0 + hh*(i%2), ww, hh])
    
    axes[i].tick_params(axis='both', which='major', labelsize=16, width=1.5, 
                        size=6., pad=6.)
    axes[i].tick_params(axis='both', which='minor', labelsize=16)
    
    # Set axis limits
    #axes[i].set_xlim((0.25, 2.2))
    axes[i].set_xlim((0., 2.43))
    axes[i].set_ylim((0., ymax[i]))
    
    # Add label to panel
    P.figtext(l0 + ww*j + 0.02, b0 + hh*(0.86+(i%2)), ax_lbls[i], 
              fontdict={'size':'x-large'}) #, bbox=dict(ec='k', fc='none', lw=1.2))
    
    # Fix ticks
    if j==1:
        axes[i].yaxis.tick_right()
        axes[i].yaxis.set_label_position("right")
    if i%2 == 1:
        for tick in axes[i].xaxis.get_major_ticks():
            tick.label1.set_visible(False)
    if i % 2 == 1: j += 1
    
    # Hide alternating ticks
    kk = 0
    for tick in axes[i].yaxis.get_major_ticks():
        if kk%2==1:
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
        kk += 1

# Manually add shared x label
P.figtext(0.5, 0.02, "$z$", fontdict={'size':'xx-large'})

# Legend
P.legend(bbox_to_anchor=[0.93,-0.04], frameon=False)

# Set size
P.gcf().set_size_inches(10., 7.)
if not PLOT_DIFFERENT_MEASURES:
    #pass
    P.savefig('fig06-zfns.pdf', transparent=True)
    #P.savefig('BINGO-zfns.pdf', transparent=True)
P.show()
