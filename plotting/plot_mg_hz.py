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

#fname = 'mg-Hz.pdf'
fname = 'TEST-Hz.pdf'

names = ['BOSS_mg', 'EuclidRef_mg', 'HETDEXdz03_mg', 
         'gSKA2MG_mg', 'gMIDMK_B2_Rebase_mg', 'gMIDMK_B2_Alt_mg', 
         'hMID_B1_Rebase_mg', 'hMID_B1_Octave_mg', 
         'hMID_B2_Rebase_mg', 'hMID_B2_Octave_mg',
         'aLOW_Upd_mg', 'aLOW_Alt_mg', ]

labels = ['BOSS forecast (GS)', 'H$\\alpha$ survey (GS)', 'HETDEX (GS)', 
          'SKA2 (GS)',
          'MID B2 + MK (GS)', None, 
          'MID B1 + MK (IM)', None, 
          'MID B2 + MK (IM)', None, 
          'LOW (IM)', None, ]

colours = ['#a6a6a6', '#757575', '#000000', 
           '#1619A1',
           '#1C7EC5', '#1C7EC5', 
           '#FFB928', '#FFB928', '#CC0000', '#CC0000',
           '#007A10', '#007A10', '#95CD6D', 
           '#ff6600',
           '#858585', '#c1c1c1', 'c', 'm']

linestyle = [[], [], [], [], [], [], [], [], [], [], [], [], [],]
marker = ['D', 'D', 'D', 's', 's', 's', 'o', 'o', 'o', 'o', 'v', 'v', 'o', 'D', 'd', 'd']
ms = [5., 5., 5., 6., 6., 6., 6., 6., 5., 5., 6., 6., 5., 5., 5., 5.]


# FIXME

names = ['iCVTEST1_hiraxtest', 'iCVTEST1_hiraxtest_kmaxcut', 'iCVTEST2_hiraxtest', 'gCV_hiraxtest']
labels = ['HIRAX CV-lim 0 < z < 3', 'HIRAX CV-lim 0 < z < 3 kmax', 'HIRAX CV-lim 2 < z < 5', 'Galaxies CV-lim 0 < z < 3']

# Fiducial value and plotting
P.subplot(111)

# Span redshift ranges from other surveys
#P.axvspan(0., 1., color='y', alpha=0.05)
#P.axvspan(1., 1.9, color='y', alpha=0.1)
#P.axvspan(1.9, 3.4, color='y', alpha=0.05)
P.axvline(0.75, ls='dashed', color='k', lw=1.5, alpha=0.3)
P.axvline(2.0, ls='dashed', color='k', lw=1.5, alpha=0.3)
P.axvline(3.5, ls='dashed', color='k', lw=1.5, alpha=0.3)

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
    #zfns = ['A', 'bs8', 'fs8', 'H', 'DA', 'aperp', 'apar']
    zfns = ['bs8', 'fs8', 'H', 'DA',]
    excl = ['Tb', 'n_s', 'sigma8', 'omegak', 'omegaDE', 'w0', 'wa', 'h',
            'gamma', 'N_eff', 'pk*', 'f', 'b_HI', 
            'gamma0', 'gamma1', 'eta0', 'eta1', 'A_xi', 'logkmg',
            'sigma8tot', 'sigma_8', 'k*', 'A', ] #'sigma_NL']
    
    #excl = ['A', 'bs8', 'fs8', 'H', 'DA', 'aperp', 'apar', 'Tb', 'N_eff', 'pk*', 'f', 'wa', 'gamma0', 'gamma1', 'eta0', 'eta1', 'A_xi', 'logkmg',]
    #zfns = ['w0', ]
    
    # Marginalising over b_1
    F, lbls = rf.combined_fisher_matrix( F_list,
                                         expand=zfns, names=pnames,
                                         exclude=excl )
    print lbls
    
    # FIXME
    F[0,0] += 1e-2
    print F[0,:]
    
    cov = np.linalg.inv(F)
    errs = np.sqrt(np.diag(cov))
    
    # Identify functions of z
    #errDA = 1e3 * errs[pDA] / dAc
    pH = rf.indices_for_param_names(lbls, 'H*')
    errH = 1e2 * errs[pH] / Hc
    
    # FIXME: Disable to get redshift markers
    #marker[k] = None
    
    """
    # Plot errors as fn. of redshift (b_1 marginalised)
    err = errs[pfs8] / (cosmo['sigma_8']*fc*Dc)
    line = P.plot( zc, err, color=colours[k], lw=1.8, ls='dashed',
                   marker=marker[k], markersize=ms[k], markeredgecolor=colours[k] )
    #line[0].set_dashes('dashed')
    """
    # Plot errors as fn. of redshift
    if labels[k] is not None:
        P.plot( zc, errH, color=colours[k], label=labels[k], lw=1.8,
                marker=marker[k], markersize=ms[k], markeredgecolor=colours[k] )
    else:
        P.plot( zc, errH, color=colours[k], label=labels[k], lw=1.8,
                marker=marker[k], markersize=ms[k], markeredgecolor=colours[k],
                dashes=[4,3] )

P.tick_params(axis='both', which='major', labelsize=20, width=1.5, size=8., pad=10)
P.tick_params(axis='both', which='minor', labelsize=20, width=1.5, size=5.)

# Set axis limits
##P.xlim((-0.05, 2.6))
#P.ylim((0., 0.045))
##P.ylim((0., 0.083))
##P.xlim((-0.05, 6.2))
P.xlim((-0.001, 6.1))
P.ylim((1e-3, 1.5))
#P.xscale('log')

# Add label to panel
#P.figtext(l0 + 0.02, b0 + hh*(0.86+i), ax_lbls[i], 

#          fontdict={'size':'x-large'}) #, bbox=dict(ec='k', fc='none', lw=1.2))
P.xlabel('$z$', labelpad=7., fontdict={'fontsize':'xx-large'})
P.ylabel('$\sigma_H / H$', labelpad=10., fontdict={'fontsize':'xx-large'})

# Set tick locations
#P.gca().yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
#P.gca().yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.01))
P.gca().xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.5))
P.yscale('log')
    
leg = P.legend(prop={'size':14}, loc='upper right', frameon=True, ncol=2)
leg.get_frame().set_edgecolor('w')
leg.get_frame().set_alpha(0.1)

# Set size
P.tight_layout()
#P.gcf().set_size_inches(8.4, 7.8)
#P.gcf().set_size_inches(9.5, 6.8)
P.savefig(fname, transparent=True)
P.show()
