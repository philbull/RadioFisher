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

fname = 'mg-fsig8-corr.pdf'
#fname = 'mg-fsigma8-scaledep.pdf'

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
#colours = ['#a6a6a6', '#757575', '#000000', '#5DBEFF', '#1C7EC5', '#1619A1', 
#           '#FFB928', '#ff6600', '#CC0000', '#CC0000', '#95CD6D', 
#           '#007A10', '#ff6600',
#           '#858585', '#c1c1c1', 'c', 'm']
linestyle = [[], [], [], [], [], [], [], [], [], [], [], [], [],]
marker = ['D', 'D', 'D', 's', 's', 's', 'o', 'o', 'o', 'o', 'v', 'v', 'o', 'D', 'd', 'd']
ms = [5., 5., 5., 6., 6., 6., 6., 6., 5., 5., 6., 6., 5., 5., 5., 5.]

# Fiducial value and plotting
P.subplot(111)

# Span redshift ranges from other surveys
#P.axvspan(0., 1., color='y', alpha=0.05)
#P.axvspan(1., 1.9, color='y', alpha=0.1)
#P.axvspan(1.9, 3.4, color='y', alpha=0.05)
#P.axvline(0.75, ls='dashed', color='k', lw=1.5, alpha=0.3)
#P.axvline(1.95, ls='dashed', color='k', lw=1.5, alpha=0.3)
#P.axvline(3.5, ls='dashed', color='k', lw=1.5, alpha=0.3)

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
    zfns = ['A', 'bs8', 'fs8', 'H', 'DA', 'aperp', 'apar']
    excl = ['Tb', 'n_s', 'sigma8', 'omegak', 'omegaDE', 'w0', 'wa', 'h',
            'gamma', 'N_eff', 'pk*', 'f', 'b_HI', 
            'gamma0', 'gamma1', 'eta0', 'eta1', 'A_xi', 'logkmg',
            'sigma8tot', 'sigma_8', 'k*']
    
    # Calculate Fisher matrix
    F, lbls = rf.combined_fisher_matrix( F_list, expand=zfns, names=pnames,
                                         exclude=excl )
    print lbls
    cov = np.linalg.inv(F)
    errs = np.sqrt(np.diag(cov))
    
    # Identify functions of z
    pfs8 = rf.indices_for_param_names(lbls, 'fs8*')
    pbs8 = rf.indices_for_param_names(lbls, 'bs8*')
    
    # (2) Plot fs8-bs8 correlation as fn. of redshift (b_1 fixed)
    r = cov[pfs8, pbs8] / np.sqrt(cov[pfs8, pfs8] * cov[pbs8, pbs8])
    
    #line = P.plot( zc, r, color=colours[k], label=labels[k], lw=1.8, ls='solid',
    #               marker=marker[k], markersize=ms[k], markeredgecolor=colours[k] )
    
    # Plot errors as fn. of redshift
    if labels[k] is not None:
        P.plot(zc, r, color=colours[k], label=labels[k], lw=1.8, ls='solid',
               marker=marker[k], markersize=ms[k], markeredgecolor=colours[k] )
    else:
        P.plot(zc, r, color=colours[k], label=labels[k], lw=1.8,
               marker=marker[k], markersize=ms[k], markeredgecolor=colours[k], 
               dashes=[4,3] )

P.axhline(0., lw=1.5, color='k', alpha=0.5)

P.tick_params(axis='both', which='major', labelsize=20, width=1.5, size=8., pad=10)
P.tick_params(axis='both', which='minor', labelsize=20, width=1.5, size=5.)

# Set axis limits
##P.xlim((-0.05, 2.6))
#P.ylim((0., 0.045))
##P.ylim((0., 0.083))
##P.xlim((-0.05, 6.2))
P.xlim((-0.001, 6.))
P.ylim((-1., 1.))
#P.xscale('log')

P.xlabel('$z$', labelpad=7., fontdict={'fontsize':'xx-large'})
P.ylabel(r'$r(f \sigma_8, b \sigma_8)$', labelpad=10., fontdict={'fontsize':'xx-large'})

# Set tick locations
P.gca().xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.5))
P.gca().yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
    
#leg = P.legend(prop={'size':14}, loc='lower right', frameon=True, ncol=1)
#leg.get_frame().set_edgecolor('w')
#leg.get_frame().set_alpha(0.1)

# Set size
P.tight_layout()
P.savefig(fname, transparent=True)
P.show()
