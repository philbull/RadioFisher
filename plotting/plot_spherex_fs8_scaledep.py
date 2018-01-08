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

fname = 'spherex-fsig8-scaledep.pdf'

names = [ 'gSPHEREx1_mgscaledep', 'gSPHEREx2_mgscaledep', 'EuclidRef_mg' ]
#labels = [ 'SPHEREx 0.003', 'SPHEREx 0.008', 'Euclid spectro-z', 'DESI spectro-z']
labels = [ '$\sigma(z)/(1+z) < 0.003$', '$\sigma(z)/(1+z) < 0.008$', 
          r'${\rm Euclid}$ ${\rm spectro-z}$', 'DESI spectro-z']
#colours = [ '#80B6D6', '#93C993', '#757575', '#a8a8a8'] # '#c8c8c8',
colours = [ '#F82E1E', '#1F7CC0', '#333333', '#a8a8a8']

# Dashes for different k bins
dashes = [[], [6,4], [4,2], [4,2,4,4]]
lws = [2.8, 2.8, 1.5, 1.]
alphas = [1., 1., 0.5, 0.2]

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
    print pnames
    
    #zfns = ['A', 'b_HI', 'f', 'H', 'DA', 'aperp', 'apar']
    zfns = ['H', 'DA', 'aperp', 'apar']
    for j in range(4): zfns += ['k%dbs8' % j, 'k%dfs8' % j]
    excl = ['A', 'Tb', 'n_s', 'sigma8', 'omegak', 'omegaDE', 'w0', 'wa', 'h',
            'gamma', 'N_eff', 'pk*', 'f', 'b_HI', 
            'gamma0', 'gamma1', 'eta0', 'eta1', 'A_xi', 'logkmg',
            'fs8*', 'bs8*', 'sigma8tot', 'sigma_8',]
    
    # Combine Fisher matrices
    F, lbls = rf.combined_fisher_matrix( F_list,
                                         expand=zfns, names=pnames,
                                         exclude=excl )
    print lbls
    
    # Look for unconstrained bins and remove them
    excl = []
    for j in range(4):
        for n in range(zc.size):
            pbs8 = 'k%dbs8%d' % (j, n)
            pfs8 = 'k%dfs8%d' % (j, n)
            Fbs8 = np.abs(np.diag(F)[lbls.index(pbs8)])
            Ffs8 = np.abs(np.diag(F)[lbls.index(pfs8)])
            if (Fbs8 < 1e-8) or (Ffs8 < 1e-8):
                excl += [pbs8, pfs8]
    F, lbls = rf.combined_fisher_matrix( [F,], expand=[], 
                     names=lbls, exclude=excl )
    cov = np.linalg.inv(F)
    errs = np.sqrt(np.diag(cov))
    
    # Identify scale-dependent functions
    for j in range(2):
      l = labels[k] if j == 1 else None
      try:
        pfs8 = rf.indices_for_param_names(lbls, 'k%dfs8*' % j)
        pbs8 = rf.indices_for_param_names(lbls, 'k%dbs8*' % j)
        
        # Plot errors as fn. of redshift
        err = errs[pfs8] / (cosmo['sigma_8']*fc*Dc)
        
        if 'Euclid' in labels[k]:
            idxs = np.where(np.logical_and(zc >= 1.1, zc <= 1.9))
            zc = zc[idxs]
            err = err[idxs]
        
        P.plot( zc, err, color=colours[j], lw=lws[j],
                alpha=alphas[j], dashes=dashes[k] )
        
        # Dummy, for legend
        P.plot( zc, -err, color='k', label=l, lw=lws[j],
                alpha=alphas[j], dashes=dashes[k] )
        # marker=marker[k], markersize=ms[k], markeredgecolor=colours[k],
      except:
        pass

P.tick_params(axis='both', which='major', labelsize=20, width=1.5, size=8., pad=10)
P.tick_params(axis='both', which='minor', labelsize=20, width=1.5, size=5.)

# Set axis limits
P.xlim((-0.001, 2.4))
P.ylim((8e-3, 5.))
#P.xscale('log')

#          fontdict={'size':'x-large'}) #, bbox=dict(ec='k', fc='none', lw=1.2))
P.xlabel('$z$', labelpad=10., fontdict={'fontsize':'xx-large'})
P.ylabel('$\sigma(f \sigma_8) / (f \sigma_8)$', labelpad=15., fontdict={'fontsize':'xx-large'})

# Set tick locations
#P.gca().yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
#P.gca().yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.01))
P.gca().xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.5))
P.yscale('log')
    
leg = P.legend(prop={'size':17}, loc='upper left', frameon=False, ncol=2)
#leg.get_frame().set_edgecolor('w')
#leg.get_frame().set_alpha(0.5)

# Scale-dependence labels
P.figtext(0.25, 0.26, "0.01 < k < 0.1 Mpc$^{-1}$", fontsize=16, backgroundcolor='w', color=colours[1])
P.figtext(0.38, 0.705, "0 < k < 0.01 Mpc$^{-1}$", fontsize=16, backgroundcolor='w', color=colours[0])

# Set size
P.tight_layout()
#P.gcf().set_size_inches(8.4, 7.8)
#P.gcf().set_size_inches(9.5, 6.8)
P.savefig(fname, transparent=True)
P.show()
