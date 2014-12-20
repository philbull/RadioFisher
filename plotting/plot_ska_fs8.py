#!/usr/bin/python
"""
Plot f.sigma_8 as a function of z
"""
import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
import matplotlib.ticker
from radiofisher import euclid
import copy

cosmo = rf.experiments.cosmo

#------------------------------------------------
# SKA RSD chapter: All distance measures (f.sigma_8)
#------------------------------------------------

names = ['fSKA1SUR350_25000', 'EuclidRef', 'SKA1MID350_25000', 'gSKA2']
labels = ['SKA1-SUR B1 (IM)', 'Euclid (gal.)', 'SKA1-MID B1 (IM)', 'SKA2 (gal.)']

fig_name = "ska-fs8.pdf"

colours = ['#FFA728', '#6B6B6B', '#1619A1', '#CC0000']


names = ['SKA1MID350_25000', 'SKA1MID900_25000', 'fSKA1SUR350_25000', 'fSKA1SUR650_25000', 
         'gSKAMIDMKB2', 'gSKASURASKAP', 'gSKA2', 'EuclidRef']
labels = ['SKA1-MID B1 (IM)', 'SKA1-MID B2 (IM)', 'SKA1-SUR B1 (IM)', 
          'SKA1-SUR B2 (IM)', 'SKA1-MID (gal.)', 'SKA1-SUR (gal.)', 
          'Full SKA (gal.)', 'Euclid (gal.)']
colours = ['#8082FF', '#1619A1', '#FFB928', '#ff6600', '#95CD6D', '#007A10', '#CC0000', 
           '#000000', '#858585', '#c1c1c1', 'y']

linestyle = [[], [], [], [], [], [], [], [], []]
marker = ['D', 'D', 'D', 'D', 's', 's', 'o', 'o', 'o']
ms = [6., 6., 6., 6., 6., 6., 5., 5., 5.]

XLIM = (-0.02, 2.2)
YLIM = (0., 0.045)

################################################################################
# Fiducial value and plotting
fig = P.figure()
ax = fig.add_subplot(111)

_k = range(len(names)) #[::-1]
for k in _k:
    root = "output/" + names[k]

    # Load cosmo fns.
    dat = np.atleast_2d( np.genfromtxt(root+"-cosmofns-zc.dat") ).T
    zc, Hc, dAc, Dc, fc = dat
    zs, Hs, dAs, Ds, fs = np.genfromtxt(root+"-cosmofns-smooth.dat").T
    kc = np.genfromtxt(root+"-fisher-kc.dat").T
    
    # Load Fisher matrices as fn. of z
    Nbins = zc.size
    F_list = [np.genfromtxt(root+"-fisher-full-%d.dat" % i) for i in range(Nbins)]
    
    # EOS FISHER MATRIX
    pnames = rf.load_param_names(root+"-fisher-full-0.dat")
    zfns = ['A', 'bs8', 'fs8', 'H', 'DA',]
    excl = ['Tb', 'n_s', 'sigma8', 'omegak', 'omegaDE', 'w0', 'wa', 'h',
            'gamma', 'N_eff', 'pk*', 'f', 'b_HI']
    F, lbls = rf.combined_fisher_matrix( F_list, expand=zfns, names=pnames,
                                         exclude=excl )
    cov = np.linalg.inv(F)
    errs = np.sqrt(np.diag(cov))
    print lbls
    
    # Identify functions of z
    pfs8 = rf.indices_for_param_names(lbls, 'fs8*')
    
    print ""
    print "#", names[k]
    print "# z, fsigma8, sigma(fsigma8)"
    for j in range(zc.size):
        print "%4.4f %5.5e %5.5e" % (zc[j], errs[pfs8][j], (cosmo['sigma_8']*fc*Dc)[j])
    
    # Plot errors as fn. of redshift
    err = errs[pfs8] / (cosmo['sigma_8']*fc*Dc)
    line = P.plot( zc, err, color=colours[k], lw=1.8, label=labels[k], 
                   marker=marker[k], markersize=ms[k], markeredgecolor=colours[k] )
    line[0].set_dashes(linestyle[k])

P.tick_params(axis='both', which='major', labelsize=20, width=1.5, size=8., pad=5)
P.tick_params(axis='both', which='minor', labelsize=20, width=1.5, size=8.)

# Add label to panel
#P.figtext(l0 + 0.02, b0 + hh*(0.86+i), ax_lbls[i], 
#          fontdict={'size':'x-large'}) #, bbox=dict(ec='k', fc='none', lw=1.2))

P.xlabel('$z$', labelpad=10., fontdict={'fontsize':'xx-large'})
P.ylabel('$\sigma(f \sigma_8) / (f \sigma_8)$', labelpad=15., fontdict={'fontsize':'xx-large'})

# Set tick locations
P.gca().yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.01))
P.gca().yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.01))
    
leg = P.legend(prop={'size':'large'}, loc='upper right', frameon=True, ncol=2)
leg.get_frame().set_edgecolor('w')
leg.get_frame().set_alpha(0.8)

P.xlim(XLIM)
P.ylim(YLIM)

# Set size
P.tight_layout()
P.gcf().set_size_inches(8.,6.)
P.savefig(fig_name, transparent=True)
print "Figure saved to: %s" % fig_name
P.show()
