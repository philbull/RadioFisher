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
fname = 'weave-fsigma8.pdf'

names = ['WEAVEhdeep', 'WEAVEhmid', 'WEAVEhwide', 
         'WEAVEldeep', 'WEAVElmid', 'WEAVElwide']
labels = [r'Ly-$\alpha$ Deep', r'Ly-$\alpha$ Mid', r'Ly-$\alpha$ Wide',
          r'Starforming Deep', r'Starforming Mid', r'Starforming Wide',]

colours = ['#8082FF', '#1619A1', '#FFB928', '#ff6600', '#95CD6D', '#007A10', '#CC0000', 
           '#000000', '#858585', '#c1c1c1']
linestyle = [[], [], [], [], [], [], [], [], []]
marker = ['s', 's', 's', 's', 'o', 'o', 'o', 'o', 'o']
ms = [6., 6., 6., 6., 6., 6., 5., 5., 5.]

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
    
    #zfns = ['A', 'b_HI', 'f', 'H', 'DA', 'aperp', 'apar']
    zfns = ['A', 'bs8', 'fs8', 'H', 'DA', 'aperp', 'apar']
    excl = ['Tb', 'n_s', 'sigma8', 'omegak', 'omegaDE', 'w0', 'wa', 'h',
            'gamma', 'N_eff', 'pk*', 'f', 'b_HI', 
            'gamma0', 'gamma1', 'eta0', 'eta1']
    
    # Marginalising over b_1
    F, lbls = rf.combined_fisher_matrix( F_list,
                                         expand=zfns, names=pnames,
                                         exclude=excl )
    cov = np.linalg.inv(F)
    errs = np.sqrt(np.diag(cov))
    
    # Fixing b_1
    #excl.append('b_1')
    #F2, lbls2 = rf.combined_fisher_matrix( F_list,
    #                                     expand=zfns, names=pnames,
    #                                     exclude=excl )
    #cov2 = np.linalg.inv(F2)
    #errs2 = np.sqrt(np.diag(cov2))
    
    # Identify functions of z
    pfs8 = rf.indices_for_param_names(lbls, 'fs8*')
    #pfs8_2 = rf.indices_for_param_names(lbls2, 'fs8*')
    
    print ""
    print "#", names[k]
    print "# z, fsigma8, sigma(fsigma8)"
    for j in range(zc.size):
        print "%4.4f %5.5e %5.5e" % (zc[j], (cosmo['sigma_8']*fc*Dc)[j], errs[pfs8][j])
    
    # FIXME: Disable to get redshift markers
    #marker[k] = None
    
    """
    # Plot errors as fn. of redshift (b_1 marginalised)
    err = errs[pfs8] / (cosmo['sigma_8']*fc*Dc)
    line = P.plot( zc, err, color=colours[k], lw=1.8, ls='dashed',
                   marker=marker[k], markersize=ms[k], markeredgecolor=colours[k] )
    #line[0].set_dashes('dashed')
    """
    # Plot errors as fn. of redshift (b_1 fixed)
    err2 = errs2[pfs8_2] / (cosmo['sigma_8']*fc*Dc)
    line = P.plot( zc, err2, color=colours[k], label=labels[k], lw=1.8, ls='solid',
                   marker=marker[k], markersize=ms[k], markeredgecolor=colours[k] )
    
# Load actual f.sigma_8 data and plot it
#dat = np.genfromtxt("fsigma8_data.dat").T
#P.plot(dat[1], dat[3], 'kD')

P.tick_params(axis='both', which='major', labelsize=20, width=1.5, size=8., pad=10)
P.tick_params(axis='both', which='minor', labelsize=20, width=1.5, size=8.)

# Set axis limits
P.xlim((-0.05, 2.2))
#P.ylim((0., 0.045))
P.ylim((0., 0.095))

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

# Set size
P.tight_layout()
#P.gcf().set_size_inches(8.4, 7.8)
P.gcf().set_size_inches(9.5, 6.8)
P.savefig(fname, transparent=True)
P.show()
