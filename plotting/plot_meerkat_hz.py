#!/usr/bin/env python
"""
Plot functions of redshift for RSDs.
"""
import numpy as np
import pylab as plt
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
import os

cosmo = rf.experiments.cosmo

BAND = 'UHF'

#-------------------------------------------------------------------------------
# Define colours
#-------------------------------------------------------------------------------
red1 = '#fb9a99' # IM Band 1
red2 = '#e31a1c' # IM Band 2

orange1 = '#fdbf6f' # LOW (lower band)
orange2 = '#FFD025' # LOW (upper band)

yellow1 = '#FAEB00'

green1 = '#b2df8a' # WL/Continuum Band 1
green2 = '#33a02c' # WL/Continuum Band 2

blue1 = '#a6cee3' # HI Galaxies Band 1
blue2 = '#1f78b4' # HI Galaxies Band 2

violet1 = '#7154A6'

black1 = '#232323' # External Survey 1
black2 = '#707070' # External Survey 2
black3 = '#A9A9A9' # External Survey 3

# Survey areas
#sareas = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12500, 15000]
sareas = [1000, 2000, 4000, 8000, 15000]

if BAND == 'L':
    fname = 'MeerKAT-Lband-Hz.pdf'
    fig_title = "MeerKAT L-band"
    name_root = "MeerKATL_{hrs}hr_{sarea}"
else:
    fname = 'MeerKAT-UHF-Hz.pdf'
    fig_title = "MeerKAT UHF-band"
    name_root = "MeerKATUHF_{hrs}hr_{sarea}"

#colours = [red1, red2, blue1, blue2, black1, black2, 
#           red1, red2, blue1, blue2, black1, black2, 
#           red1, red2, blue1, blue2, black1, black2 ]

colours = [red1, orange1, green1, blue2, violet1]

#linestyle = [[] for i in range(len(names))]
#marker = ['.' for i in range(len(names))]
#ms = [6. for i in range(len(names))]


def get_hz_errs(fname):
    """
    Load Fisher matrix and invert to get errors on H(z) as a function of z.
    """
    root = "output/" + fname
    print(root)
    
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
    
    zfns = ['bs8', 'fs8', 'H', 'DA',]
    excl = ['Tb', 'n_s', 'sigma8', 'omegak', 'omegaDE', 'w0', 'wa', 'h',
            'gamma', 'N_eff', 'pk*', 'f', 'b_HI', 
            'gamma0', 'gamma1', 'eta0', 'eta1', 'A_xi', 'logkmg',
            'sigma8tot', 'sigma_8', 'k*', 'A', 'aperp', 'apar']
    
    F, lbls = rf.combined_fisher_matrix( F_list,
                                         expand=zfns, names=pnames,
                                         exclude=excl )
    print(lbls)
    
    cov = np.linalg.inv(F)
    errs = np.sqrt(np.diag(cov))
    
    # Identify functions of z
    pH = rf.indices_for_param_names(lbls, 'H*')
    errH = 1e2 * errs[pH] / Hc
    
    # Return values
    return zc, errH



# Fiducial value and plotting
plt.subplot(111)


for i, sarea in enumerate(sareas):
        
    # Load Fisher matrices for each survey time
    zc_low, errH_low = get_hz_errs(name_root.format(hrs=1000, sarea=sarea))
    zc_mid, errH_mid = get_hz_errs(name_root.format(hrs=2000, sarea=sarea))
    zc_hi, errH_hi   = get_hz_errs(name_root.format(hrs=4000, sarea=sarea))
    
    if BAND == 'L':
        plt.fill_between(zc_low, errH_low, errH_hi, color=colours[i], alpha=0.5)
        plt.plot(zc_mid, errH_mid, color=colours[i], lw=1.8, marker='.', 
                 label="%d deg$^2$" % sarea)
    else:
        #plt.plot(zc_low, errH_low, color=colours[i], lw=1.2, alpha=0.5)
        plt.plot(zc_low, errH_low, color=colours[i], lw=1.8, dashes=[2,1])
        plt.plot(zc_hi, errH_hi, color=colours[i], lw=1.8, dashes=[3,2])
        plt.plot(zc_mid, errH_mid, color=colours[i], lw=1.8, marker='.', 
                 label="%d deg$^2$" % sarea)
    
        # Plot errors as fn. of redshift
        #P.plot( zc, errH, color=colours[i], label=labels[k], lw=2.2,
        #        marker=marker[k], markersize=ms[k], markeredgecolor=colours[k],
        #        dashes=linestyle[k] )
        
    

plt.tick_params(axis='both', which='major', labelsize=18, width=1.5, size=8., pad=10)
plt.tick_params(axis='both', which='minor', labelsize=18, width=1.5, size=5.)

# Set axis limits
if BAND == 'L':
    plt.xlim((0., 0.55))
    plt.ylim((0., 0.08))
else:
    plt.xlim((0.4, 1.5))
    plt.ylim((0.01, 0.09))


plt.xlabel('$z$', labelpad=7., fontdict={'fontsize':'x-large'})
plt.ylabel('$\sigma_H / H$', labelpad=10., fontdict={'fontsize':'x-large'})

if BAND == 'L':
    plt.legend(loc='upper right', frameon=False, ncol=2)
else:
    plt.legend(loc='upper left', frameon=False, ncol=2)

# Set tick locations
#P.gca().yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
#P.gca().yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.01))
plt.gca().xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.5))
#plt.yscale('log')


plt.title(fig_title)

# Set size
plt.tight_layout()
#P.gcf().set_size_inches(8.4, 7.8)
#P.gcf().set_size_inches(9.5, 6.8)
plt.savefig(fname, transparent=True)
plt.show()
