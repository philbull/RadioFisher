#!/usr/bin/python
"""
Plot constraint on omega_K, allowing it to be free in each redshift bin
"""
import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
import matplotlib.ticker
from units import *
from mpi4py import MPI

import os
import euclid

cosmo = rf.experiments.cosmo

k = 2 # Which experiment to plot
names = ["exptS", "iexptM", "cexptL"]
#colours = ['#CC0000', '#ED5F21', '#FAE300', '#5B9C0A', '#1619A1', '#56129F', '#990A9C', 'y']
colours = ['#5B9C0A', '#ED5F21', '#1619A1']
labels = ['Snapshot', 'Mature', 'Behemoth']

cosmo_fns, cosmo = rf.precompute_for_fisher(rf.experiments.cosmo, "camb/rf_matterpower.dat")
H, r, D, f = cosmo_fns

# Fiducial value and plotting
fig = P.figure()
ax = fig.add_subplot(111)
m = 0

for k in range(len(names)):
    root = "../output/" + names[k]

    # Load cosmo fns.
    dat = np.atleast_2d( np.genfromtxt(root+"-cosmofns-zc.dat") ).T
    zc, Hc, dAc, Dc, fc = dat
    zs, Hs, dAs, Ds, fs = np.genfromtxt(root+"-cosmofns-smooth.dat").T
    kc = np.genfromtxt(root+"-fisher-kc.dat").T
    
    # Load Fisher matrices as fn. of z
    Nbins = zc.size
    F_list = [np.genfromtxt(root+"-fisher-full-%d.dat" % i) for i in range(Nbins)]
    
    # EOS FISHER MATRIX
    # Actually, (aperp, apar) are (D_A, H)
    pnames = ['A', 'b_HI', 'Tb', 'sigma_NL', 'sigma8', 'n_s', 'f', 'aperp', 'apar', 
             'omegak', 'omegaDE', 'w0', 'wa', 'h', 'gamma', 'Mnu']
    pnames += ["pk%d" % i for i in range(kc.size)]
    
    zfns = [9,1]
    excl = [2,4,5,  6,7,8,   11,12,   14,15] # Fix w0, wa
    
    excl  += [i for i in range(len(pnames)) if "pk" in pnames[i]]
    
    F, lbls = rf.combined_fisher_matrix( F_list,
                                                expand=zfns, names=pnames,
                                                exclude=excl )
    
    print lbls
    # Add Planck prior
    Fpl = euclid.add_planck_prior(F, lbls, info=True)
    
    print "-"*50
    print names[k]
    print "-"*50
    
    # Invert matrices
    pok = rf.indexes_for_sampled_fns(3, zc.size, [1,3]) # Updated zfns
    cov = np.linalg.inv(F)
    cov_pl = np.linalg.inv(Fpl)
    
    # No Planck
    P.plot(zc, np.sqrt(np.diag(cov)[pok]), lw=1.5, marker='.', color=colours[k], label=labels[k])
    
    # With Planck
    P.plot(zc, np.sqrt(np.diag(cov_pl)[pok]), lw=2.5, marker='.', color=colours[k], label=labels[k] + " + Planck", alpha=0.8, ls='dotted')

P.legend(loc='upper left', prop={'size':'large'})
P.xlim((0.96*np.min(zc), 1.01*np.max(zc)))

"""
# Planck-only 1D
omegak_planck = 6.5e-3 / 2. # From Planck 2013 XVI, Table 10, Planck+WMAP+highL+BAO

# Shaded region, tick number format, and minor tick location
if not MARGINALISE_W0WA:
    minorLocator = matplotlib.ticker.MultipleLocator(0.001)    
    P.axvspan(omegak_planck, 1., ec='none', fc='#f2f2f2')
    P.axvspan(-1., -omegak_planck, ec='none', fc='#f2f2f2')
    P.axvline(-omegak_planck, ls='dotted', color='k', lw=2.)
    P.axvline(+omegak_planck, ls='dotted', color='k', lw=2.)
    majorFormatter = matplotlib.ticker.FormatStrFormatter('%3.3f')
else:
    minorLocator = matplotlib.ticker.MultipleLocator(0.01)
    majorFormatter = matplotlib.ticker.FormatStrFormatter('%3.2f')
ax.xaxis.set_minor_locator(minorLocator)
ax.xaxis.set_major_formatter(majorFormatter)
"""

fontsize = 20
for tick in ax.xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in ax.yaxis.get_major_ticks():
  #tick.label1.set_visible(False)
  tick.label1.set_fontsize(fontsize)

ax.set_xlabel(r"$z$", fontdict={'fontsize':'20'})
ax.set_ylabel(r"$\sigma(\Omega_K)$", fontdict={'fontsize':'20'})

P.tight_layout()

# Set size and save
#P.gcf().set_size_inches(16.5,10.5)
P.savefig('pub-omegak-fnz.png', dpi=100)

P.show()
