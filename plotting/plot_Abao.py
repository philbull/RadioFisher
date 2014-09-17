#!/usr/bin/python
"""
Plot A_BAO as a function of redshift.
"""

import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
from units import *
from mpi4py import MPI
import os
import euclid

cosmo = rf.experiments.cosmo

names = ['cexptL', 'iexptM', 'exptS']
labels = ['Facility', 'Mature', 'Snapshot']

#colours = ['#CC0000', '#ED5F21', '#FAE300', '#5B9C0A', '#1619A1', '#56129F', '#990A9C']
colours = ['#CC0000', '#5B9C0A', '#1619A1',   '#990A9C', '#FAE300', 'c']
#labels = ["ctestL", "itestL", "testL", "ctestM", "itestM", "testM"] #['Behemoth', 'Mature', 'Snapshot', 'iBeh', 'iiiBeh']
linestyle = ['solid', 'dashed', 'dashdot', 'solid', 'solid', 'dotted']

# Get f_bao(k) function
#cosmo = rf.load_power_spectrum(cosmo, "cache_pk.dat")
#fbao = cosmo['fbao']

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
    pnames = ['A', 'b_HI', 'Tb', 'sigma_NL', 'sigma8', 'n_s', 'f', 'aperp', 'apar', 
             'omegak', 'omegaDE', 'w0', 'wa', 'h', 'gamma', 'Mnu'] #, 'fNL']
    pnames += ["pk%d" % i for i in range(kc.size)]
    zfns = [0,1,6,7,8]
    excl = [2,4,5,  9,10,11,12,13,14,15] #,16] # Exclude all cosmo params
    excl += [i for i in range(len(pnames)) if "pk" in pnames[i]]
    
    F, lbls = rf.combined_fisher_matrix( F_list,
                                                expand=zfns, names=pnames,
                                                exclude=excl )
    cov = np.linalg.inv(F)
    
    # Get functions of z
    zfns = [0,1,3,4,5]
    pA  = rf.indexes_for_sampled_fns(0, zc.size, zfns)
    print "A:", [lbls[j] for j in pA], "\n"
    
    # Plot errorbars
    errs = np.sqrt(np.diag(cov))
    err = errs[pA]
    
    P.plot( zc, err, color=colours[k], lw=1.8, marker='.', label=labels[k], 
            ls=linestyle[k] )


P.xlim((0.31, 2.52))
#P.ylim((0.5, 1.09))

# Resize labels/ticks
fontsize = 18
ax = P.gca()
for tick in ax.xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
  if i == 0: tick.label1.set_visible(False) # Hide x lbls in upper subplots
for tick in ax.yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)

# y labels
P.ylabel(r"$\sigma_{A} / A$", fontdict={'fontsize':'20'})
P.xlabel(r"$z$", fontdict={'fontsize':'20'})
P.legend(loc='upper left', prop={'size':'large'})

P.tight_layout()
# Set size
#P.gcf().set_size_inches(8.5, 7.)

##P.savefig('pub-Abao-z.png', dpi=100)

P.show()
