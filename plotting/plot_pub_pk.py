#!/usr/bin/python
"""
Process EOS Fisher matrices and plot P(k).
"""

import numpy as np
import pylab as P
import baofisher
import matplotlib.patches
import matplotlib.cm
from units import *
from mpi4py import MPI
import experiments
import os
import euclid

cosmo = experiments.cosmo

#names = ["GBT", "BINGO", "WSRT", "APERTIF", "JVLA", "ASKAP", "KAT7", "MeerKAT", "SKA1mid", "SKA1MK", "iSKA1MK", "aSKA1MK", "SKA1MK_A0"]
names = ["SKA1MK",] #["SKA1mid",] ["MeerKAT",]
colours = ['#22AD1A', '#3399FF', '#ED7624']

# Fiducial value and plotting
fig = P.figure()
ax = fig.add_subplot(111)

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
    
    # Save P(k) rebinning info
    #np.savetxt(root+"-rebin-Fbase-%d.dat" % i, np.array(binning_info['F_base']) )
    #np.savetxt(root+"-rebin-cumul-%d.dat" % i, np.array(binning_info['cumul']) )
    #np.savetxt(root+"-rebin-kgrid-%d.dat" % i, np.array(binning_info['kgrid']) )
    #np.savetxt(root+"-rebin-Vfac-%d.dat" % i, np.array([binning_info['Vfac'],]) )
    
    # EOS FISHER MATRIX
    # Actually, (aperp, apar) are (D_A, H)
    pnames = ['A', 'b_HI', 'Tb', 'sigma_NL', 'sigma8', 'n_s', 'f', 'aperp', 'apar', 
             'omegak', 'omegaDE', 'w0', 'wa', 'h', 'gamma']
    pnames += ["pk%d" % i for i in range(kc.size)]
    zfns = [1,]
    F, lbls = baofisher.combined_fisher_matrix( F_list,
                                                expand=zfns, names=pnames,
                                                exclude=[2,4,5,6,7,8 ] )
    
    # Remove elements with zero diagonal (completely unconstrained)
    zero_idxs = np.where(np.diag(F) == 0.)[0]
    print "Zero idxs:", zero_idxs
    F = baofisher.fisher_with_excluded_params(F, excl=zero_idxs)
    lbls = lbls[:-zero_idxs.size]
    
    
    #baofisher.plot_corrmat(F, lbls)
    
    
    # Overlay error ellipses as a fn. of z
    p1 = baofisher.indexes_for_sampled_fns(4, zc.size, zfns)
    #p2 = baofisher.indexes_for_sampled_fns(5, zc.size, zfns)
    
    # Full covmat
    cov = np.linalg.inv(F)
    diags = np.sqrt(np.diag(cov))
    
    # Reduced covmat
    #F2 = baofisher.fisher_with_excluded_params(F, excl=[l for l in range(19, 55)])
    #cov2 = np.linalg.inv(F2)
    #diags2 = np.sqrt(np.diag(cov2))
    
    # Print diags.
    for i in range(diags.size):
        #if i < diags2.size:
        #    print "%2d   %10s   %3.4f   %3.4f" % (i, lbls[i], diags[i], diags2[i])
        #else:
        print "%2d   %10s   %3.4f" % (i, lbls[i], diags[i])
    exit()
    
    P.subplot(111)
    idxs = np.array([l for l in range(7, 43)])
    print idxs, idxs.size
    P.plot(kc[:idxs.size], diags[idxs], lw=1.5)
    P.xscale('log')
    P.ylim((0., 0.2))
    P.show()
    
    exit()
    
    
    print "Cond.", np.linalg.cond(F)
    
    
    F[-1,:] *= 1e5; F[:,-1] *= 1e5
    F[-2,:] *= 1e5; F[:,-2] *= 1e5
    F[-3,:] *= 1e5; F[:,-3] *= 1e5
    
    print "Cond.", np.linalg.cond(F)
    
    #print np.diag(F)
    #print F[-1,:]
    
    baofisher.plot_corrmat(F, lbls)
    P.show()
    exit()


#ax.legend((l for l in lines), (lbl for lbl in labels), prop={'size':'x-large'})
#ax.set_xlim((-1.31, -0.70))
#ax.set_ylim((-0.7, 0.7))

P.ylim((0., 1.))
P.xscale('log')

ax.set_xlabel(r"$k$", fontdict={'fontsize':'20'})
ax.set_ylabel(r"$P(k)$", fontdict={'fontsize':'20'})

fontsize = 16.
for tick in ax.yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in ax.xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)

P.tight_layout()
P.show()
