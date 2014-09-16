#!/usr/bin/python
"""
Process EOS Fisher matrices and overplot results for several experiments
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

    # Load Fisher matrices and P(k) constraints as fn. of z
    F_list = []; F_eos_list = []
    kc = []; pk = []; pkerr = []
    for i in range(zc.size):
        F_list.append( np.genfromtxt(root+"-fisher-%d.dat" % i) )
        F_eos_list.append( np.genfromtxt(root+"-fisher-eos-%d.dat" % i) )
        _kc, _pk, _pkerr = np.genfromtxt(root+"-pk-%d.dat" % i).T
        kc.append(_kc); pk.append(_pk); pkerr.append(_pkerr)
    Nbins = zc.size
    
    # EOS FISHER MATRIX
    pnames = ['A', 'b_HI', 'Tb', 'sigma_NL', 'sigma8', 'n_s', 'f', 'aperp', 'apar', 
              'omegak', 'omegaDE', 'w0', 'wa', 'h', 'gamma']
    #0:A, 1:b_HI, 2:sigma_NL, [3:omegak,] 4:omegaDE, 5:w0, 6:wa, 7:h, 8:gamma
    zfns = [1,]
    
    # Fix w0,wa
    F_eos, lbls = baofisher.combined_fisher_matrix( F_eos_list, 
                                                    #exclude=[2,4,5,6,7,8,9,12], 
                                                    exclude=[2,4,5,6,7,8,  11,12], 
                                                    expand=zfns, names=pnames)
    Finv = np.linalg.inv(F_eos) # Pre-invert, for efficiency
    px = baofisher.indexes_for_sampled_fns(3, zc.size, zfns) # x
    py = baofisher.indexes_for_sampled_fns(4, zc.size, zfns) # y
    
    #baofisher.plot_corrmat(F_eos, lbls)
    
    # Fix wa
    F_eos2, lbls2 = baofisher.combined_fisher_matrix( F_eos_list, 
                                                    exclude=[2,4,5,6,7,8,  12], 
                                                    expand=zfns, names=pnames)
    Finv2 = np.linalg.inv(F_eos2) # Pre-invert, for efficiency
    px2 = baofisher.indexes_for_sampled_fns(3, zc.size, zfns) # x
    py2 = baofisher.indexes_for_sampled_fns(4, zc.size, zfns) # y
    
    print "sigma(h) =", np.sqrt( Finv2[baofisher.indexes_for_sampled_fns(6, zc.size, zfns),baofisher.indexes_for_sampled_fns(6, zc.size, zfns)] )
    
    # Fix w0,wa,gamma
    F_eos3, lbls3 = baofisher.combined_fisher_matrix( F_eos_list, 
                                                    #exclude=[2,4,5,6,7,8,9,12], 
                                                    exclude=[2,4,5,6,7,8,  11,12,14], 
                                                    expand=zfns, names=pnames)
    Finv3 = np.linalg.inv(F_eos3) # Pre-invert, for efficiency
    px3 = baofisher.indexes_for_sampled_fns(3, zc.size, zfns) # x
    py3 = baofisher.indexes_for_sampled_fns(4, zc.size, zfns) # y
    
    
    # Fiducial point
    x = experiments.cosmo['omega_k_0']
    y = experiments.cosmo['omega_lambda_0']
    if k == 0: ax.plot(x, y, 'kx', ms=10)
    
    # omega_k, omega_DE (wa fixed)
    w, h, ang, alpha = baofisher.ellipse_for_fisher_params(px2[0], py2[0], F_eos2, Finv=Finv2)
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                 height=alpha[kk]*h, angle=ang, fc='y', ec='y', 
                 lw=2., alpha=0.25) for kk in range(0, 2)]
    for e in ellipses: ax.add_patch(e)
    
    # omega_k, omega_DE (w0,wa fixed)
    w, h, ang, alpha = baofisher.ellipse_for_fisher_params(px[0], py[0], F_eos, Finv=Finv)
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                 height=alpha[kk]*h, angle=ang, fc='none', ec='b', 
                 lw=2., alpha=1.0) for kk in range(0, 2)]
    for e in ellipses: ax.add_patch(e)
    
    # omega_k, omega_DE (w0,wa,gamma fixed)
    w, h, ang, alpha = baofisher.ellipse_for_fisher_params(px3[0], py3[0], F_eos3, Finv=Finv3)
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                 height=alpha[kk]*h, angle=ang, fc='none', ec='r', 
                 lw=2., alpha=1.0) for kk in range(0, 2)]
    for e in ellipses: ax.add_patch(e)
    
    """
    # Add error ellipse for Euclid (omega_k = 0)
    w, h, ang, alpha = baofisher.ellipse_for_fisher_params( 0, 1, None, 
                                            Finv=euclid.cov_w0_wa_fixed_gamma_ok_ref )
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                 height=alpha[kk]*h, angle=ang, fc='b', ec='b', lw=2., 
                 alpha=0.15) for kk in range(0, 2)]
    for e in ellipses: ax.add_patch(e)
    """
    
    """
    # Ellipse for Euclid (omega_k marginalised)
    w, h, ang, alpha = baofisher.ellipse_for_fisher_params( 0, 1, None, 
                                            Finv=euclid.cov_gamma_w_okmarg_ref )
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                 height=alpha[kk]*h, angle=ang, fc='none', ec='b', lw=2., 
                 alpha=0.9) for kk in range(0, 2)]
    for e in ellipses: ax.add_patch(e)
    """

#for i in range(len(names)):
#    l = matplotlib.lines.Line2D([0.,], [0.,], lw=2.5, color=colours[i])
#    lines.append(l)

labels = ['SKA+MK ($w_a$ fixed)', 'SKA+MK ($w_0, w_a$ fixed)', 'SKA+MK ($w_0, w_a, \gamma$ fixed)']
lines = []
lines.append( matplotlib.lines.Line2D([0.,], [0.,], lw=8.5, color='y', alpha=0.3) )
lines.append( matplotlib.lines.Line2D([0.,], [0.,], lw=2.5, color='b') )
lines.append( matplotlib.lines.Line2D([0.,], [0.,], lw=2.5, color='r') )
#lines.append( matplotlib.lines.Line2D([0.,], [0.,], lw=8.5, color='b', alpha=0.3) )
#lines.append( matplotlib.lines.Line2D([0.,], [0.,], lw=2.5, color=colours[0]) )

ax.legend((l for l in lines), (lbl for lbl in labels), prop={'size':'x-large'})

ax.set_xlim((-0.04, 0.04))
ax.set_ylim((0.705, 0.775))

ax.set_xlabel(r"$\Omega_K$", fontdict={'fontsize':'20'})
ax.set_ylabel(r"$\Omega_{DE}$", fontdict={'fontsize':'20'})

fontsize = 16.
for tick in ax.yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in ax.xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)

P.title(r"z=%3.3f - %3.3f ($\nu_\mathrm{max}=1400$ MHz)" % (np.min(zc)-0.05, np.max(zc)+0.05))

P.tight_layout()
P.show()
