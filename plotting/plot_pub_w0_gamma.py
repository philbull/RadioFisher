#!/usr/bin/python
"""
Process EOS Fisher matrices and overplot results for several rf.experiments.
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

#names = ["GBT", "BINGO", "WSRT", "APERTIF", "JVLA", "ASKAP", "KAT7", "MeerKAT", "SKA1mid", "SKA1MK", "iSKA1MK", "aSKA1MK", "SKA1MK_A0"]
names = ["SKA1MK",] #["MeerKAT", "SKA1mid", "SKA1MK"]
colours = ['#22AD1A', '#3399FF', '#ED7624']

# Fiducial value and plotting
fig = P.figure()
ax = fig.add_subplot(111)

for k in range(len(names)):
    root = "../output/" + names[k]

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
              'omegak', 'omegaDE', 'w0', 'wa', 'h', 'gamma', 'Mnu']
    #0:A, 1:b_HI, 2:sigma_NL, 3:omegak, 4:omegaDE, 5:w0, [6:wa,] 7:h, 8:gamma, 9:Mnu
    zfns = [1,]
    
    # Marginalise over omega_k
    F_eos, lbls = rf.combined_fisher_matrix( F_eos_list, 
                                                    #exclude=[2,4,5,6,7,8,9,12], 
                                                    exclude=[2,4,5,6,7,8,  12], 
                                                    expand=zfns, names=pnames)
    Finv = np.linalg.inv(F_eos) # Pre-invert, for efficiency
    px = rf.indexes_for_sampled_fns(7, zc.size, zfns) # x
    py = rf.indexes_for_sampled_fns(5, zc.size, zfns) # y
    
    """
    print "sigma(Mnu) =", np.sqrt( Finv[rf.indexes_for_sampled_fns(8, zc.size, [1,8]),rf.indexes_for_sampled_fns(8, zc.size, [1,8])] )
    sigma_mnu = np.sqrt( Finv[rf.indexes_for_sampled_fns(8, zc.size, [1,8]),rf.indexes_for_sampled_fns(8, zc.size, [1,8])] )
    
    P.subplot(111)
    P.plot(zc, sigma_mnu, 'r-', lw=1.5)
    P.plot(zc, sigma_mnu, 'r.', lw=1.5)
    P.axhline(0.04, color='b', ls='dotted', lw=1.5)
    P.ylabel(r"$\sigma(\Sigma m_\nu)$")
    P.xlabel("z")
    P.title(r"$\Sigma m_\nu|_\mathrm{fid.} = 0.1 \mathrm{eV}$")
    P.show()
    exit()
    
    #rf.plot_corrmat(Finv, lbls)
    """
    
    # Fix omega_k
    F_eos2, lbls2 = rf.combined_fisher_matrix( F_eos_list, 
                                                    exclude=[2,4,5,6,7,8,  9,12], 
                                                    expand=zfns, names=pnames)
    Finv2 = np.linalg.inv(F_eos2) # Pre-invert, for efficiency
    px2 = rf.indexes_for_sampled_fns(6, zc.size, zfns) # x
    py2 = rf.indexes_for_sampled_fns(4, zc.size, zfns) # y
    
    print "sigma(Mnu) =", np.sqrt( Finv[rf.indexes_for_sampled_fns(7, zc.size, zfns),rf.indexes_for_sampled_fns(7, zc.size, zfns)] )
    
    # Fiducial point
    x = rf.experiments.cosmo['gamma']
    y = rf.experiments.cosmo['w0']
    if k == 0: ax.plot(x, y, 'kx', ms=10)
    
    # Marginalise over omega_k
    w, h, ang, alpha = rf.ellipse_for_fisher_params(px[0], py[0], F_eos, Finv=Finv)
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                 height=alpha[kk]*h, angle=ang, fc='none', ec=colours[k], 
                 lw=2., alpha=0.9) for kk in range(0, 2)]
    for e in ellipses: ax.add_patch(e)
    
    # Fix omega_k
    w, h, ang, alpha = rf.ellipse_for_fisher_params(px2[0], py2[0], F_eos2, Finv=Finv2)
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                 height=alpha[kk]*h, angle=ang, fc=colours[k], ec=colours[k], 
                 lw=2., alpha=0.25) for kk in range(0, 2)]
    for e in ellipses: ax.add_patch(e)
    
    # Add error ellipse for Euclid (omega_k = 0)
    w, h, ang, alpha = rf.ellipse_for_fisher_params( 0, 1, None, 
                                            Finv=euclid.cov_gamma_w_ref )
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                 height=alpha[kk]*h, angle=ang, fc='b', ec='b', lw=2., 
                 alpha=0.15) for kk in range(0, 2)]
    for e in ellipses: ax.add_patch(e)
    
    # Ellipse for Euclid (omega_k marginalised)
    w, h, ang, alpha = rf.ellipse_for_fisher_params( 0, 1, None, 
                                            Finv=euclid.cov_gamma_w_okmarg_ref )
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                 height=alpha[kk]*h, angle=ang, fc='none', ec='b', lw=2., 
                 alpha=0.9) for kk in range(0, 2)]
    for e in ellipses: ax.add_patch(e)

#for i in range(len(names)):
#    l = matplotlib.lines.Line2D([0.,], [0.,], lw=2.5, color=colours[i])
#    lines.append(l)

labels = ['Euclid', 'Euclid ($\Omega_K=0$)', 'SKA+MK', 'SKA+MK ($\Omega_K=0$)']
lines = []
lines.append( matplotlib.lines.Line2D([0.,], [0.,], lw=2.5, color='b') )
lines.append( matplotlib.lines.Line2D([0.,], [0.,], lw=8.5, color='b', alpha=0.3) )
lines.append( matplotlib.lines.Line2D([0.,], [0.,], lw=2.5, color=colours[0]) )
lines.append( matplotlib.lines.Line2D([0.,], [0.,], lw=8.5, color=colours[0], alpha=0.3) )

ax.legend((l for l in lines), (lbl for lbl in labels), prop={'size':'x-large'})
# (0.75, 0.65)
#ax.legend()

ax.set_xlim((0.45, 0.651))
ax.set_ylim((-1.2, -0.80))

#ax.set_xlim((0.35, 0.751))
#ax.set_ylim((-1.51, -0.50))

ax.set_xlabel(r"$\gamma$", fontdict={'fontsize':'20'})
ax.set_ylabel(r"$w_0$", fontdict={'fontsize':'20'})

fontsize = 16.
for tick in ax.yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in ax.xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)

P.title(r"z=%3.3f - %3.3f ($\nu_\mathrm{max}=1400$ MHz)" % (np.min(zc)-0.05, np.max(zc)+0.05))
P.tight_layout()
P.show()
