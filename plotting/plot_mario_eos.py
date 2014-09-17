#!/usr/bin/python
"""
Process EOS Fisher matrices and plot P(k).
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

#names = ["GBT", "BINGO", "WSRT", "APERTIF", "JVLA", "ASKAP", "KAT7", "MeerKAT", "SKA1", "SKAMID", "SKAMID_COMP", "iSKAMID", "iSKAMID_COMP", "SKA1_CV"]
#names = ["xSKA", "xSKA_extended"]  #"SKA1"] # "SKA1"] # , "iSKAMID_COMP"]
#names = ['SKAMID', 'iSKAMID']
names = ["exptL", "cexptL"] #["exptS", "iexptM", 

cols = ['r', 'g', 'c']
colours = ['#22AD1A', '#3399FF', '#ED7624']

cosmo_fns, cosmo = rf.precompute_for_fisher(rf.experiments.cosmo, "camb/rf_matterpower.dat")
H, r, D, f = cosmo_fns


# Fiducial value and plotting
fig = P.figure()
ax = fig.add_subplot(111)

for k in range(len(names)):
    root = "../output/" + names[k]

    # Load cosmo fns.
    dat = np.atleast_2d( np.genfromtxt(root+"-cosmofns-zc.dat") ).T
    zc, Hc, dAc, Dc, fc = dat
    zs, Hs, dAs, Ds, fs = np.genfromtxt(root+"-cosmofns-smooth.dat").T
    kc = np.genfromtxt(root+"-fisher-kc.dat").T
    
    print "z-range:", zc.size, np.min(zc), np.max(zc)
    
    # Load Fisher matrices as fn. of z
    Nbins = zc.size
    F_list = [np.genfromtxt(root+"-fisher-full-%d.dat" % i) for i in range(Nbins)]
    
    # EOS FISHER MATRIX
    # Actually, (aperp, apar) are (D_A, H)
    pnames = ['A', 'b_HI', 'Tb', 'sigma_NL', 'sigma8', 'n_s', 'f', 'aperp', 'apar', 
             'omegak', 'omegaDE', 'w0', 'wa', 'h', 'gamma', 'Mnu']
    pnames += ["pk%d" % i for i in range(kc.size)]
    
    zfns = []
    excl = [2,4,5, 6,7,8,  9, 14,] #15 # FLAT
    excl += [i for i in range(len(pnames)) if "pk" in pnames[i]]
    
    F, lbls = rf.combined_fisher_matrix( F_list,
                                                expand=zfns, names=pnames,
                                                exclude=excl )
    # Add Planck prior
    if len(zfns) == 0:
        l = 0
    else:
        l = Nbins - 1
    Fpl = F.copy()
    lbls_pl = ['omega_k', 'omega_DE', 'w0', 'wa']
    for i in range(1,4):
      for j in range(1,4):
        Fpl[l+2+i,l+2+j] += euclid.planck_prior[i,j]
        #print lbls[l+2+i], lbls_pl[i], lbls[l+2+j], lbls_pl[j]
    
    print "-"*50
    print names[k]
    print "-"*50
    
    
    cov = np.linalg.inv(F)
    cov_pl = np.linalg.inv(Fpl)
    for j in range(len(lbls)):
        print "%10s    %3.4f  %3.4f  %3.4f" % (lbls[j], np.sqrt(cov[j,j]), np.sqrt(cov_pl[j,j]), 1./np.sqrt(np.diag(F))[j])
    
    # Invert matrices
    cov = np.linalg.inv(F)
    cov_pl = np.linalg.inv(Fpl)
    
    # Indices of fns. of z
    pw0 = rf.indexes_for_sampled_fns(4, zc.size, zfns)
    pwa = rf.indexes_for_sampled_fns(5, zc.size, zfns)
    
    # Fiducial point
    x = rf.experiments.cosmo['w0']
    y = rf.experiments.cosmo['wa']
    if k == 0: ax.plot(x, y, 'kx', ms=10)
    
    """
    # omega_k, gamma fixed
    w, h, ang, alpha = rf.ellipse_for_fisher_params(pw0, pwa, cov, Finv=cov)
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                 height=alpha[kk]*h, angle=ang, fc='none', ec=colours[k], 
                 lw=2.5, alpha=1.) for kk in range(0, 2)]
    for e in ellipses: ax.add_patch(e)
    """
    
    # omega_k, gamma fixed, plus Planck prior
    w, h, ang, alpha = rf.ellipse_for_fisher_params(pw0, pwa, cov, Finv=cov_pl)
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                 height=alpha[kk]*h, angle=ang, fc='none', ec=colours[k], 
                 lw=2.5, alpha=1.) for kk in range(0, 2)]
    for e in ellipses: ax.add_patch(e)
    
    """
    # Fix omega_k
    w, h, ang, alpha = rf.ellipse_for_fisher_params(px2[0], py2[0], F_eos2, Finv=Finv2)
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                 height=alpha[kk]*h, angle=ang, fc=colours[k], ec=colours[k], 
                 lw=2., alpha=0.25) for kk in range(0, 2)]
    for e in ellipses: ax.add_patch(e)
    """
    
    """
    # Ellipse for Euclid (omega_k marginalised)
    w, h, ang, alpha = rf.ellipse_for_fisher_params( 0, 1, None, 
                                            Finv=euclid.cov_gamma_w_okmarg_ref )
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                 height=alpha[kk]*h, angle=ang, fc='none', ec='b', lw=2., 
                 alpha=0.9) for kk in range(0, 2)]
    for e in ellipses: ax.add_patch(e)
    """

# Add error ellipse for Euclid (omega_k = 0)
w, h, ang, alpha = rf.ellipse_for_fisher_params( 0, 1, None, 
                                        Finv=euclid.cov_w0_wa_fixed_gamma_ok_ref )
ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
             height=alpha[kk]*h, angle=ang, fc='m', ec='m', lw=2., 
             alpha=0.15) for kk in range(0, 2)]
for e in ellipses: ax.add_patch(e)


labels = ["Behemoth (SD) + Planck", "Behemoth (SD+int) + Planck", "Euclid"]
lines = []
lines.append( matplotlib.lines.Line2D([0.,], [0.,], lw=2.5, color=colours[0]) )
lines.append( matplotlib.lines.Line2D([0.,], [0.,], lw=2.5, color=colours[1]) )
lines.append( matplotlib.lines.Line2D([0.,], [0.,], lw=8.5, color='m', alpha=0.3) )
ax.legend((l for l in lines), (lbl for lbl in labels), prop={'size':'x-large'})
ax.legend(loc='upper left', prop={'size':'x-large'})

P.title("$\Omega_K, \gamma \,\,\mathrm{fixed}$", fontdict={'fontsize':'20'})

ax.plot(x, y, 'kx')
#ax1.set_xlim((0.4, 1.52))
#ax1.set_ylim((0.68, 0.96))


fontsize = 20
for tick in ax.xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in ax.yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)

ax.set_xlabel(r"$w_0$", fontdict={'fontsize':'20'})
ax.set_ylabel(r"$w_a$", fontdict={'fontsize':'20'})

# Set size and save
P.gcf().set_size_inches(16.5,10.5)
P.savefig('mario-w0wa-planck.png', dpi=100)

P.show()
