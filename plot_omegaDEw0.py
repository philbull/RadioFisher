#!/usr/bin/python
"""
Plot 2D constraints on (w0, Omega_DE).
"""
import numpy as np
import pylab as P
import baofisher
import matplotlib.patches
import matplotlib.cm
import matplotlib.ticker
from units import *
from mpi4py import MPI
import experiments
import os
import euclid

cosmo = experiments.cosmo

names = ["cexptL", "iexptM", "exptS"]
colours = ['#CC0000', '#ED5F21', '#FAE300', '#5B9C0A', '#1619A1', '#56129F', '#990A9C', 'y']
colours = ['#CC0000', '#5B9C0A', '#1619A1',   '#990A9C', '#FAE300']
labels = ['Behemoth', 'Mature', 'Snapshot']


colours = [ ['#CC0000', '#F09B9B'],
            ['#5B9C0A', '#BAE484'],
            ['#1619A1', '#B1C9FD'] ]

cosmo_fns, cosmo = baofisher.precompute_for_fisher(experiments.cosmo, "camb/baofisher_matterpower.dat")
H, r, D, f = cosmo_fns

# Fiducial value and plotting
fig = P.figure()
ax = fig.add_subplot(111)

_k = range(len(names))[::-1]
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
    # Actually, (aperp, apar) are (D_A, H)
    pnames = ['A', 'b_HI', 'Tb', 'sigma_NL', 'sigma8', 'n_s', 'f', 'aperp', 'apar', 
             'omegak', 'omegaDE', 'w0', 'wa', 'h', 'gamma', 'Mnu']
    pnames += ["pk%d" % i for i in range(kc.size)]
    
    zfns = [1,]
    excl = [2,4,5,  6,7,8, 14,15] # omega_k free
    excl  += [i for i in range(len(pnames)) if "pk" in pnames[i]]
    
    F, lbls = baofisher.combined_fisher_matrix( F_list,
                                                expand=zfns, names=pnames,
                                                exclude=excl )
    # Add Planck prior
    Fpl = euclid.add_planck_prior(F, lbls, info=False)
    
    print "-"*50
    print names[k]
    print "-"*50
    
    # Invert matrices
    pw0 = baofisher.indexes_for_sampled_fns(5, zc.size, zfns)
    pDE = baofisher.indexes_for_sampled_fns(4, zc.size, zfns)
    cov_pl = np.linalg.inv(Fpl)
    
    fom = baofisher.figure_of_merit(pw0, pDE, None, cov=cov_pl)
    print "%s: FOM = %3.2f" % (names[k], fom)
    
    x = experiments.cosmo['w0']
    y = experiments.cosmo['omega_lambda_0']
    
    # Plot contours for w0, wa; omega_k free
    w, h, ang, alpha = baofisher.ellipse_for_fisher_params(pw0, pDE, None, Finv=cov_pl)
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                height=alpha[kk]*h, angle=ang, fc=colours[k][kk], 
                ec=colours[k][0], lw=1.5, alpha=1.) for kk in [1,0]]
    for e in ellipses: ax.add_patch(e)
    
    # Centroid
    ax.plot(x, y, 'kx')


# Legend
labels = [labels[k] + " + Planck" for k in range(len(labels))]
lines = [ matplotlib.lines.Line2D([0.,], [0.,], lw=8.5, color=colours[k][0], alpha=0.65) for k in range(len(labels))]

P.gcf().legend((l for l in lines), (name for name in labels), loc='upper right', prop={'size':'large'})

fontsize = 20
for tick in ax.xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in ax.yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)

xminorLocator = matplotlib.ticker.MultipleLocator(0.1)
yminorLocator = matplotlib.ticker.MultipleLocator(0.5)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)

ax.set_xlabel(r"$w_0$", fontdict={'fontsize':'20'})
ax.set_ylabel(r"$\Omega_\mathrm{DE}$", fontdict={'fontsize':'20'})
ax.set_xlim((-2.3, 0.3))
ax.set_ylim((0.68, 0.8))

P.tight_layout()

# Set size and save
#P.gcf().set_size_inches(16.5,10.5)
P.savefig('pub-w0omegaDE.png', dpi=100)

P.show()