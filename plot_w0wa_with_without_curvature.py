#!/usr/bin/python
"""
Plot 2D constraints on (w0, wa) with and without curvature free
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

k = 1 # Which experiment to plot
names = ["exptS", "iexptM", "cexptL"]
colours = ['#CC0000', '#ED5F21', '#FAE300', '#5B9C0A', '#1619A1', '#56129F', '#990A9C', 'y']
labels = ['Snapshot', 'Mature', 'Behemoth']

cosmo_fns, cosmo = baofisher.precompute_for_fisher(experiments.cosmo, "camb/baofisher_matterpower.dat")
H, r, D, f = cosmo_fns

# Fiducial value and plotting
fig = P.figure()
ax = fig.add_subplot(111)
m = 0

for k in [k,]:
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
    excl = [2,4,5,  6,7,8,  14,15] # omega_k free
    excl2 = [2,4,5,  6,7,8, 9, 14,15] # omega_k fixed
    
    excl  += [i for i in range(len(pnames)) if "pk" in pnames[i]]
    excl2 += [i for i in range(len(pnames)) if "pk" in pnames[i]]
    
    F, lbls = baofisher.combined_fisher_matrix( F_list,
                                                expand=zfns, names=pnames,
                                                exclude=excl )
    F2, lbls2 = baofisher.combined_fisher_matrix( F_list,
                                                expand=zfns, names=pnames,
                                                exclude=excl2 )
    # Add Planck prior
    Fpl = euclid.add_planck_prior(F, lbls, info=False)
    Fpl2 = euclid.add_planck_prior(F2, lbls2, info=False)
    
    print "-"*50
    print names[k]
    print "-"*50
    
    # Invert matrices
    pw0 = baofisher.indexes_for_sampled_fns(5, zc.size, zfns)
    pwa = baofisher.indexes_for_sampled_fns(6, zc.size, zfns)
    cov_pl = np.linalg.inv(Fpl)
    
    pw02 = baofisher.indexes_for_sampled_fns(4, zc.size, zfns)
    pwa2 = baofisher.indexes_for_sampled_fns(5, zc.size, zfns)
    cov_pl2 = np.linalg.inv(Fpl2)
    
    
    fom = baofisher.figure_of_merit(pw0, pwa, None, cov=cov_pl)
    print "Curvature FOM = %3.2f" % fom
    fom = baofisher.figure_of_merit(pw02, pwa2, None, cov=cov_pl2)
    print "No Curv. FOM = %3.2f" % fom
    
    x = experiments.cosmo['w0']
    y = experiments.cosmo['wa']
    
    # Plot contours for w0, wa; omega_k free
    w, h, ang, alpha = baofisher.ellipse_for_fisher_params(pw0, pwa, None, Finv=cov_pl)
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                 height=alpha[kk]*h, angle=ang, fc=colours[0], ec='none', lw=0., alpha=0.18) for kk in range(0, 2)]
    for e in ellipses: ax.add_patch(e)
    
    # Plot contours for w0, wa; omega_k fixed
    w, h, ang, alpha = baofisher.ellipse_for_fisher_params(pw02, pwa2, None, Finv=cov_pl2)
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                 height=alpha[kk]*h, angle=ang, fc='none', ec=colours[4], lw=2., alpha=1.0) for kk in range(0, 2)]
    for e in ellipses: ax.add_patch(e)
    
    # Centroid
    ax.plot(x, y, 'kx')

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

# Legend
labels = [labels[k] + " + Planck ($\Omega_K$ free)", labels[k] + " + Planck ($\Omega_K$ fixed)"]
lines = []
lines.append( matplotlib.lines.Line2D([0.,], [0.,], lw=8.5, color=colours[0], alpha=0.4) )
lines.append( matplotlib.lines.Line2D([0.,], [0.,], lw=2.5, color=colours[4]) )

P.gcf().legend((l for l in lines), (name for name in labels), loc='upper right', prop={'size':'large'})


fontsize = 20
for tick in ax.xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in ax.yaxis.get_major_ticks():
  #tick.label1.set_visible(False)
  tick.label1.set_fontsize(fontsize)

ax.set_xlabel(r"$w_0$", fontdict={'fontsize':'20'})
ax.set_ylabel(r"$w_a$", fontdict={'fontsize':'20'})

#ax.set_ylim((0., 6.))
"""
if MARGINALISE_W0WA:
    ax.set_xlim((-1.01e-1, 1.01e-1))
else:
    ax.set_xlim((-8.1e-3, 8.1e-3))
"""

P.tight_layout()

# Set size and save
#P.gcf().set_size_inches(16.5,10.5)
##P.savefig('pub-w0wa-omegakfixed-'+names[k]+'.png', dpi=100)

P.show()
