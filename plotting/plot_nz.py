#!/usr/bin/python
"""
Plot 2D constraints on (w0, wa).
"""
import numpy as np
import pylab as P
import matplotlib.patches
import matplotlib.cm
import matplotlib.ticker
from units import *


cosmo = rf.experiments.cosmo
#colours = ['#CC0000', '#5B9C0A', '#1619A1',   '#990A9C', '#FAE300']

# Load n(z) data
zmin, zmax, n_opt, n_ref, n_pess = np.genfromtxt("euclid_nz.dat").T
ngal = np.array([n_opt, n_ref, n_pess]) * cosmo['h']**3. # Rescale to Mpc^-3 units
zc = 0.5 * (zmin + zmax)

print zc

# Add zeros before and after
zc = np.concatenate(([zmin[0]-0.3, zmin[0]-0.2, zmin[0]-0.1,], zmin, [zmin[-1]+0.1, zmin[-1]+0.2, zmin[-1]+0.3]))
nz = np.concatenate(([0., 0., 0.,], ngal[1], [0., 0., 0.]))
#nz = ngal[1]

# Plot n(z)
fig = P.figure()
ax = fig.add_subplot(111)
ax.plot(zc, nz*1e4, 'k-', lw=2., drawstyle="steps-post")

# Resize axis labels, ticks etc.
fontsize = 20
for tick in ax.xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in ax.yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)

xminorLocator = matplotlib.ticker.MultipleLocator(0.1)
yminorLocator = matplotlib.ticker.MultipleLocator(0.5)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)

ax.set_xlabel(r"$z$", fontdict={'fontsize':'20'}, labelpad=15.)
ax.set_ylabel(r"$n(z)$ $[10^{-4} \mathrm{Mpc}^{-3}]$", fontdict={'fontsize':'20'}, labelpad=15.)
ax.set_xlim((0.5, 2.2))
#ax.set_ylim((-2.6, 2.6))

P.figtext(0.7, 0.8, r"$N_\mathrm{gal} = 6 \times 10^{7}$", fontdict={'size':'xx-large'})

P.tight_layout()

# Set size and save
P.gcf().set_size_inches(10.,7.)
P.savefig("pub-nz.pdf")
P.show()
