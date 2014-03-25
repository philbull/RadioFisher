#!/usr/bin/python
"""
Plot evolution of the resolution with redshift in both the single dish and 
interferometer cases.
"""
import numpy as np
import pylab as P
import baofisher
import matplotlib.patches
import matplotlib.cm
import matplotlib.ticker
import scipy.integrate
from units import *
from mpi4py import MPI
import experiments
import os
import euclid


cosmo = experiments.cosmo
#names = ['EuclidRef', 'cexptL', 'iexptM'] #, 'exptS']
#labels = ['DETF IV + Planck', 'Facility + Planck', 'Mature + Planck'] #, 'Snapshot']


colours = [ ['#CC0000', '#F09B9B'],
            ['#1619A1', '#B1C9FD'],
            ['#5B9C0A', '#BAE484'],
            ['#FFB928', '#FFEA28'] ]

# Define values for interferom./single-dish
Ddish = 15.
Dmin = 15.
Dmax = 1000.
dnutot = 600. / 1420. # Dimensionless
Sarea = 25e3 * (D2RAD)**2.


# Get r(z)
HH, rr, DD, ff = baofisher.background_evolution_splines(cosmo)
zz = np.linspace(0., 3., 1000)
r = rr(zz)
rnu = C*(1.+zz)**2. / HH(zz)
l = (1.+zz) * 3e8 / 1420e6 # metres


# Calculate horizon size
zeq = 3265. # From WMAP 9; horizon size is pretty insensitive to this
om = cosmo['omega_M_0']
ol = cosmo['omega_lambda_0']
h = cosmo['h']
orad = om / (1. + zeq)
aa = np.linspace(0., 1., 10000)
_z = 1./aa - 1.
integ = 1. / np.sqrt(om*aa + ol*aa**4. + orad)
rh = C / (100.*h) * scipy.integrate.cumtrapz(integ, aa, initial=0.)
k_hor = 2.*np.pi / rh
idxs = np.where(_z <= np.max(zz))
_z = _z[idxs]
k_hor = k_hor[idxs]

print "Horizon size today (z=0): %6.0f Mpc" % rh[-1]

# Get values
kpar_min = 2. * np.pi / (rnu * dnutot)
kpar_max = 1. / cosmo['sigma_nl'] * np.ones(zz.shape)
kperp_min_int = 2.*np.pi * Dmin / (r * l)
kperp_max_int = 2.*np.pi * Dmax / (r * l)
kperp_min_sd = 2.*np.pi / np.sqrt(r**2. * Sarea)
kperp_max_sd = 16.*np.log(2.) / 1.22 * Ddish / (r * l)

# Fiducial value and plotting
fig = P.figure()
ax2 = fig.add_subplot(211)
ax = fig.add_subplot(212)


# Interferom. transverse
ax.plot(kperp_min_int, zz, lw=1.8, color='#1619A1')
ax.plot(kperp_max_int, zz, lw=1.8, color='#1619A1')
ax.fill_betweenx(zz, kperp_min_int, kperp_max_int, color='#B1C9FD', alpha=1.)

ax.annotate("Int.", xy=(0.9, 0.8), xytext=(0., 0.), fontsize='xx-large', 
            textcoords='offset points', ha='center', va='center')

# Single-dish transverse
ax.plot(kperp_min_sd, zz, lw=1.8, color='#CC0000')
ax.plot(kperp_max_sd, zz, lw=1.8, color='#CC0000')
ax.fill_betweenx(zz, kperp_min_sd, kperp_max_sd, color='#F09B9B', alpha=0.7)

ax.annotate("SD", xy=(0.01, 0.8), xytext=(0., 0.), fontsize='xx-large', 
            textcoords='offset points', ha='center', va='center')

# Radial
##ax.plot(kpar_min, zz, lw=2.8, color='k', ls='dashed')
##ax.plot(kpar_max, zz, lw=2.8, color='k', ls='dashed')
#ax.fill_betweenx(zz, kpar_min, kpar_max, color='y', alpha=0.5)

# Horizon size
ax.plot(k_hor, _z, color='#555555', lw=3., ls='solid')
ax.fill_betweenx(_z, 1e-8, k_hor, color='#555555', alpha=0.3)

ax.set_xscale('log')


############################
# Get f_bao(k) function
cosmo_fns = baofisher.background_evolution_splines(cosmo)
cosmo = baofisher.load_power_spectrum(cosmo, "cache_pk.dat", force_load=True)
fbao = cosmo['fbao']

kk = np.logspace(-4., 2., 1000)
ax2.plot(kk, fbao(kk), 'k-', lw=1.8)
ax2.set_xscale('log')
ax2.set_xlim((2e-4, 1e2))

# BAO scales
ax2.axvline(0.023, color='k', lw=2.8, ls='dashed', alpha=0.4)
ax2.axvline(0.39, color='k', lw=2.8, ls='dashed', alpha=0.4)

ax.axvline(0.023, color='k', lw=2.8, ls='dashed', alpha=0.4)
ax.axvline(0.39, color='k', lw=2.8, ls='dashed', alpha=0.4)

############################

"""
# Legend
labels = [labels[k] for k in range(len(labels))]
lines = [ matplotlib.lines.Line2D([0.,], [0.,], lw=8.5, color=colours[k][0], alpha=0.65) for k in range(len(labels))]

P.gcf().legend((l for l in lines), (name for name in labels), prop={'size':'medium'}, bbox_to_anchor=[0.95, 0.95])

ax.tick_params(axis='both', which='major', labelsize=20, size=8., width=1.5, pad=8.)
xminorLocator = matplotlib.ticker.MultipleLocator(0.1)
yminorLocator = matplotlib.ticker.MultipleLocator(0.5)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)

"""

ax.set_xlim((2e-4, 1e2))
ax.set_ylim((0., 2.5))

ax.tick_params(axis='both', which='major', labelsize=20, size=8., width=1.5, pad=8.)
ax.tick_params(axis='both', which='minor', size=4., width=1.5)

ax.set_xlabel(r"$k_\perp \, [\mathrm{Mpc}^{-1}]$", fontdict={'fontsize':'xx-large'}, labelpad=8.)
ax.set_ylabel(r"$z$", fontdict={'fontsize':'xx-large'}, labelpad=15.)


ax2.tick_params(axis='both', which='major', size=8., width=1.5, pad=8., labelbottom='off', labelleft='off')
ax2.tick_params(axis='both', which='minor', size=4., width=1.5)


# Set positions
ax.set_position([0.15, 0.17, 0.8, 0.58])
ax2.set_position([0.15, 0.75, 0.8, 0.2])



# Set size and save
#P.tight_layout()
P.gcf().set_size_inches(8.,7.)
#P.savefig('pub-resolution-z.pdf', transparent=True)
P.savefig('pub-resolution-bao.pdf', transparent=True)
P.show()
