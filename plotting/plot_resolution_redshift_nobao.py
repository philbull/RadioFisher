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
kperp_max_sd = 2.*np.pi * Ddish / (r * l) # 16.*np.log(2.) / 1.22
kperp_max_sd2 = np.sqrt(16.*np.log(2.)) * Ddish / (r * l)

# Set-up plots
P.subplot(111)

# Interferom. transverse
P.plot(kperp_min_int, zz, lw=1.8, color='#1619A1')
P.plot(kperp_max_int, zz, lw=1.8, color='#1619A1')
P.fill_betweenx(zz, kperp_min_int, kperp_max_int, color='#B1C9FD', alpha=1.)

P.annotate("Int.", xy=(0.9, 0.8), xytext=(0., 0.), fontsize='xx-large', 
            textcoords='offset points', ha='center', va='center')

# Single-dish transverse
P.plot(kperp_min_sd, zz, lw=1.8, color='#CC0000')
P.plot(kperp_max_sd, zz, lw=1.8, color='#CC0000', ls='dashed')
#ax.plot(kperp_max_sd2, zz, lw=1.8, color='g')
P.fill_betweenx(zz, kperp_min_sd, kperp_max_sd, color='#F09B9B', alpha=0.7)

P.annotate("SD", xy=(0.01, 0.8), xytext=(0., 0.), fontsize='xx-large', 
            textcoords='offset points', ha='center', va='center')

# Horizon size
P.plot(k_hor, _z, color='#555555', lw=3., ls='solid')
P.fill_betweenx(_z, 1e-8, k_hor, color='#555555', alpha=0.3)

P.xscale('log')


P.xlim((2e-4, 1e2))
P.ylim((0., 2.5))

P.tick_params(axis='both', which='major', labelsize=20, size=8., width=1.5, pad=8.)
P.tick_params(axis='both', which='minor', size=4., width=1.5)

P.xlabel(r"$k_\perp \, [\mathrm{Mpc}^{-1}]$", fontdict={'fontsize':'xx-large'}, labelpad=8.)
P.ylabel(r"$z$", fontdict={'fontsize':'xx-large'}, labelpad=15.)


#P.tick_params(axis='both', which='major', size=8., width=1.5, pad=8., labelbottom='off', labelleft='off')
#P.tick_params(axis='both', which='minor', size=4., width=1.5)


# Set size and save
P.tight_layout()
P.gcf().set_size_inches(8.,7.)
P.savefig('ska-resolution-nobao.pdf', transparent=True)
P.show()
