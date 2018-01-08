#!/usr/bin/python
"""
Plot 1/P(k) as fn. of z
"""
import numpy as np
import pylab as P
import radiofisher as rf

# Load cosmo functions
cosmo_fns = rf.background_evolution_splines(rf.experiments.cosmo)
H, r, D, f = cosmo_fns

# Load P(k)
k, pk = np.genfromtxt("cache_pk.dat").T

# Load bias for SKA1 galaxy survey
#dat = np.genfromtxt("nz_SKA1MID_B2_Upd.dat").T
dat = np.genfromtxt("nz_SKA2MG.dat").T
#dat = np.genfromtxt("nz_wfirst.dat").T
zc = dat[0]
bz = dat[4]
#bz = 1.5 + 0.4*(zc - 1.5)


# Get range of 1/P(k) values

nz_upper = 1./ ( pk * (D(zc[0]) * bz[0])**2. )
nz_lower = 1./ ( pk * (D(zc[-1]) * bz[-1])**2. )
print zc[0], zc[-1]
print D(zc[0])*bz[0], D(zc[-1])*bz[-1]

# Plot n(z) ~ - / P(k)
P.subplot(111)

cols = ['k', 'r', 'b', 'g', 'y', 'c', 'm']

P.fill_between(k, nz_lower, nz_upper, color='k', alpha=0.2)

P.axvspan(2e-2, 3e-1, color='b', alpha=0.1)

P.xscale('log')
P.yscale('log')

P.tick_params(axis='both', which='major', labelsize=20, width=1.5, size=8., pad=10)
P.tick_params(axis='both', which='minor', labelsize=20, width=1.5, size=5.)

P.xlabel(r'$k$ $[{\rm Mpc}^{-1}]$', labelpad=10., fontdict={'fontsize':'xx-large'})
P.ylabel(r'$n(z)$ $[{\rm Mpc}^{-3}]$', labelpad=15., fontdict={'fontsize':'xx-large'})

P.xlim((1e-3, 1e0))
#P.ylim((5e-6, 1e-2))
P.ylim((1e-5, 1e-2))

P.tight_layout()
P.show()
