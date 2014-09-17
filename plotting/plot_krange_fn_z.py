#!/usr/bin/python
"""
OBSOLETE
"""
import numpy as np
import pylab as P
from rfwrapper import rf
from radiofisher import experiments as e
from units import *

expt = e.SKAMID
expt['Sarea'] /= 6.

cosmo_fns, cosmo = rf.precompute_for_fisher(e.cosmo, "camb/rf_matterpower.dat")
H, r, D, f = cosmo_fns

z = np.linspace(1e-2, 3., 300)
rnu = C*(1.+z)**2. / H(z) # Perp/par. dist. scales

Dmax = 100e3 # 100 km max. baseline
Dmin = 15. # 15m dish diameter (actually, this would give FOV, not Dmin)

# INTERFEROM.
kmin_int = 2.*np.pi * Dmin * (1420.0e6) / (3e8 * r(z) * (1.+z))
kmax_int = 2.*np.pi * Dmax * (1420.0e6) / (3e8 * r(z) * (1.+z))


# SINGLE-DISH

Vphys = expt['Sarea'] * (expt['survey_dnutot']/expt['nu_line']) * r(z)**2. * rnu
kmin = 2.*np.pi / Vphys**(1./3.)
kfg = 2.*np.pi * expt['nu_line'] / (0.5 * expt['survey_dnutot'] * rnu)

D0 = 0.5 * 1.22 * 300. / np.sqrt(2.*np.log(2.)) # Dish FWHM prefactor [metres]
sigma_kpar = np.sqrt(16.*np.log(2)) * expt['nu_line'] / (expt['dnu'] * rnu)
sigma_kperp = np.sqrt(2.) * expt['Ddish'] * expt['nu_line'] / (r(z) * D0 * (1. + z))

sigma_nl2_eff = (D(z) * cosmo['sigma_nl'])**2. * (1.+ f(z))**2.

k_rsd = 1. / cosmo['sigma_nl'] * np.ones(z.shape)
k_rsd2 = 1. / np.sqrt(sigma_nl2_eff)

kmax = np.sqrt(k_rsd**2. + sigma_kperp**2.)
kmax2 = np.sqrt(k_rsd2**2. + sigma_kperp**2.) # for alternative RSD fn.

kmin_sd = kfg
kmax_sd = kmax


# Plot results
P.subplot(111)
P.plot(kmin_int, z, 'r-', lw=1.2, label="SKAMID Interferom.")
P.plot(kmax_int, z, 'r-', lw=1.2)

P.plot(kmin_sd, z, 'b-', lw=1.2, label="SKAMID Dish")
#P.plot(k_rsd, z, 'b-', lw=1.5)
P.plot(kmax_sd, z, 'b-', lw=1.2)
P.plot(kmax2, z, 'b--', lw=1.2, label="SKAMID Dish (RSD evol.)")

kk = np.logspace(-4., 3., 1000)
pk = cosmo['pk_nobao'](kk) * (1. + cosmo['fbao'](kk))
P.plot(kk, 0.3*np.log10(pk / np.max(pk)) + 2.6, 'k-', lw=2., alpha=0.4)
#P.plot(kk, 10.*cosmo['fbao'](kk) + 2., 'k-', lw=2., alpha=0.4)

# RSD region
P.fill_betweenx(z, 2e-2*np.ones(z.shape), 4e-1*np.ones(z.shape), color='k', alpha=0.07)

P.fill_betweenx(z, kmin_int, kmax_int, color='r', alpha=0.1)
P.fill_betweenx(z, kmin_sd, kmax_sd, color='b', alpha=0.1)

P.legend(loc='upper right', prop={'size':'large'})

P.xscale('log')
P.xlim((1e-4, 1e3))
P.ylim((0., 3.))

P.xlabel("k [Mpc$^{-1}$]", fontdict={'fontsize':'18'})
P.ylabel("z", fontdict={'fontsize':'18'})

fontsize = 16.
for tick in P.gca().yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in P.gca().xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)

P.tight_layout()


P.show()
