#!/usr/bin/python
"""
Calculate Fisher matrix for a galaxy redshift survey, using the formalism 
from the Euclid cosmology white paper (arXiv:1206.1225; see Sect. 1.7.3).
"""
import numpy as np
import pylab as P
import scipy.integrate
import scipy.misc
import baofisher
import experiments
from units import *
import copy

RSD_FUNCTION = 'not kaiser'


def pk_galaxy(k, u, z, cosmo, expt):
    """
    P(k, mu) for galaxy redshift survey.
    """
    c = cosmo
    
    # Cosmological functions
    cosmo['z'] = z; cosmo['f'] = ff(z); cosmo['D'] = DD(z)
    cosmo['r'] = rr(z); cosmo['rnu'] = C*(1.+z)**2. / HH(z) # Perp/par. dist. scales
    cosmo['bHI'] = cosmo['bgal'] = expt['b(z)'](z)
    
    # Wavenumber and mu = cos(theta)
    u2 = u**2.
    
    # RSD function (bias 'btot' already includes scale-dep. bias/non-Gaussianity)
    if RSD_FUNCTION == 'kaiser':
        # Pedro's notes, Eq. 7
        Frsd = (c['bgal'] + c['f']*u2)**2. * np.exp(-u2*(k*c['sigma_nl'])**2.)
    else:
        # arXiv:0812.0419, Eq. 5
        sigma_nl2_eff = (c['D'] * c['sigma_nl'])**2. * (1. - u2 + u2*(1.+c['f'])**2.)
        Frsd = (c['bgal'] + c['f']*u2)**2. * np.exp(-0.5 * k**2. * sigma_nl2_eff)
    
    # Photometric redshift error (see e.g. Zhan & Knox 2006)
    Fphot = 1.
    
    # Construct signal covariance and return
    cs = Fphot * Frsd * (1. + c['A'] * c['fbao'](k)) * c['D']**2. * c['pk_nobao'](k)
    cs *= c['aperp']**2. * c['apar']
    return cs



# Load cosmology and experimental settings
cosmo = experiments.cosmo
expt = {
    'fsky':     0.242,    # Survey area, as fraction of the sky
    'kmin':     1e-4,   # Seo & Eisenstein say: shouldn't make much difference...
    'k_nl0':    0.14,   # 0.1 # Non-linear scale at z=0 (effective kmax)
    'use':      experiments.USE,
    'b(z)':     lambda z: 2.0 + z*0. # b(z) ~ 2.0 [arXiv:1010.4915] # FIXME
}
zmin, zmax, ngal = np.genfromtxt("boss_nz.dat").T
zc = 0.5 * (zmin + zmax)
#ngal *= 100. # FIXME
j = 0 # Redshift bin

print "zc:  ", zc[j]
print "ngal:", ngal[j]

# Precomputations
cosmo_fns = baofisher.background_evolution_splines(cosmo)
cosmo = baofisher.load_power_spectrum(cosmo, "cache_pk_gal.dat")
HH, rr, DD, ff = cosmo_fns


# Anisotropic P(k), with kernel roughly corresponding to alpha_par or 
# alpha_perp from BAO-only
k = np.logspace(np.log10(2e-3), np.log10(0.35), 1000)
w_bao = scipy.misc.derivative(cosmo['fbao'], k, 1e-3) / (1. + cosmo['fbao'](k))
integ_cv = k**2. * w_bao
F_cv = scipy.integrate.cumtrapz(integ_cv**2., k, initial=0.)

# Fisher matrix as a fn. of n(z)
# BOSS is ~ 1e-4, Euclid ~ 10-20e-4
ngals = np.array([0.5, 1., 2., 5., 10., 15., 20.]) * 1e-4
F_boss = []
for ngal in ngals:
    pk_par  = pk_galaxy(k, u=0., z=zc[j], cosmo=cosmo, expt=expt)
    integ = k**2. * pk_par*ngal / (1. + pk_par*ngal) * w_bao

    F_boss = scipy.integrate.cumtrapz(integ**2., k, initial=0.)
    P.plot(k, 1./np.sqrt(F_boss / F_cv), label="%3.1e"%ngal)


# Plot
#P.plot(k, 1./np.sqrt(F_boss / F_cv), 'b-')
#P.axhline(1./ngal[j], color='k')
#P.xscale('log')
#P.yscale('log')
P.ylim((0.9, 3.))
P.xlim((0., 0.35))
P.axhline(1., ls='solid', lw=2.5, color='k', alpha=0.3)
P.legend(loc='upper left', prop={'size':'small'})

P.show()
