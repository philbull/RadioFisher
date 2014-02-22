#!/usr/bin/python
"""
Plot how n(u) and noise change for different experiments.
"""

import numpy as np
import pylab as P
import matplotlib.cm
import baofisher
from experiments import *

INF_NOISE = 1e250

exptL
exptM

# Set-up redshift bin
dz = 0.1
zz = np.linspace(0.1, 3., 200)
kperp = np.logspace(np.log10(5e-3), 1., 1000)

# Calculate cosmological functions
#H, r, D, f = baofisher.background_evolution_splines(cosmo)

cosmo_fns, cosmo = baofisher.precompute_for_fisher(cosmo, "camb/baofisher_matterpower.dat")
H, r, D, f = cosmo_fns
fbao = cosmo['fbao']

def calc_dnutot(expt, zmin, zmax):
    """
    Calculate survey redshift bounds, central redshift, and total bandwidth
    """
    numin = expt['nu_line'] / (1. + zmax)
    numax = expt['nu_line'] / (1. + zmin)
    return numax - numin


def base_noise_level(expt, z, Tinst=None):
    """
    Instrumental noise only (mK)
    """
    dnutot = calc_dnutot(expt, z - 0.5*dz, z + 0.5*dz)
    Vsurvey = expt['Sarea'] * dnutot
    Tsky = 60e3 * (300.*(1. + z)/expt['nu_line'])**2.55 # Foreground sky signal (mK)
    if Tinst is not None:
        Tsys = Tinst + Tsky
    else:
        Tsys = expt['Tinst'] + Tsky
    noise = Tsys**2. * Vsurvey / (expt['ttot'] * dnutot)
    return noise

################################################################################

def I_dish(expt, z):
    """
    Multiplicity factor for single dish.
    """
    I = 1. / (expt['Ndish'] * expt['Nbeam'])
    return I * np.ones(kperp.size)


def I_nx(expt, z):
    """
    Multiplicity factor for interferometer with n(x) provided.
    """
    # Mario's interferometer noise calculation
    u = kperp * r(z) / (2. * np.pi) # UV plane: |u| = d / lambda
    nu = expt['nu_line'] / (1. + z)
    fov = (1.02 / (nu * expt['Ddish']) * (3e8 / 1e6))**2.
    
    # Rescale n(x) with freq.-dependence
    x = u / nu  # x = u / (freq [MHz])
    n_u = expt['n(x)'](x) / nu**2. # n(x) = n(u) * nu^2
    n_u[np.where(n_u == 0.)] = 1. / INF_NOISE
    
    I = 4./9. * fov / n_u
    return I


def I_int(expt, z):
    """
    Approximate expression for n(u), assuming uniform density in UV plane
    """
    u = kperp * r(z) / (2. * np.pi) # UV plane: |u| = d / lambda
    nu = expt['nu_line'] / (1. + z)
    fov = (1.02 / (nu * expt['Ddish']) * (3e8 / 1e6))**2.
    
    l = 3e8 / (nu * 1e6) # Wavelength (m)
    u_min = expt['Dmin'] / l
    u_max = expt['Dmax'] / l
    
    # New calc.
    n_u = expt['Ndish']*(expt['Ndish'] - 1.) * l**2. * np.ones(u.shape) \
          / (2. * np.pi * (expt['Dmax']**2. - expt['Dmin']**2.) )
    n_u[np.where(u < u_min)] = 1. / INF_NOISE
    n_u[np.where(u > u_max)] = 1. / INF_NOISE
    
    # Interferometer multiplicity factor, /I/
    I = 4./9. * fov / n_u
    return I

################################################################################


def packing(expt):
    """
    Packing efficiency of array, N_dish * A_dish / A_array
    """
    return expt['Ndish'] * (expt['Ddish'] / expt['Dmax'])**2.

"""
P.subplot(111)
nL = base_noise_level(exptL, z)
nM = base_noise_level(exptM, z)
nSky = base_noise_level(exptM, z, Tinst=0.)

P.plot(z, nL / nL[0], 'b-', lw=1.8)
P.plot(z, nM / nM[0], 'r-', lw=1.8)
#P.plot(z, nSky / nSky[0], 'k-', lw=1.8)

#P.plot(z, nL / (nL + nM), 'b-', lw=1.8)
#P.plot(z, nM / (nL + nM), 'r-', lw=1.8)
"""



P.subplot(211)
cmap2 = matplotlib.cm.autumn
cmap1 = matplotlib.cm.winter

print exptL
print "\n\n"
print exptM

zz = 1.
print "FACTOR:", np.min(I_int(exptM, zz)) / np.min(I_nx(exptL, zz))


for zz in np.linspace(0., 3., 4):
    P.plot(kperp, I_nx(exptL, zz), color=cmap1(zz/3.))
    P.plot(kperp, 7.*I_int(exptM, zz), color=cmap2(zz/3.))

#P.plot(kperp, I_nx(exptM, z))
#P.plot(kperp, I_int(exptM, z))

"""
print "Dmax", exptL['Dmax']
exptL['Dmax'] = 213.5
print "Packing L:", packing(exptL)
print "Packing M:", packing(exptM)

ff = 1. # Packing factor, dense array
print "Ddish:", np.sqrt(ff / exptL['Ndish']) * 100.
print "Dmax:", np.sqrt(exptL['Ndish'] / ff) * 13.5
exit()
"""

P.xscale('log')
P.yscale('log')
#P.ylim((1e-3, 1e1))
P.xlim((8e-3, 5e0))
P.ylim((1e-3, 5e0))


P.subplot(212)
P.plot(kperp, fbao(kperp), 'k', lw=3., alpha=0.3)
P.xscale('log')
P.xlim((8e-3, 5e0))

P.show()
