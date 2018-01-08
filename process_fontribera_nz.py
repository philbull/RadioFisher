#!/usr/bin/python
"""
Calculate n(z) bins, in Mpc^-3, based on dN/dz/dSarea from Tables in 
Font-Ribera et al. (arXiv:1308.4164).
"""
import numpy as np
import pylab as P
import scipy.integrate
import baofisher
import experiments as e
from units import *

INFILE = "boss_dNdzdS.dat"
fname = "boss_nz.dat"

INFILE = "wfirst_dNdzdS.dat"
fname = "wfirst_nz.dat"

INFILE = "hetdex_dNdzdS.dat"
fname = "hetdex_nz.dat"

# Calculate background redshift evolution
cosmo_fns = baofisher.background_evolution_splines(e.cosmo)
H, r, D, f = cosmo_fns

def vol(zmin, zmax):
    _z = np.linspace(zmin, zmax, 1000)
    vol = C * scipy.integrate.simps(r(_z)**2. / H(_z), _z)
    vol *= 4. * np.pi * fsky
    return vol

# Load data
zmin, zmax, dN_dzdS = np.genfromtxt(INFILE).T
fsky = 1. #Sarea / (4.*np.pi * (180./np.pi)**2.)
zc = 0.5 * (zmin + zmax)
dz = zmax - zmin

# Calculate volume of bins
dV = np.array( [vol(zmin[i], zmax[i]) for i in range(zmin.size)] )

# Convert to number density
full_sky = 4.*np.pi * (180./np.pi)**2. # deg^2 in full sky
nz = (dN_dzdS * dz * full_sky) / dV

# Save to file
np.savetxt(fname, np.column_stack((zmin, zmax, nz)))
print "Saved n(z) file to %s." % fname

P.plot(zc, nz)
P.show()
