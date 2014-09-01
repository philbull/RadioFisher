#!/usr/bin/python
"""
Calculate n(z), in Mpc^-3, for a given galaxy redshift survey.
"""
import numpy as np
import pylab as P
import scipy.interpolate
import scipy.integrate
import baofisher
from units import *
import experiments as e
import scipy.integrate
import sys

# SKA HI galaxy survey fitting fn. coeffs.
Srms = [7.3, 70., 100., 150.]
c1 = [6.76, 5.62, 5.63, 5.48]
c2 = [2.14, 1.11, 1.41, 1.33]
c3 = [7.36, 13.03, 15.49, 16.62]

try:
    j = int(sys.argv[1])
    Sarea = float(sys.argv[2])
except:
    print "Need to specify: int(idx), Sarea(deg^2)"
    sys.exit(1)

fsky = Sarea / (4.*np.pi * (180./np.pi)**2.)

fname = "ska_hi_nz_dz02_%d_%d.dat" % (Srms[j], Sarea)

print "-"*50
print "S_rms:  %3.1f" % Srms[j]
print "S_area: %d" % Sarea
print "f_sky:  %3.3f" % fsky
print "-"*50

# Calculate background redshift evolution
cosmo_fns = baofisher.background_evolution_splines(e.cosmo)
H, r, D, f = cosmo_fns


def dndz(z):
    return 10.**c1[j] * z**c2[j] * np.exp(-c3[j]*z) # deg^-2

def vol(zmin, zmax):
    _z = np.linspace(zmin, zmax, 1000)
    vol = C * scipy.integrate.simps(r(_z)**2. / H(_z), _z)
    vol *= 4. * np.pi * fsky
    return vol

def n_bin(zmin, zmax):
    """
    Number density of galaxies in a given z bin.
    """
    _z = np.linspace(zmin, zmax, 1000)
    vol = C * scipy.integrate.simps(r(_z)**2. / H(_z), _z)
    vol *= 4. * np.pi * fsky
    
    N_bin = scipy.integrate.simps(dndz(_z), _z) * Sarea
    nz = N_bin / vol
    return nz

# Get binned n(z)
zbins = np.linspace(0., 3., 31) # dz=0.1
zc = np.array([0.5 * (zbins[i] + zbins[i+1]) for i in range(zbins.size-1)])
zmin = zbins[:-1]; zmax = zbins[1:]
nz = np.array([n_bin(zbins[i], zbins[i+1]) for i in range(zbins.size-1)])

# Sum to get total N
N = 0
Ngal = []
for i in range(zbins.size-1):
    v = vol(zbins[i], zbins[i+1])
    N += v * nz[i]
    Ngal.append(int(v * nz[i]))
    print "\t%3.3f: %d" % (zc[i], v*nz[i])
Ngal = np.array(Ngal)
print "N_tot = %5.5e (calc.)" % N

# Integrate to get total N
zz = np.linspace(0., 2., 1000)
Ntot = scipy.integrate.simps(dndz(zz), zz) * Sarea
print "N_tot = %5.5e" % Ntot

# Go through bins and output n(z) [Mpc^-3]
ii = np.where(Ngal > 1000) # Keep only bins with N_gal > 1000 #n > 0
np.savetxt(fname, np.column_stack((zmin[ii], zmax[ii], nz[ii])))
print "Saved n(z) file to %s." % fname

