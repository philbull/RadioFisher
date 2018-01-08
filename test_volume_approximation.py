#!/usr/bin/python
"""
Check how good the volume approximation is in the Fisher code.
"""
import numpy as np
import pylab as P
import radiofisher as rf
import scipy.integrate

cosmo = rf.experiments.cosmo
cosmo_fns = rf.background_evolution_splines(cosmo)
H, r, D, f = cosmo_fns
C = 3e5 # km/s
nu_21 = 1420.


def V_approx(zmin, zmax):
    """
    Volume using approximate formula.
    """
    numin = nu_21 / (1. + zmax)
    numax = nu_21 / (1. + zmin)
    dnutot = numax - numin
    zc = 0.5 * (zmax + zmin)
    
    rnu = C * (1.+zc)**2. / H(zc)
    V = (dnutot/nu_21) * r(zc)**2. * rnu
    return V/1e9


def V_exact(zmin, zmax):
    """
    Volume using exact integration.
    """
    _z = np.linspace(zmin, zmax, 1000)
    V = C * scipy.integrate.simps(r(_z)**2. / H(_z), _z)
    return V/1e9







Va = V_exact(0.9, 1.1)
Ve = V_exact(0.85, 1.05)
print "V(approx): %3.3f Gpc^3" % Va
print "V(exact):  %3.3f Gpc^3" % Ve
print "Frac.diff: %3.3f" % ((Va - Ve) / Ve)
