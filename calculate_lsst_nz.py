#!/usr/bin/python
"""
Generate LSST n(z), using k=1 case from Chang et al., arXiv:1305.0793 
(Tables 1, 2, and Eq. 21). [Suggested by Sarah Bridle.]
"""
import numpy as np
import pylab as P
import baofisher
from units import *
import experiments as e
import scipy.integrate

AMIN2RAD = np.pi / (180. * 60.) # 1 arcmin in radians

# Effective number density (scaling factor?), from Table 1 of Chang
zmax = 4. # Max. redshift to calculate out to
# Eff. no. density scaling, converted from arcmin^-2 -> full sky
neff = 37. * (4.*np.pi / AMIN2RAD**2.)
fsky = 18e3 / (4.*np.pi * (180./np.pi)**2.) # Sky fraction (LSST = 18,000 deg^2)

# Smail eqn. parameters. From k=1 case of Change, Table 2
alpha = 1.27
beta = 1.02
z0 = 0.50
zm = 0.82 # Median redshift

# Calculate background redshift evolution
cosmo_fns = baofisher.background_evolution_splines(e.cosmo)
H, r, D, f = cosmo_fns

def nz(z):
    """
    Smail eqn. for galaxy number density (Eq. 21 of Change et al.)
    """
    return neff * z**alpha * np.exp(-(z/z0)**beta)

def V(zmin, zmax):
    _z = np.linspace(zmin, zmax, 1000)
    Vsurvey = C * scipy.integrate.simps(r(_z)**2. / H(_z), _z)
    Vsurvey *= 4. * np.pi #* fsky
    #print "\tSurvey volume: %3.2f Gpc^3" % (Vsurvey/1e9)
    return Vsurvey

# Integrate over redshift to get total galaxy count
Ntot, err = scipy.integrate.quad(nz, 0., zmax)
print "Ntot = %3.2e" % (Ntot * fsky)
print "fsky = %3.3f" % fsky

# Define redshift bins
zedges = np.linspace(0., 3., 11)
zc = np.array( [0.5*(zedges[i+1] + zedges[i]) for i in range(zedges.size-1)] )
zmin = np.array([zedges[i] for i in range(zedges.size - 1)])
zmax = np.array([zedges[i+1] for i in range(zedges.size - 1)])

# Get galaxy density n(z) [Mpc^-3] in each redshift bin
nn = []
for i in range(zc.size):
    Ntot, err = scipy.integrate.quad(nz, zedges[i], zedges[i+1])
    dV = V(zedges[i], zedges[i+1])
    nn.append(Ntot / dV) # in Mpc^-3
nn = np.array(nn)

# Output n(z) [Mpc^-3]
ii = np.where(nn > 0.) # Keep only bins with n > 0
np.savetxt("lsst_nz.dat", np.column_stack((zmin[ii], zmax[ii], nn[ii])), header="zmin zmax n(z)[Mpc^-3]")
print "Saved to", "lsst_nz.dat"

exit()
# Plot n(z)
z = np.linspace(0., zmax, 1000)
n = nz(z)
P.subplot(111)
P.plot(z, n, 'r-', lw=1.5)
P.plot(zc, nn, 'bo', lw=1.5)

P.xlabel("z")
P.ylabel("n(z) [full sky]")
P.show()
