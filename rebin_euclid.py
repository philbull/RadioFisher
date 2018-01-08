#!/usr/bin/python
"""
Re-bin Euclid n(z)
"""
import numpy as np
import pylab as P
import scipy.integrate
import radiofisher as rf

# Precompute cosmo fns.
cosmo_fns = rf.background_evolution_splines(rf.experiments.cosmo)
HH, rr, DD, ff = cosmo_fns

expt = rf.experiments_galaxy.EuclidRef

def vol(zmin, zmax):
    """
    Calculate volume of redshift bin.
    """
    C = 3e5
    _z = np.linspace(zmin, zmax, 1000)
    vol = C * scipy.integrate.simps(rr(_z)**2. / HH(_z), _z)
    vol *= 4. * np.pi
    return vol


# Load Euclid n(z) and b(z)
e = rf.experiments_galaxy.load_expt(expt)
nz = expt['nz']
zmin = expt['zmin']
zmax = expt['zmax']
zc = 0.5*(zmax + zmin)

# Get volume for each bin
v = np.array( [vol(zmin[i], zmax[i]) for i in range(zmin.size)] )

# Rebin in dz=0.3 bins
i = np.arange(zmin.size)
j = i // 3 # 3 small bins per big bin
print j

# Calculate new n(z) bins by volume-weighted averaging
new_nz = [ np.sum( nz[np.where(j==jj)] * v[np.where(j==jj)] )
           / np.sum( v[np.where(j==jj)] )
           for jj in range(np.max(j)+1) ]
new_nz = np.array(new_nz)

# Calculate new z bin edges
new_zmin = np.array([ np.min(zmin[np.where(j==jj)]) for jj in range(np.max(j)+1) ])
new_zmax = np.array([ np.max(zmax[np.where(j==jj)]) for jj in range(np.max(j)+1) ])
new_zc = 0.5 * (new_zmin + new_zmax)

for i in range(nz.size):
    print i, zmin[i], zmax[i], nz[i]
print "-"*50
for i in range(new_nz.size):
    print i, new_zmin[i], new_zmax[i], new_nz[i]
    
# Plot
P.subplot(111)
P.errorbar(zc, nz, xerr=0.5*(zmax-zmin), color='r', lw=1.5, marker='.', ls='none')
P.errorbar(new_zc, new_nz, xerr=0.5*(new_zmax-new_zmin), color='b', lw=1.5, marker='.', ls='none')
P.yscale('log')
P.show()
