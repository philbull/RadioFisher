#!/usr/bin/python
"""
Plot n(z) and b(z) for WEAVE surveys.
"""
import numpy as np
import pylab as P
import scipy.integrate
import radiofisher as rf

C = 3e5 # km/s

def vol(zmin, zmax, sarea=None):
    """
    Volume of survey.
    """
    _z = np.linspace(zmin, zmax, 100)
    vol = 4.*np.pi*C * scipy.integrate.simps(r(_z)**2. / H(_z), _z)
    fsky = sarea * (np.pi/180.)**2. if sarea is not None else 1.
    return fsky * vol
    
def nz_bins(z, N, bz, Sarea, zbins):
    """
    Binned number density, bias, and volume.
    """
    Nz = np.atleast_2d( N * Sarea * (z[1] - z[0]) ) # FIXME
    bz = np.atleast_2d(bz)
    
    # Calculate number density and volume
    NN = []; vv = []
    for i in range(zbins.size - 1):
        idxs = np.where(np.logical_and(z >= zbins[i], z < zbins[i+1]))
        NN += [np.sum(Nz[:,idxs]),]
        vv += [vol(zbins[i], zbins[i+1], sarea=Sarea),]
    nn = np.array(NN) / np.array(vv)
    
    # Calculate bias
    bb = []
    for i in range(zbins.size - 1):
        idxs = np.where(np.logical_and(z >= zbins[i], z < zbins[i+1]))
        bb += [np.sum(bz[:,idxs] * Nz[:,idxs]) / np.sum(Nz[:,idxs]),]
    return nn, bb, vv

# Calculate cosmo. functions
cosmo_fns = rf.background_evolution_splines(rf.experiments.cosmo)
H, r, D, f = cosmo_fns

# Surveys (email from M. Jarvis, 2014-07-01
# Tier 1: 150Mhz 0.1mJy 20,000sq.deg (wide)
# Tier 2: 150Mhz 0.025mJy 1000sq.deg (mid)
# Tier 3: 150 MHz 10uJy 70 sq.deg (in 10 fields) (deep)
S_deep = 70. * 10. # FIXME
S_mid  = 1e3
S_wide = 20e3


# 1: redshift (in bins of dz=0.01 up to z=25), 
# 2-6: N(z) in sources/deg^2/z
# [normal star-forming galaxies / starbursts / RQ AGN / FR-I AGN / FR-II AGN]
# 7-11: b(z) model for each population
d_deep = np.genfromtxt("LOFAR_Nz+bias_deep.dat").T
d_mid  = np.genfromtxt("LOFAR_Nz+bias_mid.dat").T
d_wide = np.genfromtxt("LOFAR_Nz+bias_wide.dat").T
z = d_deep[0]

# (1) WEAVE star-forming galaxy survey at z < 1.3 (H-alpha and [OII])
zbins_lz = np.arange(0., 1.31, 0.1)
zl = [0.5 * (zbins_lz[i] + zbins_lz[i+1]) for i in range(zbins_lz.size - 1)]
nd_low, bd_low, vd_low = nz_bins(z, d_deep[1], d_deep[6], S_deep, zbins_lz)
nm_low, bm_low, vm_low = nz_bins(z, d_mid[1],  d_mid[6],  S_mid,  zbins_lz)
nw_low, bw_low, vw_low = nz_bins(z, d_wide[1], d_wide[6], S_wide, zbins_lz)

# (2) WEAVE Lyman-alpha survey at z > 2 (all populations)
zbins_hz = np.arange(2., 5., 0.25)
zh = [0.5 * (zbins_hz[i] + zbins_hz[i+1]) for i in range(zbins_hz.size - 1)]
nd_high, bd_high, vd_high = nz_bins(z, d_deep[1:6], d_deep[6:], S_deep, zbins_hz)
nm_high, bm_high, vm_high = nz_bins(z, d_mid[1:6],  d_mid[6:],  S_mid,  zbins_hz)
nw_high, bw_high, vw_high = nz_bins(z, d_wide[1:6], d_wide[6:], S_wide, zbins_hz)


# Output binned survey info
fmt = "%2.2f  %2.2f  %2.2f   %3.3e   %6.3f  %6.2f   %5.3e"
columns = "\nzc    zmin  zmax  n(z)[Mpc^-3]  b(z)   V[Gpc^3] Ngal"

print "\nLyman-alpha Deep", columns
for i in range(len(zh)-1): print fmt % (zh[i], zbins_hz[i], zbins_hz[i+1], 
                                        nd_high[i], bd_high[i], vd_high[i]/1e9, 
                                        nd_high[i]*vd_high[i],)

print "\nLyman-alpha Mid", columns
for i in range(len(zh)-1): print fmt % (zh[i], zbins_hz[i], zbins_hz[i+1], 
                                        nm_high[i], bm_high[i], vm_high[i]/1e9, 
                                        nm_high[i]*vm_high[i],)

print "\nLyman-alpha Wide", columns
for i in range(len(zh)-1): print fmt % (zh[i], zbins_hz[i], zbins_hz[i+1], 
                                        nw_high[i], bw_high[i], vw_high[i]/1e9, 
                                        nw_high[i]*vw_high[i],)

print "\nStarforming Deep", columns
for i in range(len(zl)-1): print fmt % (zl[i], zbins_lz[i], zbins_lz[i+1], 
                                        nd_low[i], bd_low[i], vd_low[i]/1e9, 
                                        nd_low[i]*vd_low[i],)

print "\nStarforming Mid", columns
for i in range(len(zl)-1): print fmt % (zl[i], zbins_lz[i], zbins_lz[i+1], 
                                        nm_low[i], bm_low[i], vm_low[i]/1e9, 
                                        nm_low[i]*vm_low[i],)

print "\nStarforming Wide", columns
for i in range(len(zl)-1): print fmt % (zl[i], zbins_lz[i], zbins_lz[i+1], 
                                        nw_low[i], bw_low[i], vw_low[i]/1e9, 
                                        nw_low[i]*vw_low[i],)

# Plot n(z)
P.subplot(111)
P.plot(zh, nd_high, 'r-', lw=1.5)
P.plot(zh, nm_high, 'g-', lw=1.5)
P.plot(zh, nw_high, 'b-', lw=1.5)

P.plot(zl, nd_low, 'r--', lw=1.5)
P.plot(zl, nm_low, 'g--', lw=1.5)
P.plot(zl, nw_low, 'b--', lw=1.5)

P.ylabel("n(z) [Mpc^-3]")
P.yscale('log')
P.show()
