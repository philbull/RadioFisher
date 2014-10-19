#!/usr/bin/python
"""
Calculate and plot the comoving volumes of some surveys.
"""
import numpy as np
import pylab as P
import scipy.integrate
import scipy.interpolate
from rfwrapper import rf

C = 3e5
cosmo = rf.experiments.cosmo

# Precalculate background evolution
H, r, D, f = rf.background_evolution_splines(cosmo, zmax=10., nsamples=500)
_z = np.linspace(0., 10., 1000)
_vol = C * scipy.integrate.cumtrapz(r(_z)**2. / H(_z), _z, initial=0.)
_vol *= (np.pi/180.)**2. / 1e9 # per deg^2, in Gpc^3
vol = scipy.interpolate.interp1d(_z, _vol, kind='linear', bounds_error=False)

def Vsurvey(zmin, zmax, sarea):
    return sarea * (vol(zmax) - vol(zmin))

# zmin, zmax, sarea, name
surveys = [
    [0., 0.8, 10e3, "BOSS"],
    [1.9, 3.5, 420., "HETDEX"],
    [0.1, 1.9, 14e3, "DESI"],
    [0.6, 2.1, 15e3, "Euclid"],
    [1.0, 2.8, 2e3, "WFIRST"],
    [0.0, 3.0, 25e3, "SKA1-MID (IM)"],
    [0.18, 1.86, 30e3, "SKA Full (gal. survey)"],
]

for s in surveys:
    zmin, zmax, sarea, lbl = s
    zc = 0.5 * (zmin + zmax)
    dz = zmax - zc
    vs = Vsurvey(zmin, zmax, sarea)
    
    xx = 0.; yy = 0.; col='k'
    if "SKA1-MID" in lbl: col = 'r'
    if "SKA Full" in lbl: col = 'b'
    if "HETDEX" in lbl:
        xx = 60.
        yy = -5.
    if "DESI" in lbl: yy = -35.
    
    P.errorbar(zc, vs, xerr=dz, color=col, marker='s', markeredgecolor=col, lw=2.)
    P.annotate(lbl, xy=(zc, vs), color=col,
                       xytext=(0.+xx, 15.+yy), fontsize='x-large', 
                       textcoords='offset points', ha='center', va='center' )

P.xlim((-0.05, 3.7))
P.ylim((-10., 810.))
P.axvline(0., ls='dotted', lw=1.5, color='k')
P.axhline(0., ls='dotted', lw=1.5, color='k')
#P.yscale('log')

P.gca().tick_params(axis='both', which='major', labelsize=20, size=8., width=1.5, pad=8.)
P.gca().tick_params(axis='both', which='minor', labelsize=20, size=5., width=1.5, pad=8.)
P.xlabel(r"$z$", fontdict={'fontsize':'xx-large'}, labelpad=10.)
P.ylabel(r"Volume [Gpc$^3$]", fontdict={'fontsize':'xx-large'})

P.tight_layout()
P.savefig("survey-volumes.png")
P.show()
