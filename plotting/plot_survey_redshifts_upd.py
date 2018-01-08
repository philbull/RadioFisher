#!/usr/bin/python
"""
Calculate and plot the comoving volumes of some surveys.
"""
import numpy as np
import pylab as P
import scipy.integrate
import scipy.interpolate
import matplotlib.ticker
from rfwrapper import rf

C = 3e5
cosmo = rf.experiments.cosmo
HEIGHT = 2.8

# Precalculate background evolution
H, r, D, f = rf.background_evolution_splines(cosmo, zmax=10., nsamples=500)
_z = np.linspace(0., 10., 1000)
_vol = C * scipy.integrate.cumtrapz(r(_z)**2. / H(_z), _z, initial=0.)
_vol *= (np.pi/180.)**2. / 1e9 # per deg^2, in Gpc^3
vol = scipy.interpolate.interp1d(_z, _vol, kind='linear', bounds_error=False)

def Vsurvey(zmin, zmax, sarea):
    return sarea * (vol(zmax) - vol(zmin))

#cphoto = '#F8A618' #'#FFB928' #'#1619A1'
cphoto = '#CC0000'
cim = '#007A10' #'#1C7EC5'
cspec = '#1C7EC5' #'#5DBEFF'
#cphoto = '#858585' #'#CC0000'

# zmin, zmax, sarea, name
surveys = [
    
    [0.057, 0.098, 30e3, "Parkes-IM", cim],
    [0.00, 0.49, 25e3, "SKA1-MID Band 2 IM", cim],
    [0.13, 0.48, 30e3, "BINGO", cim],
    [0.35, 3.06, 25e3, "SKA1-MID Band 1 IM", cim],
    [0.53, 1.12, 30e3, "GBT-IM", cim],
    [0.8, 2.5, 30e3, "CHIME", cim],
    [0.8, 2.5, 30e3, "HIRAX", cim],
    [3.06, 20., 1e3, 'SKA1-LOW', cim],
    
    [0., 0.4, 25e3, "SKA1-MID Band 2 HI GRS", cspec],
    [0., 0.8, 10e3, "BOSS", cspec],
    [0., 2.0, 30e3, "SKA2 HI GRS", cspec],
    [0.6, 1.85, 14e3, "DESI", cspec],
    [0.6, 2., 15e3, "Euclid", cspec],
    [1.05, 2., 2e3, "WFIRST", cspec],
    [1.9, 3.5, 420., "HETDEX", cspec],
    
    [0., 1.5, 18e3, "DES", cphoto],
    [0., 1.5, 30e3, "SPHEREx", cphoto],
    [0., 3.0, 18e3, "LSST", cphoto],
    [0., 5.0, 30e3, "SKA1 Continuum", cphoto],
]

#P.axvline(0.75, ls='dashed', color='k', lw=1.5, alpha=0.17)
#P.axvline(2.0, ls='dashed', color='k', lw=1.5, alpha=0.17)
#P.axvline(3.5, ls='dashed', color='k', lw=1.5, alpha=0.17)

P.axvspan(0., 0.75, color='k', alpha=0.05)
P.axvspan(2., 3.5, color='k', alpha=0.05)


# SPHEREx extra sample
zc = 0.5 * (0. + 2.6)
dz = 2.6 - zc
P.errorbar(zc, 16*HEIGHT, xerr=dz, color=cphoto, marker=None, 
           markeredgecolor=cphoto, lw=4., capsize=0., alpha=0.5)

# WFIRST O[III] sample
zc = 0.5 * (1.7 + 2.9)
dz = 2.9 - zc
P.errorbar(zc, 13*HEIGHT, xerr=dz, color=cspec, marker=None, 
           markeredgecolor=cspec, lw=4., capsize=0., alpha=0.5)

for i, s in enumerate(surveys):
    zmin, zmax, sarea, lbl, col = s
    zc = 0.5 * (zmin + zmax)
    dz = zmax - zc
    #vs = Vsurvey(zmin, zmax, sarea)
    
    xx = 0.; yy = 0.
    
    P.errorbar(zc, i*HEIGHT, xerr=dz, color=col, marker=None, markeredgecolor=col, lw=4.,
               capsize=0.)
    P.annotate(lbl, xy=(zc-dz+0.1, i*HEIGHT), color=col,
                       xytext=(xx, 8.+yy), fontsize=13, fontweight='medium',
                       textcoords='offset points', ha='left', va='center' )

P.xlim((0.0, 5.5))
P.ylim((-1, (len(surveys)+1)*HEIGHT))
#P.axvline(0., ls='dotted', lw=1.5, color='k')
#P.axhline(0., ls='dotted', lw=1.5, color='k')
#P.yscale('log')

P.gca().tick_params(axis='both', which='major', labelsize=20, size=8., width=1.5, pad=8.)
P.gca().tick_params(axis='both', which='minor', labelsize=20, size=5., width=1.5, pad=8.)
P.gca().tick_params(axis='y', which='major', labelleft='off', size=0., width=0., pad=8.)
P.xlabel(r"$z$", fontdict={'fontsize':'xx-large'}, labelpad=10.)
P.ylabel(r".", fontdict={'fontsize':'xx-large'}, labelpad=10., color='w')
#P.ylabel(r"Volume [Gpc$^3$]", fontdict={'fontsize':'xx-large'}, labelpad=10.)

P.gca().xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.5))

P.gcf().set_size_inches(8., 6.8)
P.tight_layout()
P.savefig("survey_redshifts_upd.pdf")
P.show()
