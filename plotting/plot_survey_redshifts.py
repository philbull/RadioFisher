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

# Precalculate background evolution
H, r, D, f = rf.background_evolution_splines(cosmo, zmax=10., nsamples=500)
_z = np.linspace(0., 10., 1000)
_vol = C * scipy.integrate.cumtrapz(r(_z)**2. / H(_z), _z, initial=0.)
_vol *= (np.pi/180.)**2. / 1e9 # per deg^2, in Gpc^3
vol = scipy.interpolate.interp1d(_z, _vol, kind='linear', bounds_error=False)

def Vsurvey(zmin, zmax, sarea):
    return sarea * (vol(zmax) - vol(zmin))


cols = ['#a6a6a6', '#000000', '#5DBEFF', '#1C7EC5', '#1619A1', '#FFB928', 
        '#ff6600', '#CC0000', '#95CD6D', '#007A10', '#858585', '#c1c1c1', 
        'c', 'm']
cols += cols

cmid1 = '#F8A618' #'#FFB928' #'#1619A1'
cmid2 = '#CC0000'
clow = '#007A10' #'#1C7EC5'
c2 = '#1C7EC5' #'#5DBEFF'
coth = '#858585' #'#CC0000'
cols = [cmid2, cmid2, coth, coth, c2, coth, cmid1, coth, coth, cmid1, coth, clow, coth, clow]

# zmin, zmax, sarea, name
surveys = [
    
    [0.00, 0.49, 25e3, "SKA1-MID B2 Rebase."],
    [0.00, 0.79, 25e3, "SKA1-MID B2 Alt."],
    [0., 0.8, 10e3, "BOSS"],
    [0., 1.5, 30e3, "SPHEREx"],
    
    [0.1, 2.0, 30e3, "SKA2"],
    [0.1, 2.6, 18e3, "LSST"],
    [0.35, 3.06, 25e3, "SKA1-MID B1 Rebase."],
    [0.6, 1.85, 14e3, "DESI"],
    [0.6, 2., 15e3, "Euclid"],
    [0.72, 2.16, 25e3, "SKA1-MID B1 Alt."],
    [1.0, 2.8, 2e3, "WFIRST"],
    
    [1.84, 6.1, 1e3, 'SKA1-LOW Alt.'],
    [1.9, 3.5, 420., "HETDEX"],
    
    [3.06, 6.1, 1e3, 'SKA1-LOW Rebase.'],
]

#P.axvline(0.75, ls='dashed', color='k', lw=1.5, alpha=0.17)
#P.axvline(2.0, ls='dashed', color='k', lw=1.5, alpha=0.17)
#P.axvline(3.5, ls='dashed', color='k', lw=1.5, alpha=0.17)

P.axvspan(0., 0.75, color='k', alpha=0.05)
P.axvspan(2., 3.5, color='k', alpha=0.05)


# SPHEREx extra sample
zc = 0.5 * (0. + 2.6)
dz = 2.6 - zc
P.errorbar(zc, 3*1.5, xerr=dz, color='#C8C8C8', marker=None, 
           markeredgecolor='#C8C8C8', lw=4., capsize=0.)


for i, s, col in zip(range(len(surveys)), surveys, cols):
    zmin, zmax, sarea, lbl = s
    zc = 0.5 * (zmin + zmax)
    dz = zmax - zc
    #vs = Vsurvey(zmin, zmax, sarea)
    
    xx = 0.; yy = 0.
    
    P.errorbar(zc, i*1.5, xerr=dz, color=col, marker=None, markeredgecolor=col, lw=4.,
               capsize=0.)
    P.annotate(lbl, xy=(zc-dz+0.1, i*1.5), color=col,
                       xytext=(xx, 8.+yy), fontsize=14, fontweight='medium',
                       textcoords='offset points', ha='left', va='center' )

P.xlim((0.0, 6.2))
P.ylim((-1, (len(surveys)+1)*1.5))
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

P.tight_layout()
P.savefig("survey-redshifts.pdf")
P.show()
