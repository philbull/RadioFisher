#!/usr/bin/python
"""
Calculate dn/dz for SKA HI galaxy redshift surveys, using dn/dz curves for 
given flux thresholds (from Mario Santos) and flux scalings with redshift for 
specific arrays.

Use these to find the flux rms / survey area / time needed to get a certain 
sensitivity at a given redshift.
"""
import numpy as np
import pylab as P
import scipy.interpolate
import scipy.optimize
import radiofisher as rf
import radiofisher.experiments as experiments
from radiofisher.units import *
import sys

DEBUG_PLOT = True # Whether to plot fitting functions or not

NU_LINE = 1.420 # HI emission line freq. in GHz
FULLSKY = (4.*np.pi * (180./np.pi)**2.) # deg^2 in the full sky
NGAL_MIN = 1e3 # Min. no. of galaxies to tolerate in a redshift bin
CBM = 1. #np.sqrt(1.57) # Correction factor due to effective beam for MID/MK (OBSOLETE)
CTH = 0.5 # Correction factor due to taking 5 sigma (not 10 sigma) cuts for SKA1
SBIG = 500. # Flux rms to extrapolate dn/dz out to (constrains behaviour at large Srms)

#-------------------------------------------------------------------------------
# Survey specifications
#-------------------------------------------------------------------------------

name = "SKA1-MID B2 x"
numin = 350. #785.
numax = 1420.
Sarea = 5e3
Tinst = 20.
Ddish = 15.
Ndish = 200.
ttot = 1e3
dnu = 0.01
effic = 0.8
Nsig = 5.

#-------------------------------------------------------------------------------

# Fitting coeffs. from Table 3 in v1 of Yahya et al. paper
Srms = np.array([0., 1., 3., 5., 6., 7.3, 10., 23., 40., 70., 100., 150., 200.])
c1 = [6.21, 6.55, 6.53, 6.55, 6.58, 6.55, 6.44, 6.02, 5.74, 5.62, 5.63, 5.48, 5.0]
c2 = [1.72, 2.02, 1.93, 1.93, 1.95, 1.92, 1.83, 1.43, 1.22, 1.11, 1.41, 1.33, 1.04]
c3 = [0.79, 3.81, 5.22, 6.22, 6.69, 7.08, 7.59, 9.03, 10.58, 13.03, 15.49, 16.62, 17.52]
c4 = [0.5874, 0.4968, 0.5302, 0.5504, 0.5466, 0.5623, 0.5928, 0.6069, 0.628, 
      0.6094, 0.6052, 0.6365, 1., 1.] # Needs end padding 1
c5 = [0.3577, 0.7206, 0.7809, 0.8015, 0.8294, 0.8233, 0.8072, 0.8521, 0.8442, 
      0.9293, 1.0859, 0.965, 0., 0.] # Needs end padding 0
c1 = np.array(c1); c2 = np.array(c2); c3 = np.array(c3)
c4 = np.array(c4); c5 = np.array(c5)
Smax = np.max(Srms)

# Calculate cosmo. functions
cosmo_fns = rf.background_evolution_splines(experiments.cosmo)
H, r, D, f = cosmo_fns


def fluxrms(nu, Tinst, Ddish, Ndish, Sarea, ttot, dnu=0.01, effic=0.7):
    """
    Calculate the flux rms [uJy] for a given array config., using expression 
    from Yahya et al. (2015), near Eq. 3.
    """
    Tsys = Tinst + 60. * (300./nu)**2.55 # [K]
    Aeff = effic * Ndish * np.pi * (Ddish/2.)**2. # [m^2]
    fov = (np.pi/8.) * (1.3 * 3e8 / (nu*1e6 * Ddish))**2. * (180./np.pi)**2. # [deg^2]
    tp = ttot * (fov / Sarea)
    Srms = 260. * (Tsys/20.) * (25e3 / Aeff) * np.sqrt( (0.01/dnu) * (1./tp) )
    return Srms

def extend_with_linear_interp(xnew, x, y):
    """
    Extend an array using a linear interpolation from the last two points.
    """
    dx = x[-1] - x[-2]
    dy = y[-1] - y[-2]
    ynew = y[-1] + dy * (xnew - x[-1]) / dx
    y = np.concatenate((y, [ynew,]))
    return y

def n_bin(zmin, zmax, dndz, bias=None):
    """
    Number density of galaxies in a given z bin (assumes full sky). Also 
    returns volume of bin. dndz argument expects an interpolation fn. in units 
    of deg^-2.
    """
    _z = np.linspace(zmin, zmax, 500)
    vol = 4.*np.pi*C * scipy.integrate.simps(r(_z)**2. / H(_z), _z)
    N_bin = FULLSKY * scipy.integrate.simps(dndz(_z), _z)
    nz = N_bin / vol
    
    # Calculate mean bias (weighted by number density)
    if bias is not None:
        b = scipy.integrate.simps(bias(_z)*dndz(_z), _z) / (N_bin / FULLSKY)
        #b = bias(0.5*(zmin+zmax))
        return nz, vol, b
    return nz, vol

def redshift_bins(dz=0.1, Nbins=None):
    """
    Calculate redshift bins.
    """
    zmin = NU_LINE*1e3 / numax - 1.
    zmax = NU_LINE*1e3 / numin - 1.
    if zmin < 0.: zmin = 0.
    if Nbins is not None:
        zbins = np.linspace(zmin, zmax, Nbins+1)
    else:
        Nbins = np.floor((zmax - zmin) / dz)
        zbins = np.linspace(zmin, zmin + dz*Nbins, Nbins+1)
        if zmax - np.max(zbins) > 0.04:
            zbins = np.concatenate((zbins, [zmax,]))
    return zbins


# Extrapolate fitting functions to high flux rms
c1 = extend_with_linear_interp(SBIG, Srms, c1)
c2 = np.concatenate((c2, [1.,])) # Asymptote to linear fn. of redshift
c3 = extend_with_linear_interp(SBIG, Srms, c3)
Srms = np.concatenate((Srms, [SBIG,]))

# Construct grid of dn/dz (deg^-2) as a function of flux rms and redshift and 
# then construct 2D interpolator
z = np.linspace(0., 4., 400)
nu = NU_LINE / (1. + z)
_dndz = np.array([10.**c1[j] * z**c2[j] * np.exp(-c3[j]*z) for j in range(Srms.size)])
_bias = np.array([c4[j] * np.exp(c5[j]*z) for j in range(Srms.size)])
dndz = scipy.interpolate.RectBivariateSpline(Srms, z, _dndz, kx=1, ky=1)
bias = scipy.interpolate.RectBivariateSpline(Srms, z, _bias, kx=1, ky=1)


def nz_for_specs(Sarea, ttot):
    """
    Calculate n(z) for a given set of specs.
    """
    # Construct dndz(z) interpolation fn. for the sensitivity of actual experiment
    fsky = Sarea / FULLSKY
    nu = 1420. / (1. + z)

    # Calculate flux
    Sz = (Nsig/10.) * fluxrms(nu, Tinst, Ddish, Ndish, Sarea, ttot, dnu, effic) # 5 sigma
    Sref = fluxrms(1000., Tinst, Ddish, Ndish, Sarea, ttot, dnu, effic)
    #print "Srms = %3.1f uJy [%d sigma]" % (Sref, Nsig)

    dndz_expt = scipy.interpolate.interp1d(z, dndz.ev(Sz, z))
    bias_expt = scipy.interpolate.interp1d(z, bias.ev(Sz, z))

    # Calculate binned number densities
    nz, vol, b = np.array( [n_bin(zbins[i], zbins[i+1], dndz_expt, bias_expt) 
                            for i in range(zbins.size-1)] ).T
    vol *= fsky
    return nz


# Define redshift bins
zbins = redshift_bins(dz=0.1)
zc = np.array([0.5*(zbins[i] + zbins[i+1]) for i in range(zbins.size-1)])

# Get n(z) as a function of survey area
sareas = np.logspace(2., np.log10(30e4), 25)
TTOT = 1e3
nzs = np.array( [nz_for_specs(_s, TTOT) for _s in sareas] ) # nz(sarea, z)

print nzs.shape


# Plot results
P.subplot(111)

cols = ['#a6a6a6', '#000000', '#5DBEFF', '#1C7EC5', '#1619A1', 
           '#FFB928', '#ff6600', '#CC0000', '#95CD6D', 'g']
bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for i in bins:
    P.plot(sareas, nzs[:,i], lw=2.2, color=cols[bins.index(i)], label="z = %2.2f" % zc[i])

P.xscale('log')
P.yscale('log')
P.axhline(1e-4, color='k', ls='dashed', lw=2.8)

# Reference survey areas
P.axvline(1e3, color='k', ls='dotted', lw=2.4, alpha=0.6)
P.axvline(5e3, color='k', ls='dotted', lw=2.4, alpha=0.6)
P.axvline(20e3, color='k', ls='dotted', lw=2.4, alpha=0.6)

P.legend(loc='lower left', frameon=True, framealpha=0.8, prop={'size':'large'}, ncol=2)

P.figtext(0.2, 0.91, "1,000 hrs, 5-sig", fontsize='xx-large')

P.ylim((1e-8, 8e-1))
P.xlim((100., 3e4))

P.tick_params(axis='both', which='major', labelsize=20, width=1.5, size=8., pad=10)
P.tick_params(axis='both', which='minor', labelsize=20, width=1.5, size=5.)

P.xlabel(r'$S_{\rm area}$ $[{\rm deg}^{2}]$', labelpad=10., fontdict={'fontsize':'xx-large'})
P.ylabel(r'$n(z)$ $[{\rm Mpc}^{-3}]$', labelpad=15., fontdict={'fontsize':'xx-large'})

P.tight_layout()
P.show()
