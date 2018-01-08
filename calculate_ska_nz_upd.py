#!/usr/bin/python
"""
Calculate dn/dz for SKA HI galaxy redshift surveys, using dn/dz curves for 
given flux thresholds (from Mario Santos) and flux scalings with redshift for 
specific arrays.
  -- Phil Bull, 2015
"""
import numpy as np
import pylab as P
import scipy.interpolate
import scipy.optimize
import radiofisher as rf
import radiofisher.experiments as experiments
from radiofisher.units import *
import sys

try:
    EXPT_ID = int(sys.argv[1])
    if len(sys.argv) > 2:
        NSIGMA = float(sys.argv[2])
    else:
        NSIGMA = 5.
except:
    print "Expects one or two arguments: expt_id(int) [nsigma(int)]"
    sys.exit(1)

DEBUG_PLOT = True # Whether to plot fitting functions or not

NU_LINE = 1.420 # HI emission line freq. in GHz
FULLSKY = (4.*np.pi * (180./np.pi)**2.) # deg^2 in the full sky
NGAL_MIN = 1e3 # Min. no. of galaxies to tolerate in a redshift bin
CBM = 1. #np.sqrt(1.57) # Correction factor due to effective beam for MID/MK (OBSOLETE)
CTH = 0.5 # Correction factor due to taking 5 sigma (not 10 sigma) cuts for SKA1
SBIG = 500. # Flux rms to extrapolate dn/dz out to (constrains behaviour at large Srms)

DZBINS = 0.03 #0.1 # Width of redshift bins

#-------------------------------------------------------------------------------
# Survey specifications
#-------------------------------------------------------------------------------

## SKA1-MID B2 Baseline
#name = "SKA1-MID B2 Baseline"
#numin = 900.
#numax = 1420.
#Sarea = 5e3
#Tinst = 20.
#Ddish = 15.
#Ndish = 254.
#ttot = 1e4
#dnu = 0.01
#effic = 0.8
#Nsig = 5.

## SKA1-MID B2 Updated
#name = "SKA1-MID B2 Updated"
#numin = 900.
#numax = 1420.
#Sarea = 5e3
#Tinst = 20.
#Ddish = 15.
#Ndish = 200.
#ttot = 1e4
#dnu = 0.01
#effic = 0.8
#Nsig = 5.

## SKA1-MID B2 Alternative
#name = "SKA1-MID B2 Alternative"
#numin = 785.
#numax = 1420.
#Sarea = 5e3
#Tinst = 20.
#Ddish = 15.
#Ndish = 200.
#ttot = 1e4
#dnu = 0.01
#effic = 0.8
#Nsig = 5.

## SKA2
#name = "SKA2"
#numin = 470.
#numax = 1290.
#Sarea = 30e3
#Tinst = 15.
#Ddish = 3.1
#Ndish = 7e4
#ttot = 1e4
#dnu = 0.01
#effic = 0.8
#Nsig = 10.

## SKA1-MID B1 TEST
#name = "SKA1-MID B1 TEST"
#numin = 430.
#numax = 800.
#Sarea = 1e2
#Tinst = 23.
#Ddish = 15.
#Ndish = 200.
#ttot = 1e4
#dnu = 0.01
#effic = 0.8
#Nsig = 5.


# FIXME
#name = "SKA1-MID B2 x"
#numin = 785.
#numax = 1420.
#Sarea = 5e3
#Tinst = 20.
#Ddish = 15.
#Ndish = 200.
#ttot = 1e4
#dnu = 0.01
#effic = 0.8
#Nsig = 5.


#-------------------------------------------------------------------------------
# Combined survey specifications
#-------------------------------------------------------------------------------

if EXPT_ID == 0:
    # SKA1-MID B2 Rebaselined
    name = "SKA1-MID B2 Rebase. + MK"
    numin = 900.
    numax = 1420.
    Sarea = 5e3
    ttot = 1e4
    dnu = 0.01
    Nsig = NSIGMA #5.

    # Tinst1, Ddish1, Ndish1, effic1, numin1, numax1
    expt1 = (15.5, 15., 130., 0.85, 950., 1420.) # MID B2 Rebase.
    expt2 = (30., 13.5, 64., 0.85, 900., 1420.) # MeerKAT

if EXPT_ID == 1:
    # SKA1-MID B2 Alternative
    name = "SKA1-MID B2 Alt. + MK"
    numin = 795.
    numax = 1420.
    Sarea = 5e3
    ttot = 1e4
    dnu = 0.01
    Nsig = NSIGMA #5.

    # Tinst1, Ddish1, Ndish1, effic1, numin1, numax1
    expt1 = (15.5, 15., 130., 0.85, 795., 1420.) # MID B2 Alt.
    expt2 = (30., 13.5, 64., 0.85, 900., 1420.) # MeerKAT

if EXPT_ID == 2:
    # SKA2
    name = "SKA2"
    numin = 470.
    numax = 1290.
    
    Sarea = 30e3
    ttot = 1e4
    dnu = 0.01
    Nsig = NSIGMA #10.

    # Tinst1, Ddish1, Ndish1, effic1, numin1, numax1
    #expt1 = (15., 10., 4000., 0.8, numin, numax) # MID B2 Alt.
    expt1 = (15., 3.1, 70000., 0.8, numin, numax) # MID B2 Alt.
    expt2 = (10., 1., 0., 0.85, numin, numax) # MeerKAT
    
    #Sarea = 30e3
    #Tinst = 15.
    #Ddish = 3.1
    #Ndish = 7e4
    #ttot = 1e4
    #dnu = 0.01
    #effic = 0.8
    #Nsig = 10.

if EXPT_ID == 3:
    # FAST
    name = "FAST"
    numin = 800.
    numax = 1420.
    Sarea = 20e3
    ttot = 1e4
    dnu = 0.01
    Nsig = NSIGMA

    # Tinst1, Ddish1, Ndish1, effic1, numin1, numax1
    expt1 = (22., 300., 19., 0.7, 900., 1420.) # FAST
    expt2 = (0., 1., 0., 0., 1500., 1600.) # Nothing

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

def fluxrms_combined(nu, expt1, expt2, Sarea, ttot, dnu=0.01):
    """
    Calculate the flux rms [uJy] for a combination of arrays, using expression 
    from Yahya et al. (2015), near Eq. 3.
    """
    Tinst1, Ddish1, Ndish1, effic1, numin1, numax1 = expt1
    Tinst2, Ddish2, Ndish2, effic2, numin2, numax2 = expt2
    
    # Calculate Aeff / Tsys for each sub-array
    Tsys1 = Tinst1 + 60. * (300./nu)**2.55 # [K]
    Tsys2 = Tinst2 + 60. * (300./nu)**2.55 # [K]
    Aeff1 = effic1 * Ndish1 * np.pi * (Ddish1/2.)**2. # [m^2]
    Aeff2 = effic2 * Ndish2 * np.pi * (Ddish2/2.)**2. # [m^2]
    
    # Define band masks
    msk1 = np.zeros(nu.shape); msk2 = np.zeros(nu.shape)
    msk1[np.where(np.logical_and(nu >= numin1, nu <= numax1))] = 1.
    msk2[np.where(np.logical_and(nu >= numin2, nu <= numax2))] = 1.
    
    # Calculate combined Aeff / Tsys
    Aeff_over_Tsys = Aeff1/Tsys1*msk1 + Aeff2/Tsys2*msk2
    
    # Calculate mean FOV
    fov1 = (np.pi/8.) * (1.3 * 3e8 / (nu*1e6 * Ddish1))**2.
    fov2 = (np.pi/8.) * (1.3 * 3e8 / (nu*1e6 * Ddish1))**2.
    fov = (Ndish1 * fov1 + Ndish2 * fov2) / float(Ndish1 + Ndish2)
    fov *= (180./np.pi)**2. # [deg^2]
    
    # Calculate time per pointing and overall sensitivity
    tp = ttot * (fov / Sarea)
    Srms = 260. * (25e3/20.) / Aeff_over_Tsys * np.sqrt( (0.01/dnu) * (1./tp) )
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

# Construct dndz(z) interpolation fn. for the sensitivity of actual experiment
fsky = Sarea / FULLSKY
nu = 1420. / (1. + z)

# Calculate flux
Sz = (Nsig/10.) * fluxrms_combined(nu, expt1, expt2, Sarea, ttot, dnu)
#Sref = fluxrms(1000., Tinst, Ddish, Ndish, Sarea, ttot, dnu, effic)
#print "Srms = %3.1f uJy [%d sigma]" % (Sref, Nsig)

dndz_expt = scipy.interpolate.interp1d(z, dndz.ev(Sz, z))
bias_expt = scipy.interpolate.interp1d(z, bias.ev(Sz, z))

# Fit function to dn/dz [deg^-2]
_z = np.linspace(1e-7, 1., 1e4)
dndz_vals = dndz_expt(_z)
bias_vals = bias_expt(_z)
p0 = [100.*np.max(dndz_vals), 2., 10.]
def lsq(params):
    A, c2, c3 = params
    model = A * _z**c2 * np.exp(-c3*_z)
    return model - dndz_vals
p = scipy.optimize.leastsq(lsq, p0)[0]

# Fit function to bias
p0 = [np.max(bias_vals), 0.5]
def lsq(params):
    c4, c5 = params
    model = c4 * np.exp(c5*_z)
    w = np.sqrt(dndz_vals) # Weight fit by sqrt(dn/dz)
    return (model - bias_vals) * w
pb = scipy.optimize.leastsq(lsq, p0)[0]


# Print best-fit coefficients
print "-"*50
print "%s (%d deg^2) [%s-sigma]" % (name, Sarea, Nsig)
print "-"*50
print "Fitting coeffs."
print "c1: %6.4f" % np.log10(p[0])
print "c2: %6.4f" % p[1]
print "c3: %6.4f" % p[2]
print "c4: %6.4f" % pb[0]
print "c5: %6.4f" % pb[1]

print " & ".join(["%6.4f" % n for n in [np.log10(p[0]), p[1], p[2], pb[0], pb[1]]])


# Calculate cosmo. functions
cosmo_fns = rf.background_evolution_splines(experiments.cosmo)
H, r, D, f = cosmo_fns

# Calculate binned number densities
zbins = redshift_bins(dz=DZBINS)
zc = np.array([0.5*(zbins[i] + zbins[i+1]) for i in range(zbins.size-1)])
nz, vol, b = np.array( [n_bin(zbins[i], zbins[i+1], dndz_expt, bias_expt) 
                        for i in range(zbins.size-1)] ).T
vol *= fsky

# Find z_max
zz = np.linspace(0., 3., 1500)
zzc = 0.5 * (zz[:-1] + zz[1:])
_nz, _vol, _b = np.array( [n_bin(zz[i], zz[i+1], dndz_expt, bias_expt) 
                        for i in range(zz.size-1)] ).T
#print "z_min = %3.3f" % zz[np.argmin(np.abs(_nz - 5e-4))]
print name

# Load P(k) and get 1/(b^2 P(k_NL))
k, pk = np.genfromtxt("cache_pk.dat").T
knl = 0.14 * (1. + zzc)**(2./(2. + experiments.cosmo['ns']))
kref = knl #0.1
pk02 = scipy.interpolate.interp1d(k, pk, kind='linear')(0.5*kref)
pkinv = 1./ ( pk02 * (D(zzc) * _b)**2. )

print "z_max = %3.3f" % zzc[np.argmin(np.abs(_nz - pkinv))]

# Output survey info
print "-"*30
print "zc    zmin  zmax   n Mpc^-3    bias    vol.   Ngal         Srms"
for i in range(zc.size):
    #Szz = fluxrms[ID] * Scorr[ID]
    #Szz = NU_LINE/(1.+zc[i]) * Szz if not Sconst[ID] else Szz
    nu_c = np.atleast_1d( 1420. / (1. + zc[i]) )
    #Szz = (Nsig/10.) * fluxrms(nu_c, Tinst, Ddish, Ndish, Sarea, ttot, dnu, effic)
    Szz = (Nsig/10.) * fluxrms_combined(nu_c, expt1, expt2, Sarea, ttot, dnu) # 5 sigma
    
    print "%2.2f  %2.2f  %2.2f   %3.3e  %6.3f  %5.2f   %5.3e   %6.2f" % \
    (zc[i], zbins[i], zbins[i+1], nz[i], b[i], vol[i]/1e9, nz[i]*vol[i], Szz),
    if (nz[i]*vol[i]) < NGAL_MIN: print "*",
    if Szz > Smax: print "#",
    print ""
    
print "-"*30
print "Ntot: %3.3e" % np.sum(nz * vol)
print "fsky: %3.3f" % fsky
_zmin = (NU_LINE*1e3 / numax - 1.)
print "zmin: %3.3f" % (_zmin if _zmin >= 0. else 0.)
print "zmax: %3.3f" % (NU_LINE*1e3 / numin - 1.)
#print "Srms const: %s" % Sconst[ID]
print "-"*30
print ""

# Output fitting function coeffs as a fn. of survey area
print "%10s %d %6.4f %6.4f %6.4f %6.4f %6.4f %3.3e" % (name, Sarea, np.log10(p[0]), p[1], p[2], pb[0], pb[1], np.sum(nz * vol))


# Plot dn/dz in arcmin^-2
P.subplot(111)
P.plot(_z, dndz_expt(_z) / 60.**2., 'b-', lw=1.8)

P.tick_params(axis='both', which='major', labelsize=18, size=8., width=1.5, pad=5.)

P.ylabel(r"$N(z)$ $[{\rm amin}^{-2}]$", fontsize=18.)
P.xlabel("$z$", fontsize=18.)
P.xlim((0., 0.6))
P.tight_layout()
P.show()
exit()

# Comparison plot of dndz, bias, and fitting function
if DEBUG_PLOT:
    P.subplot(211)
    P.plot(_z, dndz_expt(_z))
    P.plot(_z, p[0] * _z**p[1] * np.exp(-p[2]*_z))

    P.subplot(212)
    P.plot(_z, bias_expt(_z))
    P.plot(_z, pb[0] * np.exp(pb[1]*_z))
    P.show()

