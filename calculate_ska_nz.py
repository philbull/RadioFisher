#!/usr/bin/python
"""
Calculate dn/dz for SKA HI galaxy redshift surveys, using dn/dz curves for 
given flux thresholds (from Mario Santos) and flux scalings with redshift for 
specific arrays.
"""
import numpy as np
import pylab as P
import scipy.interpolate
import scipy.optimize
#import radiofisher as rf
#import radiofisher.experiments as experiments
#from radiofisher.units import *
import sys

DEBUG_PLOT = True # Whether to plot fitting functions or not

NU_LINE = 1.420 # HI emission line freq. in GHz
FULLSKY = (4.*np.pi * (180./np.pi)**2.) # deg^2 in the full sky
NGAL_MIN = 1e3 # Min. no. of galaxies to tolerate in a redshift bin
CBM = 1. #np.sqrt(1.57) # Correction factor due to effective beam for MID/MK (OBSOLETE)
CTH = 0.5 # Correction factor due to taking 5 sigma (not 10 sigma) cuts for SKA1
SBIG = 500. # Flux rms to extrapolate dn/dz out to (constrains behaviour at large Srms)

# Specify which experiment to calculate for
try:
    ID = int(sys.argv[1])
    sarea_in = float(sys.argv[2]) if len(sys.argv) > 2 else None
except:
    print "Expects 1 or 2 arguments: int(experiment_id) [float(S_area, deg^2)]"
    sys.exit()

#ax1 = P.subplot(211)
#ax2 = P.subplot(212)
#for ID in [4, 5, 9, 10]:

# Flux rms limits at 1GHz for various configurations
name = ['SKA1MID-B1', 'SKA1MID-B2', 'MEERKAT-B1', 'MEERKAT-B2', 'MID+MK-B1',
        'MID+MK-B2', 'SKA1SUR-B1', 'SKA1SUR-B2', 'ASKAP', 'SUR+ASKAP', 'SKA2',
        'SKA1-opt', 'SKA1-ref', 'SKA1-pess', 'SKA2-opt', 'SKA2-ref', 'SKA2-pess']
#fluxrms = [251., 149., 555., 598., 197., 122., 174., 192., 645., 139., 5.] # Old
fluxrms = [315., 187., 696., 750., 247., 152., 174., 192., 645., 179., 5.14,
           70., 150., 200., 3.0, 5.4, 23.0]
numin = [350., 950., 580., 900., 580., 950., 350., 650., 700., 700., 500.,
         950., 950., 950., 470., 470., 470.]
numax = [1050., 1760., 1015., 1670., 1015., 1670., 900., 1670., 1800., 1670., 1300.,
         1670., 1670., 1670., 1290., 1290., 1290.]
nucrit = [None, None, None, None, None, None, 710., 1300., 1250., 1300., None,
         None, None, None, None, None, None]
Sarea = [5e3, 5e3, 5e3, 5e3, 5e3, 5e3, 5e3, 5e3, 5e3, 5e3, 30e3,
         5e3, 5e3, 5e3, 30e3, 30e3, 30e3] # Assumed Sarea for fits
Sconst = [False, False, False, False, False, False, False, False, False, False, True,
          False, False, False, True, True, True]
Scorr = [CBM*CTH, CBM*CTH, CBM*CTH, CBM*CTH, CBM*CTH, CBM*CTH, CTH, CTH, CTH, CTH, 1.,
         CBM*CTH, CBM*CTH, CBM*CTH, 1., 1., 1.]

# Use default survey area if none was specified
if sarea_in is None: sarea_in = Sarea[ID]

# Define fitting coefficients from Mario's note (HI_specs.pdf)
#Srms = np.array([0., 1., 3., 5., 6., 7.3, 10., 23., 40., 70., 100., 150., 200.,])
#c1 = [6.23, 7.33, 6.91, 6.77, 6.84, 6.76, 6.64, 6.02, 5.74, 5.62, 5.63, 5.48, 5.00]
#c2 = [1.82, 3.02, 2.38, 2.17, 2.23, 2.14, 2.01, 1.43, 1.22, 1.11, 1.41, 1.33, 1.04]
#c3 = [0.98, 5.34, 5.84, 6.63, 7.13, 7.36, 7.95, 9.03, 10.58, 13.03, 15.49, 16.62, 17.52]
#c4 = [0.8695, 0.5863, 0.4780, 0.5884, 0.5908, 0.5088, 0.4489, 0.5751, 0.5125, 
#      0.6193, 0.6212, 1., 1., 1.]
#c5 = [0.2338, 0.6410, 0.9181, 0.8076, 0.8455, 1.0222, 1.2069, 0.9993, 1.1842, 
#      1.0179, 1.0759, 0., 0., 0.]

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
    zmin = NU_LINE*1e3 / numax[ID] - 1.
    zmax = NU_LINE*1e3 / numin[ID] - 1.
    if zmin < 0.: zmin = 0.
    if Nbins is not None:
        zbins = np.linspace(zmin, zmax, Nbins+1)
    else:
        Nbins = np.floor((zmax - zmin) / dz)
        zbins = np.linspace(zmin, zmin + dz*Nbins, Nbins+1)
        if zmax - np.max(zbins) > 0.04:
            zbins = np.concatenate((zbins, [zmax,]))
    return zbins

def flux_redshift(z):
    """
    Flux rms as a function of redshift.
    """
    z = np.atleast_1d(z)
    nu = NU_LINE / (1. + z)
    if nucrit[ID] is not None:
        Sz = fluxrms[ID] * Scorr[ID] * np.ones(nu.size)
        idxs = np.where(nu*1e3 > nucrit[ID])
        Sz[idxs] *= (nu[idxs]*1e3 / nucrit[ID])
    else:
        Sz = fluxrms[ID] * Scorr[ID]
        Sz = nu * Sz if not Sconst[ID] else Sz * np.ones(nu.size)
    # Survey area correction
    Sz *= np.sqrt(sarea_in / Sarea[ID])
    return Sz

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
fsky = sarea_in / FULLSKY #Sarea[ID] / FULLSKY
Sz = flux_redshift(z)
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
    return model - bias_vals
pb = scipy.optimize.leastsq(lsq, p0)[0]


# Print best-fit coefficients
print "-"*30
print "%s (%d deg^2)" % (name[ID], sarea_in)
print "-"*30
print "Fitting coeffs."
print "c1: %6.4f" % np.log10(p[0])
print "c2: %6.4f" % p[1]
print "c3: %6.4f" % p[2]
print "c4: %6.4f" % pb[0]
print "c5: %6.4f" % pb[1]

print " & ".join(["%6.4f" % n for n in [np.log10(p[0]), p[1], p[2], pb[0], pb[1]]])

"""
# Calculate cosmo. functions
cosmo_fns = rf.background_evolution_splines(experiments.cosmo)
H, r, D, f = cosmo_fns

# Calculate binned number densities
zbins = redshift_bins(dz=0.1)
zc = np.array([0.5*(zbins[i] + zbins[i+1]) for i in range(zbins.size-1)])
nz, vol, b = np.array( [n_bin(zbins[i], zbins[i+1], dndz_expt, bias_expt) 
                        for i in range(zbins.size-1)] ).T
vol *= fsky

# Output survey info
print "-"*30
print "zc    zmin  zmax   n Mpc^-3    bias     vol.   Ngal         Srms"
for i in range(zc.size):
    #Szz = fluxrms[ID] * Scorr[ID]
    #Szz = NU_LINE/(1.+zc[i]) * Szz if not Sconst[ID] else Szz
    Szz = flux_redshift(zc[i])
    
    print "%2.2f  %2.2f  %2.2f   %3.3e   %6.3f  %5.2f   %5.3e   %6.2f" % \
    (zc[i], zbins[i], zbins[i+1], nz[i], b[i], vol[i]/1e9, nz[i]*vol[i], Szz),
    if (nz[i]*vol[i]) < NGAL_MIN: print "*",
    if Szz > Smax: print "#",
    print ""
    
print "-"*30
print "Ntot: %3.3e" % np.sum(nz * vol)
print "fsky: %3.3f" % fsky
_zmin = (NU_LINE*1e3 / numax[ID] - 1.)
print "zmin: %3.3f" % (_zmin if _zmin >= 0. else 0.)
print "zmax: %3.3f" % (NU_LINE*1e3 / numin[ID] - 1.)
print "Srms const: %s" % Sconst[ID]
print "-"*30
print "\n"

# Output fitting function coeffs as a fn. of survey area
print "%10s %d %6.4f %6.4f %6.4f %6.4f %6.4f %3.3e" % (name[ID], sarea_in, np.log10(p[0]), p[1], p[2], pb[0], pb[1], np.sum(nz * vol))

#exit()

# Comparison plot of dndz, bias, and fitting function
if DEBUG_PLOT:
    P.subplot(211)
    P.plot(_z, dndz_expt(_z))
    P.plot(_z, p[0] * _z**p[1] * np.exp(-p[2]*_z))

    P.subplot(212)
    P.plot(_z, bias_expt(_z))
    P.plot(_z, pb[0] * np.exp(pb[1]*_z))
    P.show()
"""
