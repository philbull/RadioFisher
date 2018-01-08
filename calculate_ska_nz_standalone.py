#!/usr/bin/python
"""
Calculate dn/dz for SKA HI galaxy redshift surveys, using dn/dz curves for 
given flux thresholds (from Mario Santos) and flux scalings with redshift for 
specific arrays.
  -- Phil Bull, 2017
"""
import numpy as np
import pylab as P
import scipy.interpolate
import scipy.optimize
import sys

C = 2.99792458e5 # Speed of light, km/s
D2RAD = np.pi / 180. # Convert degrees to radians
HRS_MHZ = 3.6e9 # 1 hour in MHz^-1

try:
    SAREA = float(sys.argv[1])
    TTOT = float(sys.argv[2])
    NSIGMA = float(sys.argv[3])
except:
    print "Expects three arguments: Sarea[deg^2] t_tot[hrs] nsigma"
    sys.exit(1)

DEBUG_PLOT = True # Whether to plot fitting functions or not

NU_LINE = 1.420406 # HI emission line freq. in GHz
FULLSKY = (4.*np.pi * (180./np.pi)**2.) # deg^2 in the full sky
NGAL_MIN = 1e3 # Min. no. of galaxies to tolerate in a redshift bin
CBM = 1. #np.sqrt(1.57) # Correction factor due to effective beam for MID/MK (OBSOLETE)
CTH = 0.5 # Correction factor due to taking 5 sigma (not 10 sigma) cuts for SKA1
SBIG = 500. # Flux rms to extrapolate dn/dz out to (constrains behaviour at large Srms)
DZBINS = 0.05 #0.1 # Width of redshift bins

# Define fiducial cosmology and parameters
# Planck 2015 flat LambdaCDM best-fit parameters (from arXiv:1502.01589, Table 3)
# (Planck TT,TE,EE+lowP)
cosmo = {
    'omega_M_0':        0.31387,
    'omega_lambda_0':   0.68613,
    'omega_b_0':        0.04917,
    'N_eff':            3.046,
    'h':                0.6727,
    'ns':               0.9645,
    'sigma_8':          0.831,
    'gamma':            0.55,
    'w0':               -1.,
    'wa':               0.,
    'fNL':              0.,
    'mnu':              0.,
    'k_piv':            0.05,
    'A':                1.,
    'sigma_nl':         7.,
}

#-------------------------------------------------------------------------------
# Survey specifications
#-------------------------------------------------------------------------------

# SKA1-MID B2 Rebaselined
name = "SKA1-MID B2 Design"
numin = 950.
numax = 1420.
Sarea = SAREA #5e3
ttot = TTOT #1e4
dnu = 0.01
Nsig = NSIGMA

# Tinst1, Ddish1, Ndish1, effic1, numin1, numax1
# Tinst = T_recv
expt1 = (7.5, 15., 133., 0.85, 950., 1420.) # MID B2 Design Baseline
expt2 = (30., 13.5, 64., 0.85, 900., 1420.) # MeerKAT L-band

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
    # Tsky = T_CMB + T_atm + T_gal
    Tsky = 2.73 + 3. + 25.2*(408./nu)**2.75 # [K]
    Tsys = Tinst + Tsky # [K]
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
    # Tsky = T_CMB + T_atm + T_gal
    Tsky = 2.73 + 3. + 25.2*(408./nu)**2.75 # [K]
    Tsys1 = Tinst1 + Tsky
    Tsys2 = Tinst2 + Tsky
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

def Ez(cosmo, z):
    """
    Dimensionless Hubble rate.
    """
    a = 1. / (1. + z)
    w0 = cosmo['w0']; wa = cosmo['wa']
    om = cosmo['omega_M_0']; ol = cosmo['omega_lambda_0']
    ok = 1. - om - ol
    omegaDE = ol * np.exp(3.*wa*(a - 1.)) / a**(3.*(1. + w0 + wa))
    return np.sqrt( om * a**(-3.) + ok * a**(-2.) + omegaDE )

def fgrowth(cosmo, z, usegamma=False):
    """
    Generalised form for the growth rate.
    
    Parameters
    ----------
    cosmo : dict
        Standard cosmological parameter dictionary.
    
    z : array_like of floats
        Redshifts.
        
    usegamma : bool, optional
        Override the MG growth parameters and just use the standard 'gamma' 
        parameter.
    """
    c = cosmo
    Oma = c['omega_M_0'] * (1.+z)**3. / Ez(cosmo, z)**2.
    a = 1. / (1. + z)
    
    # Modified gravity parameters
    if 'gamma0' not in c.keys() or usegamma == True:
        gamma = c['gamma']
    else:
        gamma = c['gamma0'] + c['gamma1']*(1. - a)
    eta = 0. if 'eta0' not in c.keys() else (c['eta0'] + c['eta1']*(1. - a))
    f = Oma**gamma * (1. + eta)
    return f

def background_evolution_splines(cosmo, zmax=10., nsamples=500):
    """
    Get interpolation functions for background functions of redshift:
      * H(z), Hubble rate in km/s/Mpc
      * r(z), comoving distance in Mpc
      * D(z), linear growth factor
      * f(z), linear growth rate
    """
    _z = np.linspace(0., zmax, nsamples)
    a = 1. / (1. + _z)
    H0 = (100.*cosmo['h']); w0 = cosmo['w0']; wa = cosmo['wa']
    om = cosmo['omega_M_0']; ol = cosmo['omega_lambda_0']
    ok = 1. - om - ol
    
    # Sample Hubble rate H(z) and comoving dist. r(z) at discrete points
    omegaDE = ol * np.exp(3.*wa*(a - 1.)) / a**(3.*(1. + w0 + wa))
    E = np.sqrt( om * a**(-3.) + ok * a**(-2.) + omegaDE )
    _H = H0 * E
    
    r_c = np.concatenate( ([0.], scipy.integrate.cumtrapz(1./E, _z)) )
    if ok > 0.:
        _r = C/(H0*np.sqrt(ok)) * np.sinh(r_c * np.sqrt(ok))
    elif ok < 0.:
        _r = C/(H0*np.sqrt(-ok)) * np.sin(r_c * np.sqrt(-ok))
    else:
        _r = (C/H0) * r_c
    
    # Integrate linear growth rate to find linear growth factor, D(z)
    # N.B. D(z=0) = 1.
    a = 1. / (1. + _z)
    _f = fgrowth(cosmo, _z)
    _D = np.concatenate( ([0.,], scipy.integrate.cumtrapz(_f, np.log(a))) )
    _D = np.exp(_D)
    
    # Construct interpolating functions and return
    r = scipy.interpolate.interp1d(_z, _r, kind='linear', bounds_error=False)
    H = scipy.interpolate.interp1d(_z, _H, kind='linear', bounds_error=False)
    D = scipy.interpolate.interp1d(_z, _D, kind='linear', bounds_error=False)
    f = scipy.interpolate.interp1d(_z, _f, kind='linear', bounds_error=False)
    return H, r, D, f


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
cosmo_fns = background_evolution_splines(cosmo)
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
knl = 0.14 * (1. + zzc)**(2./(2. + cosmo['ns']))
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

