#!/usr/bin/python
"""
Perform HI survey Fisher forecast based on Pedro's formalism (see notes from 
August 2013).

Requires up-to-date NumPy, SciPy (tested with version 0.11.0) and matplotlib.
(Phil Bull, 2013)
"""
import numpy as np
import scipy.integrate
import scipy.interpolate
import pylab as P

# TODO
# * Series of N bins in redshift, dnutot is total for each survey, but N surveys. z=0.1 bins, <~1 GHz dnutot
# * Pedro: Figure out P(k) with error bars
# * Add aperp, apar

# Constants and conversion factors
C = 3e5
D2RAD = np.pi / 180. # Convert degrees to radians
HRS_MHZ = 3.6e9 # 1 hour in MHz^-1

# No. of samples in log space in each dimension. 300 seems stable for (AA).
NSAMP_K = 500
NSAMP_U = 1500

# FIXME: Testing, should be 1
KPAR_FACTOR = 1. #45.6
KPERP_FACTOR = 1. #10.

# Location of CAMB fiducial P(k) file
CAMB_MATTERPOWER = "/home/phil/oslo/iswfunction/cosmomc/camb/testX_matterpower.dat"

# Define cosmology
cosmo = {
	 'omega_M_0': 		0.26,
	 'omega_lambda_0': 	0.74,
	 'omega_b_0': 		0.045,
	 'omega_HI':        1e-3, # FIXME: Use a proper value
	 'omega_n_0':		0.0,
	 'omega_k_0':		0.0,
	 'N_nu':			0,
	 'h':				0.70,
	 'n':				0.96,
	 'sigma_8':			0.8,
	 'gamma':           0.55
	}

# Experimental definition
expt = {
    'Tinst':            40*(1e3),           # System temp. [mK]
    'z_survey':         1.0,                # Survey redshift
    'ttot':             1e4*HRS_MHZ,       # Total integration time [MHz^-1]
    'dnutot':           100.,               # Total survey bandwidth [MHz]
    'Sarea':            1000.*(1.*D2RAD)**2.,  # Total survey area [radians^2] [>~100deg^2]
    'dnu':              0.5,                # Bandwidth of single channel [MHz]
    'beam_fwhm':        (14.5*D2RAD),        # FWHM of single beam [radians]
    'nu_line':          1400.               # Rest-frame frequency of emission line [MHz]
    }
    

################################################################################
# Cosmology functions
################################################################################

def background_evolution_splines(cosmo, zmax=10., nsamples=400):
    """
    Get interpolation functions for background functions of redshift:
      * H(z), Hubble rate in km/s/Mpc
      * r(z), comoving distance in Mpc
    """
    # Sample Hubble rate H(z) and comoving dist. r(z) at discrete points
    _z = np.linspace(0., zmax, nsamples)
    _H = (100.*cosmo['h']) * np.sqrt(cosmo['omega_M_0']*(1.+_z)**3. + cosmo['omega_lambda_0'])
    #_r = scipy.integrate.cumtrapz(C/_H, _z, initial=0.)
    _r = np.concatenate( ([0.], scipy.integrate.cumtrapz(C/_H, _z)) )
    
    # Construct interpolating functions and return
    r = scipy.interpolate.interp1d(_z, _r, kind='linear', bounds_error=False)
    H = scipy.interpolate.interp1d(_z, _H, kind='linear', bounds_error=False)
    return H, r


def Tb(z, cosmo, formula='chang'):
    """
    Brightness temperature Tb(z), in mK. Several different expressions for the 
    21cm line brightness temperature are available:
    
     * 'hall': from Hall, Bonvin, and Challinor.
     * 'chang': from Chang et al.
    """
    if formula == 'chang':
        Tb = 0.3 * (cosmo['omega_HI']/1e-3) * np.sqrt(0.29*(1.+z)**3.)
        Tb *= np.sqrt((1.+z)/2.5)
        Tb /= np.sqrt(cosmo['omega_M_0']*(1.+z)**3. + cosmo['omega_lambda_0'])
    else:
        print "WARNING: 'hall' formula not implemented in Tb(z)."
        return 0.
        #188. * cosmo['h'] * (4.e-4) * omega_HI(z)
    return Tb


def spline_pk_nobao(k_in, pk_in, kref=[3e-2, 4.5e-1]):
    """
    Construct a smooth power spectrum with BAOs removed, and a corresponding 
    BAO template function, by using a two-stage splining process.
    """
    # Get interpolating function for input P(k) in log-log space
    _interp_pk = scipy.interpolate.interp1d( np.log(k_in), np.log(pk_in), 
                                        kind='quadratic', bounds_error=False )
    interp_pk = lambda x: np.exp(_interp_pk(np.log(x)))
    
    # Spline all (log-log) points except those in user-defined "wiggle region",
    # and then get derivatives of result
    idxs = np.where(np.logical_or(k_in <= kref[0], k_in >= kref[1]))
    _pk_smooth = scipy.interpolate.UnivariateSpline( np.log(k_in[idxs]), 
                                                    np.log(pk_in[idxs]), k=3, s=0 )
    pk_smooth = lambda x: np.exp(_pk_smooth(np.log(x)))

    # Construct "wiggles" function using spline as a reference, then spline it 
    # and find its 2nd derivative
    fwiggle = scipy.interpolate.UnivariateSpline(k_in, pk_in / pk_smooth(k_in), k=3, s=0)
    derivs = np.array([fwiggle.derivatives(_k) for _k in k_in]).T
    d2 = scipy.interpolate.UnivariateSpline(k_in, derivs[2], k=3, s=1.0) # s=1 for smoothing
    
    # Find maxima and minima of the gradient (zeros of 2nd deriv.), then put a
    # low-order spline through zeros to subtract smooth trend from wiggles fn.
    wzeros = d2.roots()
    wzeros = wzeros[np.where(np.logical_and(wzeros >= kref[0], wzeros <= kref[1]))]
    wzeros = np.concatenate((wzeros, [kref[1],]))
    wtrend = scipy.interpolate.UnivariateSpline(wzeros, fwiggle(wzeros), k=3, s=0)
    
    # Construct smooth "no-bao" function by summing the original splined function and 
    # the wiggles trend function
    idxs = np.where(np.logical_and(k_in > kref[0], k_in < kref[1]))
    pk_nobao = pk_smooth(k_in)
    pk_nobao[idxs] *= wtrend(k_in[idxs])
    fk = (pk_in - pk_nobao)/pk_nobao
    
    # Construct interpolating functions
    ipk = scipy.interpolate.interp1d( k_in, pk_nobao, kind='linear',
                                      bounds_error=False, fill_value=0. )
    ifk = scipy.interpolate.interp1d( k_in, fk, kind='linear',
                                      bounds_error=False, fill_value=0. )
    return ipk, ifk



def Csignal(q, y, vals):
    """
    Get (q,y)-dependent factors of the signal covariance matrix.
    A factor of [T_b(z)]^2 nu_line / (r^2 rnu) is missing from outfront.
    """
    A, f, aperp, apar, xp, x, r, rnu, z, nu_line, bHI, sigma, csfac, fbao, pk_nobao = vals
    
    # Wavenumber and mu = cos(theta)
    k = np.sqrt( (q/(aperp*r))**2. + (y / (apar*rnu))**2. ) # FIXME Are factors apar,aperp OK?
    #u2 = y**2. / (y**2. + ((1.+z)**2. *q*(xp/x)*(aperp/apar))**2.)
    u2 = y**2. / (y**2. + (q*(rnu/r)*(aperp/apar))**2.)
    
    # RSD function
    # sigma is in Mpc^-1 here, even though it's usually quoted in km/s?
    Frsd = (bHI + f*u2)**2. * np.exp(-u2*(k*sigma)**2.)
    return Frsd * (1. + A*fbao(k)) * pk_nobao(k)


################################################################################
# Prefactors
################################################################################

# Get cosmology-dependent factors
z = 1.
bHI = 0.8
_Tb = Tb(z, cosmo) / 1e3 # Tb in K
HH, rr = background_evolution_splines(cosmo)
Ez = HH(z) / (100. * cosmo['h'])
Oma = cosmo['omega_M_0'] * (1.+z)**3. / Ez**2.
f = Oma**cosmo['gamma']

# Import CAMB P(k) and construct interpolating function
k_in, pk_in = np.genfromtxt(CAMB_MATTERPOWER).T[:2]
k_in *= cosmo['h']; pk_in /= cosmo['h']**3. # Convert h^-1 Mpc => Mpc
pk_nobao, fbao = spline_pk_nobao(k_in, pk_in)

# Perpendicular and parallel length scales, and BAO length scales
r = rr(z)
rnu = C*(1.+z)**2. / HH(z)
x = rr(z)
xp = C / HH(z)

# Calculate noise power, C_n (== P_n?)
Vsurvey = expt['Sarea'] * expt['dnutot']
Tsys = expt['Tinst'] + (300.*(1.+z)/expt['nu_line'])**2.55
noise = Tsys**2. * Vsurvey / (expt['ttot'] * expt['dnutot'])


# Define k scales
Vphys = (expt['Sarea'] * expt['dnutot']) * r**2. * rnu / expt['nu_line']
kmax = 12.16 * 0.7 #5e0
kmin = 2.*np.pi / Vphys**(1./3.) / 10. #2.*np.pi / Vphys**(1./3.) # FIXME 1/10


################################################################################
# Integrate power spectrum integrand
################################################################################

pn = noise * (r**2. * rnu) # FIXME: Is this scaled correctly?
h = 0.7

# Transverse k modes (h/Mpc) - resolution: 0.004803   kmint: 0.000000  kmaxt: 0.010466
# Parallel k modes (h/Mpc) - resolution: 0.012498   kmaxp: 12.497977
# Foreground cut: 0.002500 (h/Mpc)
kt_min = 0.
kt_max = 0.010466*h
kp_min = 0.002500*h
kp_max = 12.497977*h

_ki = np.linspace(0., kt_max, 2) # FIXME: How many samples to define?
_kj = np.linspace(0., kt_max, 2)
kp = np.linspace(kp_min, kp_max, 1000) # Matches res. of Mario's code

print "Resolution (kt):", _ki[1] - _ki[0]
print "Resolution (kp):", kp[1] - kp[0]

# Compute transverse k grid and blank-out values that are less than kt_min
ki, kj = np.meshgrid(_ki, _kj)
kt = np.sqrt(ki**2. + kj**2.)
kt[np.where(kt < kt_min)] = 0. # Remove kt values that are less than kt_min

# Compute K grid and evaluate P(k) on it
KT, KP = np.meshgrid(kt, kp)
K = np.sqrt(KT**2. + KP**2.)
pk = pk_nobao(K)

# Define bins in k
kbins = np.logspace(np.log10(kmin), np.log10(kmax), 26)
kc = np.array([0.5*(kbins[i+1] + kbins[i]) for i in range(kbins.size-1)])

psfac = (_Tb * bHI)**2. #* expt['nu_line'] / (r**2. * rnu)

# Loop through each bin, summing values of integrand in that bin
fn = np.zeros(kc.size)
pfn = np.zeros(kc.size)
nmode = np.zeros(kc.size)
kv = np.zeros(kc.size)

print "Comparison:", pn, np.max(psfac * pk)

# Sum modes into bins
for i in range(kc.size):
    idxs = np.where(np.logical_and(K >= kbins[i], K < kbins[i+1]))
    fn[i] = np.sum( 1. / (psfac*pk[idxs] + pn)**2. )
    pfn[i] = np.sum(psfac * pk[idxs])
    nmode[i] = idxs[0].size
    kv[i] = np.sum(K[idxs])

# Load Mario's results
dat = np.genfromtxt("mario_err_dish.txt").T


res0 = kv/nmode
res1 = pfn/nmode
res2 = 0.5/np.sqrt(fn)
res3 = nmode*4.

"""
P.plot(K.flatten(), psfac*pk.flatten(), 'r,')
P.axhline(pn)
P.xscale('log')
P.yscale('log')
P.show()
exit()
"""


# Plot results
P.subplot(221)
P.plot(res0, dat[0], 'bx', ls='solid')
P.plot(res0, res0, 'k-', alpha=0.5)
#P.plot(dat[0], dat[1], 'r-')
#P.xscale('log')

P.subplot(222)
P.plot(res0, res1, 'b-')
P.plot(dat[0], dat[1]*0.7**3., 'g-')
P.xscale('log')
#P.yscale('log')


P.subplot(223)
P.plot(res0, res2, 'b-')
P.plot(dat[0], dat[2], 'g-')
P.xscale('log')

P.subplot(224)
P.plot(res0, res3, 'b-')
P.plot(dat[0], dat[3], 'g-')
P.xscale('log')


P.show()
