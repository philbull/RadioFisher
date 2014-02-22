#!/usr/bin/python
"""
Reproduce the CAMB theta variable.
"""
import numpy as np
import scipy.integrate
import scipy.interpolate
import pylab as P
import copy

C = 3e5 # km/s

################################################################################
def comoving_dist(a, cosmo):
    """
    Comoving distance. Ignores radiation, which might shift results slightly.
    """
    aa = np.logspace(np.log10(a), 0., 1000)
    zz = 1./aa - 1.
    
    # Cosmological parameters
    H0 = (100.*cosmo['h']); w0 = cosmo['w0']; wa = cosmo['wa']
    ombh2 = cosmo['omega_b_0'] * cosmo['h']**2.
    om = cosmo['omega_M_0']; ol = cosmo['omega_lambda_0']
    ogam = 2.47e-5 / cosmo['h']**2. # Rad. density fraction, from Dodelson Eq. 2.70
    Neff = 3.046
    onu = (7./8.) * (4./11.)**(4./3.) * Neff * ogam
    ok = 1. - om - ol
    
    # Omega_DE(z) and 1/E(z)
    omegaDE = ol #* np.exp(3.*wa*(aa - 1.)) / aa**(3.*(1. + w0 + wa))
    invE = 1. / np.sqrt( om*aa + ok*aa**2. + ogam + onu + omegaDE*aa**4.) # 1/(a^2 H)
    
    # Calculate r(z), with curvature-dependent parts
    r_c = scipy.integrate.simps(invE, aa)
    if ok > 0.:
        _r = C/(H0*np.sqrt(ok)) * np.sinh(r_c * np.sqrt(ok))
    elif ok < 0.:
        _r = C/(H0*np.sqrt(-ok)) * np.sin(r_c * np.sqrt(-ok))
    else:
        _r = (C/H0) * r_c
    return _r

def rsound(a, cosmo):
    """
    Calculate the sound horizon at some scale factor, a. (In Mpc)
    """
    # Uses the following expressions from Dodelson:
    # Eq. 8.19: c_s(eta) = [3 (1+R)]^(-1/2)
    # Eq. 8.22: r_s(eta) = integral_0^eta c_s(eta') d(eta')
    # p82:      R = 3/4 rho_b / rho_gamma
    # Eq. 2.71: rho_b = Omega_b a^-3 rho_cr
    # Eq. 2.69: rho_gamma = (pi^2 / 15) (T_CMB)^4
    # Eq. 2.70: Omega_gamma h^2 = 2.47e-5
    # We have also converted the integral from conformal time, deta, to da
    
    # Scale-factor samples
    aa = np.logspace(-8., np.log10(a), 1000)
    
    # Cosmological parameters
    H0 = (100.*cosmo['h']); w0 = cosmo['w0']; wa = cosmo['wa']
    ombh2 = cosmo['omega_b_0'] * cosmo['h']**2.
    om = cosmo['omega_M_0']; ol = cosmo['omega_lambda_0']
    ogam = 2.47e-5 / cosmo['h']**2. # Rad. density fraction, from Dodelson Eq. 2.70
    Neff = 3.046
    onu = (7./8.) * (4./11.)**(4./3.) * Neff * ogam
    ok = 1. - om - ol
    
    # Omega_DE(z) and E(z)
    omegaDE = ol * np.exp(3.*wa*(aa - 1.)) / aa**(3.*(1. + w0 + wa))
    
    # Integrate sound speed
    R = 3.0364e4 * ombh2 * aa # Baryon-photon ratio
    cs = np.sqrt(3. + 3.*R) # Sound speed
    rs_integ = 1. / np.sqrt( om*aa + ok*aa**2. + ogam + onu + omegaDE*aa**4.) # 1/(a^2 H)
    rs_integ /= cs
    rs = (C/H0) * scipy.integrate.simps(rs_integ, aa)
    return rs


def cmb_to_theta(cosmo, h):
    """
    Convert input CMB parameters to a theta value; taken from params_CMB.f90, 
    function CMBToTheta(CMB), which implements Hu & Sugiyama fitting formula.
    
    Theta depends on (ombh2, omdmh2) only.
    """
    # Recast cosmological parameters into CAMB parameters
    p = {}
    cosmo['h'] = h
    p['hubble'] = 100.*cosmo['h']
    p['omch2'] = (cosmo['omega_M_0'] - cosmo['omega_b_0']) * cosmo['h']**2.
    p['ombh2'] = cosmo['omega_b_0'] * cosmo['h']**2.
    p['omk'] = 1. - (cosmo['omega_M_0'] + cosmo['omega_lambda_0'])
    
    # CAMB parameters
    ombh2 = p['ombh2']; omch2 = p['omch2']
    
    # Redshift of LSS (Hu & Sugiyama fitting formula, from CAMB)
    # N.B. This is only an approximate value. CAMB can also get a more precise 
    # value. Typical difference is ~2, i.e. ~0.2%.
    zstar = 1048. * (1. + 0.00124*ombh2**-0.738) \
          * ( 1. + (0.0783*ombh2**-0.238 / (1. + 39.5*ombh2**0.763)) \
                   * (omch2 + ombh2)**(0.560/(1. + 21.1*ombh2**1.81)) )
    astar = 1. / (1. + zstar)
    
    # Get theta = rs / r (typical agreement with CAMB is ~ 0.1%)
    # (N.B. Note different definition of angular diameter distance in CAMB)
    rs = rsound(astar, cosmo)
    rstar = comoving_dist(astar, cosmo)
    theta = rs / rstar
    return theta

def find_h_for_theta100(th100, cosmo, hmin=0.5, hmax=0.85, nsamp=20):
    """
    Find the value of h that gives a particular value of 100*theta_MC in CAMB.
    Similar to the algorithm in CosmoMC:params_CMB.f90:ParamsToCMBParams().
    
    Parameters
    ----------
    th100 : float
        Target value of 100*theta_MC
    
    cosmo : dict
        Dictionary of cosmological parameters. 'h' will be ignored. For this 
        calculation, only {'omega_M_0', 'omega_b_0', 'omega_lambda_0'} matter.
    
    hmin, hmax : float, optional
        Bounds of range of h values to search inside. Default is h = [0.5, 0.85]
    
    nsamp : int, optional
        Number of samples to return for interpolation function. Default: 20.
    
    Returns
    -------
    h : float
        Value of h that gives the input value of theta_MC.
    """
    
    # Get samples
    c = copy.deepcopy(cosmo)
    hvals = np.linspace(hmin, hmax, nsamp) # Range of h values to scan
    th = 100. * np.array([cmb_to_theta(c, hh) for hh in hvals])
    
    # Interpolate sample points (trend is usually very smooth, so use quadratic)
    h_match = scipy.interpolate.interp1d(th, hvals, kind='quadratic')(th100)
    return h_match
