#!/usr/bin/python
"""
Solve modified gravity linear growth rate equation from arXiv:1310.1086 and 
calculate finite difference derivatives.
"""
import numpy as np
import scipy.integrate
import scipy.interpolate
import copy

def xi(a, k, cosmo):
    """
    Modified growth parameter, mu / gamma.
    """
    w0 = cosmo['w0']; wa = cosmo['wa']
    k0 = 10.**cosmo['logkmg']; A_xi = cosmo['A_xi']
    odea = np.exp(3.*wa*(a - 1.)) / a**(3.*(1. + w0 + wa)) # Omega_DE(a)/Omega_DE(1)
    return 1. + A_xi * odea * (1. + (k0/k)**2.)

def w(a, cosmo):
    """
    Effective equation of state parameter.
    """
    return cosmo['w0'] + cosmo['wa'] * (1 - a)

def omegaM(a, cosmo):
    """
    Fractional matter density as a function of scale factor.
    """
    om = cosmo['omega_M_0']; ol = cosmo['omega_lambda_0']
    w0 = cosmo['w0']; wa = cosmo['wa']
    ok = 1. - ol - om
    omegaDE = ol * np.exp(3.*wa*(a - 1.)) / a**(3.*(1. + w0 + wa))
    E2 = om * a**(-3.) + ok * a**(-2.) + omegaDE
    return om * a**-3. / E2

def dfDdloga(y, x, k, cosmo):
    """
    Integrand for growth rate and growth function simultaneously.
    (Eq. 13 of 1310.1086)
    """
    f, logD = y
    a = np.exp(x)
    oma = omegaM(a, cosmo)
    ww = cosmo['w0'] + cosmo['wa'] * (1 - a)
    
    # Integrands for growth equation
    q = 0.5 * ( 1. - 3.*ww*(1. - oma) )
    _xi = xi(a, k, cosmo)
    dfdx = 1.5 * oma * _xi - f * (f + q)
    dlogDdx = f
    return [dfdx, dlogDdx]

def growth_k(z, cosmo, kmin=1e-4, kmax=1e2, kref=1e-1, nsamp=100):
    """
    Find growth rate f and growth function D as a function of k for a given 
    redshift.
    """
    kk = np.logspace(np.log10(kmin), np.log10(kmax), nsamp)
    aa = [1e-2, 1./(1. + z), 1.]
    x = np.log(aa)
    f = []; D = []
    
    # Calculate reference value of D(k=kref, a=1) = 1, for normalisation
    ff, logD = scipy.integrate.odeint(dfDdloga, [1.,0.], x, args=(kref, cosmo)).T
    logDref = logD[-1]
    
    # Loop through specified k values
    for k in kk:
        ff, logD = scipy.integrate.odeint(dfDdloga, [1.,0.], x, args=(k, cosmo)).T
        f.append(ff[1])
        D.append( np.exp(logD[1] - logDref) )
    
    # Interpolate growth rate and growth function
    ff = scipy.interpolate.interp1d(kk, f, kind='linear', bounds_error=False, 
                                    fill_value=0.)
    DD = scipy.interpolate.interp1d(kk, D, kind='linear', bounds_error=False,
                                    fill_value=0.)
    return ff, DD


def growth_derivs(zc, k, cosmo, mg_params=['A_xi', 'logkmg'], dx=[1e-3, 1e-3]):
    """
    Calculate derivatives of growth rate as a function of scale, with respect 
    to specified modified gravity parameters.
    
    Parameters
    ----------
    
    zc : float
        Redshift at which to evaluate derivatives.
    
    k : array_like
        k values at which to evaluate derivatives.
    
    cosmo : dict
        Dictionary of fiducial cosmological parameters.
        
    mg_params : list of str
        Names of modified gravity parameters in 'cosmo' to take derivatives wrt.
    
    dx : array_like
        Respective finite difference values for the MG params, dx, for 
        evaluating df/dx.
    
    Returns
    -------
    
    derivs : list of array_like
        List of derivatives of f(z,k) wrt MG parameters.
    """
    # Sanity check
    for param in mg_params:
        if param not in cosmo.keys():
            raise KeyError("MG parameter not found in 'cosmo' dictionary: %s" % param)
    
    # Evaluate fiducial point
    f0, D0 = growth_k(zc, cosmo)
    f0_k = f0(k)
    
    # Loop through parameters and get finite differences
    derivs = []
    for i in range(len(mg_params)):
        c = copy.deepcopy(cosmo) # Get unmodified cosmo dict.
        c[mg_params[i]] += dx[i]
        ffp, DDp = growth_k(zc, c)
        derivs.append( (ffp(k) - f0_k) / dx[i] )
    return derivs

