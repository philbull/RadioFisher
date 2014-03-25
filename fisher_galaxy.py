#!/usr/bin/python
"""
Calculate Fisher matrix for a galaxy redshift survey, using the formalism 
from the Euclid cosmology white paper (arXiv:1206.1225; see Sect. 1.7.3).
"""
import numpy as np
import pylab as P
import scipy.integrate
import baofisher as rf
from units import *
import copy

RSD_FUNCTION = 'not kaiser'

def Csignal_galaxy(q, y, cosmo):
    """
    Get (q,y)-dependent factors of the signal covariance matrix for a galaxy 
    redshift survey.
    """
    c = cosmo
    
    # Wavenumber and mu = cos(theta)
    kperp = q / (c['aperp']*c['r'])
    kpar = y / (c['apar']*c['rnu'])
    k = np.sqrt(kpar**2. + kperp**2.)
    u2 = (kpar / k)**2.
    
    # RSD function (bias 'btot' already includes scale-dep. bias/non-Gaussianity)
    if RSD_FUNCTION == 'kaiser':
        # Pedro's notes, Eq. 7
        Frsd = (c['bgal'] + c['f']*u2)**2. * np.exp(-u2*(k*c['sigma_nl'])**2.)
    else:
        # arXiv:0812.0419, Eq. 5
        sigma_nl2_eff = (c['D'] * c['sigma_nl'])**2. * (1. - u2 + u2*(1.+c['f'])**2.)
        Frsd = (c['bgal'] + c['f']*u2)**2. * np.exp(-0.5 * k**2. * sigma_nl2_eff)
    
    # Construct signal covariance and return
    cs = Frsd * (1. + c['A'] * c['fbao'](k)) * c['D']**2. * c['pk_nobao'](k)
    cs *= c['aperp']**2. * c['apar']
    return cs


def fisher_galaxy_survey( zmin, zmax, ngal, cosmo, expt, cosmo_fns, 
                          kbins=None, return_pk=False ):
    """
    Calculate Fisher matrix for a galaxy redshift survey.
    
    Parameters
    ----------
    
    zmin, zmax : float
        Min. and max. redshift of this sub-survey.
    
    ngal : float
        Mean number density of galaxies in this redshift bin.
        
    """
    # Copy, to make sure we don't modify input expt or cosmo
    cosmo = copy.deepcopy(cosmo)
    expt = copy.deepcopy(expt)
    HH, rr, DD, ff = cosmo_fns
    
    # Pack values and functions into the dictionaries cosmo, expt
    z = 0.5 * (zmin + zmax) # Survey central redshift
    cosmo['z'] = z; cosmo['f'] = ff(z); cosmo['D'] = DD(z)
    cosmo['r'] = rr(z); cosmo['rnu'] = C*(1.+z)**2. / HH(z) # Perp/par. dist. scales
    
    # Use effective bias parameter; setup n(z)
    cosmo['bHI'] = cosmo['bgal'] = np.sqrt(1. + z)
    cosmo['ngal'] = ngal
    
    # Calculate Vsurvey
    _z = np.linspace(zmin, zmax, 1000)
    Vsurvey = C * scipy.integrate.simps(rr(_z)**2. / HH(_z), _z)
    Vsurvey *= 4. * np.pi * expt['fsky']
    print "\tSurvey volume: %3.2f Gpc^3" % (Vsurvey/1e9)
    
    # Define kmin, kmax
    kmin = 2.*np.pi / Vsurvey**(1./3.)
    # Eq. 20 of Smith et al. 2003 (arXiv:astro-ph/0207664v2)
    kmax = expt['k_nl0'] * (1.+z)**(2./(2. + cosmo['ns']))
    
    print "\t   z = %3.2f" % z
    print "\tkmin = %4.4f Mpc^-1" % kmin
    print "\tkmax = %4.4f Mpc^-1" % kmax
    
    # Set-up integration sample points in (k, u)-space (FIXME: Limits)
    ugrid = np.linspace(-1., 1., rf.NSAMP_U) # N.B. Order of integ. limits is correct
    kgrid = np.logspace(np.log10(kmin), np.log10(kmax), rf.NSAMP_K)
    
    # Calculate fbao(k) derivative
    cosmo['dfbao_dk'] = rf.fbao_derivative(cosmo['fbao'], kgrid)
    
    # Sanity check on P(k)
    if kmax > cosmo['k_in_max']:
        raise ValueError(
          "Input P(k) only goes up to %3.2f Mpc^-1, but kmax is %3.2f Mpc^-1." \
          % (cosmo['k_in_max'], kmax) )
    
    # Calculate derivatives and integrate
    derivs, paramnames = rf.fisher_integrands( kgrid, ugrid, cosmo, expt=expt, 
                                   massive_nu_fn=None, transfer_fn=None, 
                                   galaxy_survey=True, cs_galaxy=Csignal_galaxy )
    Vfac = Vsurvey / (8. * np.pi**2.)
    F = Vfac * rf.integrate_fisher_elements(derivs, kgrid, ugrid)
    
    # Calculate cross-terms between binned P(k) and other params
    if return_pk:
        # Do cumulative integrals for cross-terms with P(k)
        cumul = rf.integrate_fisher_elements_cumulative(-1, derivs, kgrid, ugrid)
        
        # Calculate binned P(k) and cross-terms with other params
        kc, pbins = rf.bin_cumulative_integrals(cumul, kgrid, kbins) # FIXME: kbins
        
        # Add k-binned terms to Fisher matrix (remove non-binned P(k))
        pnew = len(cumul) - 1
        FF, paramnames = rf.fisher_with_excluded_params(F, excl=[F.shape[0]-1], 
                                                        names=paramnames)
        F_pk = Vfac * rf.expand_fisher_with_kbinned_parameter(FF / Vfac, pbins, pnew)
        
        # Construct dict. with info needed to rebin P(k) and cross-terms
        binning_info = {
          'F_base':  FF,
          'Vsurvey': Vsurvey,
          'Vfac':    Vfac,
          'cumul':   cumul,
          'kgrid':   kgrid
        }
        
        # Append paramnames
        paramnames += ["pk%d" % i for i in range(kc.size)]
    
    # Return results
    if return_pk: return F_pk, kc, binning_info, paramnames
    return F, paramnames

# FIXME:
# * What about an angular cut-off? No beam on angular scales is currently defined.
# * Sensitive to kmin? Probably not, get poor constraints there already. (Result: only slightly sensitive.)
