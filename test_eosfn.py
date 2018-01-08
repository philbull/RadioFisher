#!/usr/bin/python
"""
Test the function that maps from EOS 
"""
import numpy as np
import pylab as P
import scipy.integrate
import scipy.interpolate
import radiofisher as rf
from radiofisher.experiments import cosmo

C = 3e5

ax1 = P.subplot(111)

def old_eos_fisher_matrix_derivs(cosmo, cosmo_fns):
    """
    Pre-calculate derivatives required to transform (aperp, apar) into dark 
    energy parameters (Omega_k, Omega_DE, w0, wa, h, gamma).
    
    Returns interpolation functions for d(f,a_perp,par)/d(DE params) as fn. of a.
    """
    w0 = cosmo['w0']; wa = cosmo['wa']
    om = cosmo['omega_M_0']; ol = cosmo['omega_lambda_0']
    ok = 1. - om - ol
    
    # Omega_DE(a) and E(a) functions
    omegaDE = lambda a: ol * np.exp(3.*wa*(a - 1.)) / a**(3.*(1. + w0 + wa))
    E = lambda a: np.sqrt( om * a**(-3.) + ok * a**(-2.) + omegaDE(a) )
    
    # Derivatives of E(z) w.r.t. parameters
    #dE_omegaM = lambda a: 0.5 * a**(-3.) / E(a)
    if np.abs(ok) < 1e-7: # Effectively zero
        dE_omegak = lambda a: 0.5 * a**(-2.) / E(a)
    else:
        dE_omegak = lambda a: 0.5 * a**(-2.) / E(a) * (1. - 1./a)
    dE_omegaM = lambda a: 0.5 * a**(-3.) / E(a)
    dE_omegaDE = lambda a: 0.5 / E(a) * (1. - 1./a**3.)
    dE_w0 = lambda a: -1.5 * omegaDE(a) * np.log(a) / E(a)
    dE_wa = lambda a: -1.5 * omegaDE(a) * (np.log(a) + 1. - a) / E(a)
    
    # Bundle functions into list (for performing repetitive operations with them)
    fns = [dE_omegak, dE_omegaDE, dE_w0, dE_wa]
    
    # Set sampling of scale factor, and precompute some values
    HH, rr, DD, ff = cosmo_fns
    aa = np.linspace(1., 1e-4, 500)
    zz = 1./aa - 1.
    EE = E(aa); fz = ff(aa)
    gamma = cosmo['gamma']; H0 = 100. * cosmo['h']; h = cosmo['h']
    
    # Derivatives of apar w.r.t. parameters
    derivs_apar = [f(aa)/EE for f in fns]
    
    # Derivatives of f(z) w.r.t. parameters
    f_fac = -gamma * fz / EE
    df_domegak  = f_fac * (EE/om + dE_omegak(aa))
    df_domegaDE = f_fac * (EE/om + dE_omegaDE(aa))
    df_w0 = f_fac * dE_w0(aa)
    df_wa = f_fac * dE_wa(aa)
    df_dh = np.zeros(aa.shape)
    df_dgamma = fz * np.log(rf.omegaM_z(zz, cosmo))
    derivs_f = [df_domegak, df_domegaDE, df_w0, df_wa, df_dh, df_dgamma]
    
    # Calculate comoving distance (including curvature)
    r_c = scipy.integrate.cumtrapz(1./(aa**2. * EE)) # FIXME!
    r_c = np.concatenate(([0.], r_c))
    if ok > 0.:
        r = C/(H0*np.sqrt(ok)) * np.sinh(r_c * np.sqrt(ok))
    elif ok < 0.:
        r = C/(H0*np.sqrt(-ok)) * np.sin(r_c * np.sqrt(-ok))
    else:
        r = C/H0 * r_c
    
    # Perform integrals needed to calculate derivs. of aperp
    # FIXME: No factor of 2!
    print "*"*190
    derivs_aperp = [(C/H0)/r[1:] * scipy.integrate.cumtrapz( f(aa)/(aa * EE)**2.) 
                        for f in fns] # FIXME
    
    # Add additional term to curvature integral (idx 1)
    # N.B. I think Pedro's result is wrong (for fiducial Omega_k=0 at least), 
    # so I'm commenting it out
    #derivs_aperp[1] -= (H0 * r[1:] / C)**2. / 6.
    
    # Add initial values (to deal with 1/(r=0) at origin)
    inivals = [0.5, 0.0, 0., 0.] # FIXME: Are these OK?
    derivs_aperp = [ np.concatenate(([inivals[i]], derivs_aperp[i])) 
                     for i in range(len(derivs_aperp)) ]
    
    # Add (h, gamma) derivs to aperp,apar
    derivs_aperp += [np.ones(aa.shape)/h, np.zeros(aa.shape)]
    derivs_apar  += [np.ones(aa.shape)/h, np.zeros(aa.shape)]
    
    # Construct interpolation functions
    interp_f     = [scipy.interpolate.interp1d(aa[::-1], d[::-1], 
                    kind='linear', bounds_error=False) for d in derivs_f]
    interp_apar  = [scipy.interpolate.interp1d(aa[::-1], d[::-1], 
                    kind='linear', bounds_error=False) for d in derivs_apar]
    interp_aperp = [scipy.interpolate.interp1d(aa[::-1], d[::-1], 
                    kind='linear', bounds_error=False) for d in derivs_aperp]
    return [interp_f, interp_aperp, interp_apar]


def eos_fisher_matrix_derivs(cosmo, cosmo_fns):
    """
    Pre-calculate derivatives required to transform (aperp, apar) into dark 
    energy parameters (Omega_k, Omega_DE, w0, wa, h, gamma).
    
    Returns interpolation functions for d(f,a_perp,par)/d(DE params) as fn. of a.
    """
    w0 = cosmo['w0']; wa = cosmo['wa']
    om = cosmo['omega_M_0']; ol = cosmo['omega_lambda_0']
    ok = 1. - om - ol
    
    # Omega_DE(a) and E(a) functions
    omegaDE = lambda a: ol * np.exp(3.*wa*(a - 1.)) / a**(3.*(1. + w0 + wa))
    E = lambda a: np.sqrt( om * a**(-3.) + ok * a**(-2.) + omegaDE(a) )
    
    # Derivatives of E(z) w.r.t. parameters
    #dE_omegaM = lambda a: 0.5 * a**(-3.) / E(a)
    if np.abs(ok) < 1e-7: # Effectively zero
        dE_omegak = lambda a: 0.5 * a**(-2.) / E(a)
    else:
        dE_omegak = lambda a: 0.5 * a**(-2.) / E(a) * (1. - 1./a)
    dE_omegaM = lambda a: 0.5 * a**(-3.) / E(a)
    dE_omegaDE = lambda a: 0.5 / E(a) * (1. - 1./a**3.)
    dE_w0 = lambda a: -1.5 * omegaDE(a) * np.log(a) / E(a)
    dE_wa = lambda a: -1.5 * omegaDE(a) * (np.log(a) + 1. - a) / E(a)
    
    # Bundle functions into list (for performing repetitive operations with them)
    fns = [dE_omegak, dE_omegaDE, dE_w0, dE_wa]
    
    # Set sampling of scale factor, and precompute some values
    HH, rr, DD, ff = cosmo_fns
    aa = np.linspace(1., 1e-4, 500)
    zz = 1./aa - 1.
    EE = E(aa); fz = ff(aa)
    gamma = cosmo['gamma']; H0 = 100. * cosmo['h']; h = cosmo['h']
    
    # Derivatives of apar w.r.t. parameters
    derivs_apar = [f(aa)/EE for f in fns]
    
    # Derivatives of f(z) w.r.t. parameters
    f_fac = -gamma * fz / EE
    df_domegak  = f_fac * (EE/om + dE_omegak(aa))
    df_domegaDE = f_fac * (EE/om + dE_omegaDE(aa))
    df_w0 = f_fac * dE_w0(aa)
    df_wa = f_fac * dE_wa(aa)
    df_dh = np.zeros(aa.shape)
    df_dgamma = fz * np.log(rf.omegaM_z(zz, cosmo)) # FIXME: rf.omegaM_z
    derivs_f = [df_domegak, df_domegaDE, df_w0, df_wa, df_dh, df_dgamma]
    
    # Calculate comoving distance (including curvature)
    r_c = scipy.integrate.cumtrapz(1./(aa**2. * EE), aa) # FIXME!
    r_c = np.concatenate(([0.], r_c))
    if ok > 0.:
        r = C/(H0*np.sqrt(ok)) * np.sinh(r_c * np.sqrt(ok))
    elif ok < 0.:
        r = C/(H0*np.sqrt(-ok)) * np.sin(r_c * np.sqrt(-ok))
    else:
        r = C/H0 * r_c
    
    # Perform integrals needed to calculate derivs. of aperp
    print "*"*190
    derivs_aperp = [(C/H0)/r[1:] * scipy.integrate.cumtrapz( f(aa)/(aa * EE)**2., aa) 
                        for f in fns] # FIXME
    
    # Add additional term to curvature integral (idx 1)
    # N.B. I think Pedro's result is wrong (for fiducial Omega_k=0 at least), 
    # so I'm commenting it out
    #derivs_aperp[1] -= (H0 * r[1:] / C)**2. / 6.
    
    # Add initial values (to deal with 1/(r=0) at origin)
    inivals = [0.5, 0.0, 0., 0.] # FIXME: Are these OK?
    derivs_aperp = [ np.concatenate(([inivals[i]], derivs_aperp[i])) 
                     for i in range(len(derivs_aperp)) ]
    
    # Add (h, gamma) derivs to aperp,apar
    derivs_aperp += [np.ones(aa.shape)/h, np.zeros(aa.shape)]
    derivs_apar  += [np.ones(aa.shape)/h, np.zeros(aa.shape)]
    
    # Construct interpolation functions
    interp_f     = [scipy.interpolate.interp1d(aa[::-1], d[::-1], 
                    kind='linear', bounds_error=False) for d in derivs_f]
    interp_apar  = [scipy.interpolate.interp1d(aa[::-1], d[::-1], 
                    kind='linear', bounds_error=False) for d in derivs_apar]
    interp_aperp = [scipy.interpolate.interp1d(aa[::-1], d[::-1], 
                    kind='linear', bounds_error=False) for d in derivs_aperp]
    return [interp_f, interp_aperp, interp_apar]


# Precompute cosmo functions
cosmo_fns = rf.background_evolution_splines(cosmo)

# OLD
old_f, old_aperp, old_apar = old_eos_fisher_matrix_derivs(cosmo, cosmo_fns)

# NEW
new_f, new_aperp, new_apar = eos_fisher_matrix_derivs(cosmo, cosmo_fns)

z = np.linspace(0., 7., 1000)
a = 1. / (1. + z)


# Plot results
P.subplot(111)
cols = ['r', 'g', 'b', 'y', 'c', 'm']
for i in range(len(new_f)):
    P.plot(z, old_f[i](a), lw=1.5, color=cols[i], alpha=0.4)
    P.plot(z, new_f[i](a), lw=1.5, color=cols[i], ls='dashed')

P.show()
