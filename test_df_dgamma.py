#!/usr/bin/python
"""
Plot derivative of f(z) w.r.t. gamma.
"""
import numpy as np
import pylab as P
from experiments import cosmo


z = np.linspace(0., 4., 1000)
a = 1. / (1. + z)

def df_dgamma(a, gamma):
    H0 = (100.*cosmo['h']); w0 = cosmo['w0']; wa = cosmo['wa']
    om = cosmo['omega_M_0']; ol = cosmo['omega_lambda_0']
    ok = 1. - om - ol

    omegaDE = ol * np.exp(3.*wa*(a - 1.)) / a**(3.*(1. + w0 + wa))
    E = np.sqrt( om * a**(-3.) + ok * a**(-2.) + omegaDE )
    H = H0 * E
    Oma = cosmo['omega_M_0'] * (1.+z)**3. * (100.*cosmo['h'] / H)**2.
    f = Oma**cosmo['gamma']

    # Derivative
    return f * np.log(Oma)
    

# Make plot
P.subplot(111)
P.plot(z, df_dgamma(a, 0.55), 'k-', lw=1.5)
P.show()
