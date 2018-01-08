#!/usr/bin/python
"""
Calculate symmetric PSF for a given n(u) distribution.
Phil Bull 2014
"""
import numpy as np
import pylab as P
import baofisher
import experiments as e
import scipy.integrate
import scipy.special

z = 1.
nu = 1420. / (1.+z)
expt = e.SKA1MIDbase1 # only needed for n(x)

def psf(p):
    """
    Calculate PSF at a given theta.
    
    A(p) * PSF(p) = 2 pi \int n(u) u J_0(2 pi u p) du
    
    (where p = theta and A(p) = primary beam pattern)
    """
    # FIXME: Expression needs checking!
    y = n_u * u * scipy.special.jv(0, 2.*np.pi*u*p)
    return 2.*np.pi * scipy.integrate.simps(y, u)

# Load n(u)
u = np.linspace(0., 1e4, 1e3)
n_x = baofisher.load_interferom_file(expt['n(x)'])
x = u / nu  # x = u / (freq [MHz])
n_u = n_x(x) / nu**2. # n(x) = n(u) * nu^2

# Calculate PSF
p = np.logspace(-6, np.log10(0.5*np.pi/180.), 200)
B = [psf(pp) for pp in p]

P.plot(p, B)
#P.xscale('log')

P.ylabel("PSF(theta)")
P.xlabel("theta")
P.show()
