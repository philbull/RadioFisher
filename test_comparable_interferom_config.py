#!/usr/bin/python
"""
Find an array configuration under the constant uv density approximation that 
gives a broadly similar n(u) to an actual array.
"""

import numpy as np
import pylab as P
import baofisher

z = 1.
nu = 1420. / (1. + z) # in MHz
l = 3e8 / (nu * 1e6) # Wavelength (m)
u = np.logspace(0., 6.5, 5000)

def load_nu(u, fname, Ddish):
    """
    Load n(u) interpolation function for a real experiment.
    """
    # Load actual array config and convert, n(u) = n(x) / nu^2
    n_x = baofisher.load_interferom_file(fname)
    x = u / nu
    n_u = n_x(x) / nu**2.
    fov = (1.02 / (nu * Ddish) * (3e8 / 1e6))**2.
    return n_u, fov

def const_nu(u, Ndish, Dmin, Dmax, Ddish=None, ff=None):
    """
    Return n(u) for assumed constant uv-plane density
    """
    # Min/max. scales and filling factor
    u_min = Dmin / l
    u_max = Dmax / l
    
    # Calculate either filling factor or Ddish, depending on what was specified
    if ff is not None:
        Ddish = Dmax * np.sqrt(ff/Ndish)
    else:
        ff = Ndish * (Ddish / Dmax)**2.

    # Calculate n(u) assuming const. uv density
    const_nu = Ndish*(Ndish - 1.)*l**2. * np.ones(u.shape) \
             / (2.*np.pi*(Dmax**2. - Dmin**2.) )
    const_nu[np.where(u < u_min)] = 1e-100
    const_nu[np.where(u > u_max)] = 1e-100
    
    # Calculate FOV
    fov = (1.02 / (nu * Ddish) * (3e8 / 1e6))**2.
    
    print "-" * 50
    print "Ddish = %3.3f m" % Ddish
    print "Fill. = %3.3f" % ff
    print "FOV   = %3.3f deg^2" % (fov * (180./np.pi)**2.)
    print "Angle = %3.3f deg" % np.sqrt(fov * (180./np.pi)**2.)
    print "Min. angle: %3.3f deg" % ((1.22*l/Dmax) * 180./np.pi)
    print "Max. angle: %3.3f deg" % ((1.22*l/Dmin) * 180./np.pi)
    return const_nu, fov


# Experiment configs
n1, fov1 = const_nu(u, Ndish=250, Dmin=15., Dmax=500., Ddish=15.)

n2, fov2 = const_nu(u, Ndish=49, Dmin=2., Dmax=60., Ddish=2.)
n3, fov3 = const_nu(u, Ndish=160, Dmin=4., Dmax=60., Ddish=4.)
# From BAOBAB paper, Ddish~1.8m, Dmin~1.6m, Dmax~60m

# Real experiments
nn1, f1 = load_nu(u, "array_config/nx_SKAMREF2COMP_dec30.dat", Ddish=15.)
nn2, f2 = load_nu(u, "array_config/nx_SKAMREF2_dec30.dat", Ddish=15.)
nn3, f3 = load_nu(u, "array_config/nx_SKAM190_dec90.dat", Ddish=15.)


# Plot results
P.subplot(111)
P.plot(u, f1/nn1, 'm-', lw=1.5, label="SKAMREF2-COMP")
P.plot(u, f2/nn2, 'c-', lw=1.5, label="SKAMREF2")
P.plot(u, f3/nn3, 'y-', lw=1.5, label="SKAM-190")

min2 = np.min(f2/nn2)

P.axhline(min2, color='c')
P.axhline(20.*min2, color='c')

P.plot(u, fov1/n1, 'r-', lw=1.5, label="A")
P.plot(u, fov2/n2, 'b-', lw=1.5, label="B")
P.plot(u, fov3/n3, 'g-', lw=1.5, label="C")
P.xscale('log')
P.yscale('log')
P.ylim((1e-2, 1e4))
#P.ylim((1e-8, 1e2))
P.legend(loc='upper right', prop={'size':'x-small'})

P.show()
