#!/usr/bin/python
"""
Find an array configuration under the constant uv density approximation that 
gives a broadly similar n(u) to an actual array.
"""

import numpy as np
import pylab as P
import baofisher
from experiments import cosmo

u = np.logspace(0., 6.5, 5000)

def load_nu(u, fname, Ddish, z=0.):
    """
    Load n(u) interpolation function for a real experiment.
    """
    nu = 1420. / (1. + z) # in MHz
    l = 3e8 / (nu * 1e6) # Wavelength (m)
    
    # Load actual array config and convert, n(u) = n(x) / nu^2
    n_x = baofisher.load_interferom_file(fname)
    x = u / nu
    n_u = n_x(x) / nu**2.
    fov = (1.02 / (nu * Ddish) * (3e8 / 1e6))**2.
    
    print "FOV   = %3.3f deg^2" % (fov * (180./np.pi)**2.)
    return n_u, fov

def const_nu(u, Ndish, Dmin, Dmax, Ddish=None, ff=None, z=0.):
    """
    Return n(u) for assumed constant uv-plane density
    """
    nu = 1420. / (1. + z) # in MHz
    l = 3e8 / (nu * 1e6) # Wavelength (m)
    
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
    
    # Calculate sensitivity and k scale
    kperp = 2.*np.pi*u / r(z)
    sens = fov / const_nu
    
    print "-" * 50
    print "Ddish = %3.3f m" % Ddish
    print "Fill. = %3.3f" % ff
    print "FOV   = %3.3f deg^2" % (fov * (180./np.pi)**2.)
    print "Angle = %3.3f deg" % np.sqrt(fov * (180./np.pi)**2.)
    print "Min. angle: %3.3f deg" % ((1.22*l/Dmax) * 180./np.pi)
    print "Max. angle: %3.3f deg" % ((1.22*l/Dmin) * 180./np.pi)
    #return const_nu, fov
    return kperp, sens



# Get f_bao(k) function
cosmo_fns = baofisher.background_evolution_splines(cosmo)
cosmo = baofisher.load_power_spectrum(cosmo, "cache_pk.dat", force_load=True)
H, r, D, f = cosmo_fns
fbao = cosmo['fbao']


k0, Ms0 = const_nu(u, Ndish=1500, Dmin=2., Dmax=85., Ddish=2., z=1.)


print "KZN"
k0, Ls0 = const_nu(u, Ndish=1225, Dmin=5., Dmax=200., Ddish=5., z=1.) # exptL



print "MFAA"
k0, Ls0 = const_nu(u, Ndish=250*31, Dmin=0.1, Dmax=500., Ddish=2.4, z=1.) # exptL
exit()

"""
# Experiment configs
k0, Ms0 = const_nu(u, Ndish=160, Dmin=4., Dmax=60., Ddish=4., z=0.2) # exptM
k1, Ms1 = const_nu(u, Ndish=160, Dmin=4., Dmax=60., Ddish=4., z=1.) # exptM
k2, Ms2 = const_nu(u, Ndish=160, Dmin=4., Dmax=60., Ddish=4., z=2.) # exptM

k0, Ls0 = const_nu(u, Ndish=250, Dmin=15., Dmax=600., Ddish=15., z=0.2) # exptL
k1, Ls1 = const_nu(u, Ndish=250, Dmin=15., Dmax=600., Ddish=15., z=1.) # exptL
k2, Ls2 = const_nu(u, Ndish=250, Dmin=15., Dmax=600., Ddish=15., z=2.) # exptL

k0, Os0 = const_nu(u, Ndish=250, Dmin=2.5, Dmax=70., Ddish=2.5, z=0.2) # optimal
k1, Os1 = const_nu(u, Ndish=250, Dmin=2.5, Dmax=70., Ddish=2.5, z=1.) # optimal
k3, Os2 = const_nu(u, Ndish=250, Dmin=2.5, Dmax=70., Ddish=2.5, z=2.) # optimal
# From BAOBAB paper, Ddish~1.8m, Dmin~1.6m, Dmax~60m
"""

# Real experiments
nn1, f1 = load_nu(u, "array_config/nx_SKAMREF2COMP_dec30.dat", Ddish=15.)
nn2, f2 = load_nu(u, "array_config/nx_CHIME_800.dat", Ddish=5.)


P.plot(u, f1/nn1, 'm-', lw=1.5, label="SKAMREF2-COMP")
P.plot(u, f2/nn2, 'c-', lw=1.5, label="CHIME")

P.xscale('log')
P.yscale('log')
P.ylim((1e-6, 1e4))

P.legend(loc='upper right', prop={'size':'x-small'})

P.show()




exit()
nn2, f2 = load_nu(u, "array_config/nx_SKAMREF2_dec30.dat", Ddish=15.)
nn3, f3 = load_nu(u, "array_config/nx_SKAM190_dec90.dat", Ddish=15.)

nn4, f4 = load_nu(u, "array_config/nx_CHIME_800.dat", Ddish=2.3, z=0.775) # 800 MHz
nn5, f5 = load_nu(u, "array_config/nx_CHIME_400.dat", Ddish=2.3, z=0.775) # 800 MHz

# Plot results
P.subplot(111)

#P.plot(k0, Ms0, 'r-', lw=1.5, label="M")
#P.plot(k1, Ms1, 'm-', lw=1.5)
#P.plot(k2, Ms2, 'y-', lw=1.5)

#P.plot(k0, Ls0, 'b-', lw=1.5, label="L")
#P.plot(k1, Ls1, 'c-', lw=1.5)
#P.plot(k2, Ls2, 'g-', lw=1.5)

P.plot(k0, Os0, 'k-', lw=1.5, label="O")
P.plot(k1, Os1, color='0.5', lw=1.5)
P.plot(k2, Os2, color='0.7', lw=1.5)

P.plot(u, f1/nn1, 'm-', lw=1.5, label="SKAMREF2-COMP")
P.plot(u, f2/nn2, 'c-', lw=1.5, label="SKAMREF2")
P.plot(u, f3/nn3, 'y-', lw=1.5, label="SKAM-190")

P.plot(u, f4/nn4, 'g-', lw=1.5, label="CHIME")
P.plot(u, f5/nn5, 'r-', lw=1.5, label="CHIME400")

#min2 = np.min(f2/nn2)
#P.axhline(min2, color='c')
#P.axhline(20.*min2, color='c')

#P.plot(u, fov1/n1, 'r-', lw=1.5, label="A")
#P.plot(u, fov2/n2, 'b-', lw=1.5, label="B")
#P.plot(u, fov3/n3, 'g-', lw=1.5, label="C")


P.plot(k1, 10. + (150.*fbao(k1)), 'k-')

P.xscale('log')
P.yscale('log')
P.ylim((1e-2, 1e4))

P.legend(loc='upper right', prop={'size':'x-small'})

P.show()
