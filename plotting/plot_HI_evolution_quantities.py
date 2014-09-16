#!/usr/bin/python
"""
Compare Omega_HI, b_HI, and T_b for different models.
"""
import numpy as np
import pylab as P
import baofisher
import experiments as e

z = np.linspace(0., 5., 1000)

def poly(p, x):
    print "y =",
    for i in range(p.size):
        print "%4.4e x^%d +" % (p[i], i),
    print ""
    return np.sum([p[i] * x**i for i in range(p.size)], axis=0)

# Load Mario's new data
zz, omegaHI, bHI, Tb = np.genfromtxt("santos_powerlaw_HI_model.dat").T


# Fit simple polynomials to data
print "OMEGA_HI", omegaHI[0]
p_omegaHI = np.polyfit(zz, omegaHI, deg=2)
print "B_HI", bHI[0]
p_bHI = np.polyfit(zz, bHI, deg=2)
print "T_B", Tb[0]
p_Tb = np.polyfit(zz, Tb, deg=2)

# Omega_HI
P.subplot(311)
P.plot(z, baofisher.omega_HI(z, e.cosmo), 'r-', lw=1.5)
P.plot(zz, omegaHI, 'bo')
P.plot(zz, poly(p_omegaHI[::-1], zz), 'g-')
P.ylabel("$\Omega_{HI}(z)$")

# b_HI
P.subplot(312)
P.plot(z, baofisher.bias_HI(z, e.cosmo), 'r-', lw=1.5)
P.plot(zz, bHI, 'bo')
P.plot(zz, poly(p_bHI[::-1], zz), 'g-')
P.ylabel("$b_{HI}(z)$")

# T_b
P.subplot(313)
P.plot(z, baofisher.Tb(z, e.cosmo, formula='powerlaw'), 'r-', lw=1.5)
P.plot(z, baofisher.Tb(z, e.cosmo, formula='santos'), 'b-', lw=1.5)
P.plot(zz, poly(p_Tb[::-1], zz), 'g-')
P.plot(zz, Tb, 'bo')
P.ylabel("$T_b(z)$")

P.show()
