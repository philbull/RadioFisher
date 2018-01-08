#!/usr/bin/python
"""
Test MG growth rate.
"""
import numpy as np
import pylab as P
import radiofisher as rf

cosmo = rf.experiments.cosmo

z = np.linspace(0., 3., 10)
f1 = [rf.mg_growth.growth_k(zz, cosmo)[0](1e-2) for zz in z]
f2 = [rf.mg_growth.growth_k(zz, cosmo)[0](1e-1) for zz in z]
f3 = [rf.mg_growth.growth_k(zz, cosmo)[0](1e0) for zz in z]

P.subplot(111)
P.plot(z, f1, lw=1.5)
P.plot(z, f2, lw=1.5)
P.plot(z, f3, lw=1.5)


P.plot(z, rf.mg_growth.omegaM(1./(1.+z), cosmo)**0.55, 'y--', lw=1.5)

P.show()
