#!/usr/bin/python
"""
Test A_eff / T_sys for a given experiment.
"""
import numpy as np
import pylab as P
import radiofisher.experiments as e

expt = e.MID_B1_Base

nu = np.linspace(300., 1100., 1000)

effic = 0.7
Tinst = 28.
Tsky = 60 * (300. / nu)**2.55 # Temp. of sky (mK)
Tsys = expt['Tinst']/1e3 + Tsky
Aeff = effic * 0.25 * np.pi * expt['Ddish']**2.

print Aeff
print Tsys

# Plotting
P.subplot(111)
P.plot(nu, Aeff / Tsys, 'r-', lw=1.8)

P.ylim((0., 6.))
P.xlabel("Freq. [MHz]")
P.ylabel("A_eff / T_sys [m^2/K]")

P.grid()
P.tight_layout()
P.show()
