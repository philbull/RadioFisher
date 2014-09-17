#!/usr/bin/python

import numpy as np
import pylab as P
from rfwrapper import rf


cosmo = rf.experiments.cosmo
fg = rf.experiments.foregrounds

"""
    'A':     [57.0, 0.014, 700., 0.088],        # FG noise amplitude [mK^2]
    'nx':    [1.1, 1.0, 2.4, 3.0],              # Angular scale powerlaw index
    'mx':    [-2.07, -2.10, -2.80, -2.15],      # Frequency powerlaw index
    'l_p':   1000.,                             # Reference angular scale
    'nu_p':  130.                               # Reference frequency [MHz]
"""
nu_line = 1420.406


nu = np.linspace(300., nu_line, 1000)
z = nu_line/nu - 1.

# Sky temp.
Tsky = 60e3 * (300./nu)**2.55 # Foreground sky signal (mK)

# Brightness temp.
Tb = rf.Tb(z, cosmo)

# Foreground cov.
lbl = ['Extragal. ptsrc.', 'Extragal. free-free', 'Gal. synch.', 'Gal. free-free']
for i in range(len(fg['A'])):
    y = fg['A'][i] * (nu / fg['nu_p'])**fg['mx'][i]
    P.plot(nu, y, label=lbl[i], lw=1.5)

P.plot(nu, Tsky**2., label="T_sky", lw=1.5)
P.plot(nu, Tb**2., label="T_b", lw=1.5)

P.xlim((400., 1400.))

P.legend(loc='upper right')
P.yscale('log')
P.show()
