#!/usr/bin/python
"""
Plot BAO wiggle function.
"""

import numpy as np
import pylab as P
import baofisher
import experiments

# Choose fiducial values for parameters (f(z) is defined later)
fiducial = {
  'aperp':     1.,
  'apar':      1.,
  'bHI0':      0.702,
  'A':         1.,
  'sigma_nl':  7. #14. #0.5
}

# Load cosmology and experimental settings
cosmo = experiments.cosmo

# Precompute cosmological functions and derivs.
camb_matterpower = "/home/phil/oslo/iswfunction/cosmomc/camb/testX_matterpower.dat"
cosmo_fns, cosmo = baofisher.precompute_for_fisher(cosmo, camb_matterpower)
H, r, D, f = cosmo_fns

# Get wiggles function and smoooth P(k)
k = np.logspace(np.log10(0.02), np.log10(0.4), 2000)
fbao = cosmo['fbao'](k)
pksmooth = cosmo['pk_nobao'](k)

# Plot results
P.subplot(111)
#P.plot(k, k**1. * pksmooth)
#P.plot(k, k**1. * pksmooth * (1. + fbao))
P.plot(k, fbao, 'k-', lw=1.5)

P.xlim((np.min(k), np.max(k)))
P.xscale('log')

P.xlabel("$k \; [\mathrm{Mpc}^{-1}]$", fontdict={'fontsize':'20'})
P.ylabel("$f_\mathrm{bao}(k)$", fontdict={'fontsize':'20'})

# Legend
P.legend(loc='upper left', prop={'size':'x-large'}, ncol=2)

# Display options
fontsize = 18.
for tick in P.gca().yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in P.gca().xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
P.tight_layout()
P.show()
