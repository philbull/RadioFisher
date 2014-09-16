#!/usr/bin/python
"""
Plot parameter 1D marginal constraints as a fn. of FG subtraction efficiency.
"""
import numpy as np
import pylab as P

epsilon = np.logspace(-3., -8., 12)

# (Abao, Omega_K, Omega_DE, w_0, w_a)
sigma_A, sigma_omegaK, sigma_omegaDE, sigma_w0, sigma_wa = np.load("eos-fg-scan.npy").T


# Plot
P.subplot(111)
P.plot(epsilon, sigma_A, lw=1.5, label="$A_\mathrm{BAO}$")
P.plot(epsilon, sigma_omegaK, lw=1.5, label="$\Omega_K$")
P.plot(epsilon, sigma_omegaDE, lw=1.5, label="$\Omega_{DE}$")
P.plot(epsilon, sigma_w0, lw=1.5, label="$w_0$")
P.plot(epsilon, sigma_wa, lw=1.5, label="$w_a$")

P.xscale('log')
P.yscale('log')

P.xlim((1e-8, 1e-3))
P.ylim((5e-3, 2e2))

# Display options
P.legend(loc='upper left', prop={'size':'x-large'})
P.ylabel("$1\sigma$ marginal", fontdict={'fontsize':'22'})
P.xlabel("$\epsilon_{FG}$", fontdict={'fontsize':'20'})

fontsize = 18.
for tick in P.gca().yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in P.gca().xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
P.tight_layout()
P.show()
