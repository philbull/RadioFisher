#!/usr/bin/python
"""
Plot parameter 1D marginal constraints as a fn. of FG subtraction efficiency.
"""
import numpy as np
import pylab as P

epsilon = np.logspace(-3., -8., 12)

# Load sigma[fn(z)] and z bins
sigma_ok = np.load("eos-errors-fnz-omegak.npy").T
sigma_w0 = np.load("eos-errors-fnz-w0.npy").T
sigma_wa = np.load("eos-errors-fnz-wa.npy").T
zc = np.load("eos-errors-fnz-zc.npy").T

# Plot
P.subplot(111)
P.plot(zc, sigma_ok, 'b-', lw=1.5, marker='.', label="$\Omega_K$")
P.plot(zc, sigma_w0, 'r-', lw=1.5, marker='.', label=r"$w_0$")
P.plot(zc, 0.1*sigma_wa, 'y-', lw=1.5, marker='.', label=r"$w_a \times \, 0.1$")

#P.xscale('log')
#P.yscale('log')

P.xlim((np.min(zc)*0.99, 1.401))
P.ylim((0., 0.29))

# Display options
P.legend(loc='upper right', prop={'size':'x-large'})
P.ylabel("$\sigma(z)$", fontdict={'fontsize':'22'})
P.xlabel("$z$", fontdict={'fontsize':'20'})

fontsize = 18.
for tick in P.gca().yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in P.gca().xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)

P.tight_layout()
P.show()
