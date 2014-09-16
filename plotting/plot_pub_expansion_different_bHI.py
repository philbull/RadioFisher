#!/usr/bin/python
"""
Plot f(z), for different bHI.
"""
import numpy as np
import pylab as P
import experiments
import baofisher

# Load cosmology and experimental settings
cosmo = experiments.cosmo
expts = [experiments.exptA, experiments.exptB, experiments.exptC, experiments.exptD, experiments.exptE]

# Precompute cosmological functions and derivs.
camb_matterpower = "/home/phil/oslo/iswfunction/cosmomc/camb/testX_matterpower.dat"
cosmo_fns, cosmo = baofisher.precompute_for_fisher(cosmo, camb_matterpower)
H, r, D, f = cosmo_fns

# Load bHI values
bHI = np.load("tmp/expansion-bias-bHI.npy")

# Loop through bHI and plot sigma_f(z)
for j in range(bHI.size):
    # Load results
    sigmas = np.load("tmp/expansion-bias-%3.3f.npy" % bHI[j])
    zc = np.load("expansion-bias-zc-%d.npy"%j)
    
    # Figure out where the redshift bins are for f(z)
    # A, bHI, f(z), sigma_NL, aperp(z), apar(z); zfns = [5, 4, 2]
    sigma_f = sigmas[2:2+zc.size]
    ff = f(zc)
    
    # Plot errors
    P.plot(zc, sigma_f/ff, lw=1.5, label=str(bHI[j]), marker='.')


P.xlabel("$z$", fontdict={'fontsize':'20'})
P.ylabel("Fractional error", fontdict={'fontsize':'20'})

P.ylim((0., 0.064))
P.xlim((0.9*np.min(zc), 1.005*np.max(zc)))

# Legend
P.legend(loc='upper left', prop={'size':'x-large'}, ncol=2)

# Shaded regions in different redshift regimes
#P.axvspan(np.max(zclist[0]), 3., ec='none', fc='#f2f2f2')
#P.axvspan(0., np.min([np.min(_zc) for _zc in zclist]), ec='none', fc='#cdcdcd')
#P.axvspan(np.min([np.min(_zc) for _zc in zclist]), np.min(zc), ec='none', fc='#f2f2f2')

# Display options
fontsize = 18.
for tick in P.gca().yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in P.gca().xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
P.tight_layout()
P.show()
