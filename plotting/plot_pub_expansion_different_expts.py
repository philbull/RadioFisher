#!/usr/bin/python
"""
Plot f(z), for different surveys.
"""
import numpy as np
import pylab as P

from rfwrapper import rf

# Load cosmology and experimental settings
cosmo = rf.experiments.cosmo
expts = [rf.experiments.exptA, rf.experiments.exptB, rf.experiments.exptC, rf.experiments.exptD, rf.experiments.exptE]

from copy import copy
e = copy(rf.experiments.exptA)
#e['interferometer'] = True
#e['n(u)'] = rf.experiments.n_ska
#expts = [rf.experiments.exptA, e]

# Precompute cosmological functions and derivs.
camb_matterpower = "/home/phil/oslo/iswfunction/cosmomc/camb/testX_matterpower.dat"
cosmo_fns, cosmo = rf.precompute_for_fisher(cosmo, camb_matterpower)
H, r, D, f = cosmo_fns

name = ['SD', 'Interferom.', 'C', 'D', 'E']
#name = ['A', 'B', 'C', 'D', 'E']
cols = ['b', 'g', 'c', 'r', 'y']

# Loop through rf.experiments.
zclist = []
for j in range(len(expts)):
    sigmas = np.load("tmp/expansion-expts-%d.npy"%j)
    zc = np.load("expansion-expts-zc-%d.npy"%j)
    zclist.append(zc)
    
    #if j == 0: continue # Skip A, it's the same as C
    
    # Figure-out where the redshift bins are for f(z)
    # A, bHI, f(z), sigma_NL, aperp(z), apar(z); zfns = [5, 4, 2]
    sigma_f = sigmas[2:2+zc.size]
    ff = f(zc)
    
    # Plot errors
    P.plot(zc, sigma_f/ff, lw=1.5, color=cols[j], label=name[j], marker='.')


P.xlabel("$z$", fontdict={'fontsize':'20'})
P.ylabel("Fractional error", fontdict={'fontsize':'20'})

#P.ylim((0., 0.16))
P.xlim((0., 2.01))

# Legend
P.legend(loc='upper left', prop={'size':'x-large'}, ncol=2)

# Shaded regions in different redshift regimes
P.axvspan(np.max(zclist[0]), 3., ec='none', fc='#f2f2f2')
P.axvspan(0., np.min([np.min(_zc) for _zc in zclist]), ec='none', fc='#cdcdcd')
#P.axvspan(np.min([np.min(_zc) for _zc in zclist]), np.min(zclist[4]), ec='none', fc='#f2f2f2')

P.ylim((0., 0.16))

# Display options
fontsize = 18.
for tick in P.gca().yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in P.gca().xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
P.tight_layout()
P.show()
