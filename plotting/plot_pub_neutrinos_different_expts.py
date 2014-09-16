#!/usr/bin/python
"""
Plot mnu, for different surveys.
"""
import numpy as np
import pylab as P
import experiments
import baofisher

# Load cosmology and experimental settings
cosmo = experiments.cosmo

expts = [experiments.exptA, experiments.exptB, experiments.exptC, experiments.exptD, experiments.exptE]

# FIXME
from copy import copy
e = copy(experiments.exptA)
e['interferometer'] = True
e['n(u)'] = experiments.n_ska
expts = [experiments.exptA, e]

mnu_vals = [0.05, 0.1, 0.2]

"""
# Precompute cosmological functions and derivs.
camb_matterpower = "/home/phil/oslo/iswfunction/cosmomc/camb/testX_matterpower.dat"
cosmo_fns, cosmo = baofisher.precompute_for_fisher(cosmo, camb_matterpower)
H, r, D, f = cosmo_fns
"""

name = ['SD', 'Interferom.']
#name = ['A', 'B', 'C', 'D', 'E']
cols = ['b', 'g', 'c', 'r', 'y']

# Loop through experiments
zclist = []
for j in range(len(expts)):
    sigmas = np.load("tmp/neutrinos-expts-%d.npy"%j)
    zc = np.load("neutrinos-expts-zc-%d.npy"%j)
    zclist.append(zc)
    
    for i in range(3):
    
        # A, bHI, f(z), sigma_NL, aperp(z), apar(z), mnu; zfns = [5, 4, 2]
        sigma_mnu = sigmas[-1]
    
        x = 1.*(i+1) + j*0.1
        y = sigmas[-1] / mnu_vals[i]
        P.plot(x, y, marker='.')
    
    # Plot errors
    #P.plot(zc, sigma_f/ff, lw=1.5, color=cols[j], label=name[j], marker='.')
        if i == 1:
            print "Expt %s, Mnu=0.1, sigma(Mnu)/Mnu = %3.4f" % (name[j], sigmas[-1] / mnu_vals[i])
P.show()
exit()

P.xlabel("$z$", fontdict={'fontsize':'20'})
P.ylabel("Fractional error", fontdict={'fontsize':'20'})

P.ylim((0., 0.16))
P.xlim((0., 2.01))

# Legend
P.legend(loc='upper left', prop={'size':'x-large'}, ncol=2)

# Shaded regions in different redshift regimes
P.axvspan(np.max(zclist[0]), 3., ec='none', fc='#f2f2f2')
P.axvspan(0., np.min([np.min(_zc) for _zc in zclist]), ec='none', fc='#cdcdcd')
P.axvspan(np.min([np.min(_zc) for _zc in zclist]), np.min(zclist[4]), ec='none', fc='#f2f2f2')

# Display options
fontsize = 18.
for tick in P.gca().yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in P.gca().xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
P.tight_layout()
P.show()
