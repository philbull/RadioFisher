#!/usr/bin/python
"""
Plot various HI evolution functions as a fn. of redshift. Uses data from 
Mario's tHIbar_z code.
"""

import numpy as np
import pylab as P
import baofisher
import matplotlib.patches
import matplotlib.cm
from units import *
import experiments
import os

root = "HI_evolution/"
fname = ['bias_linear.dat', 'bias_linear_vcut.dat', 'bias_dave.dat', 'bias_obr.dat']
names = ['Linear', 'Bagla et al. (2010)', 'Dave et al. (XXXX)', 'Obreschkow et al. (XXXX)']
colours = ['k', 'r', 'b', 'y']

# Plot all curves
ax1 = P.subplot(121)
ax2 = P.subplot(122)
#ax3 = P.subplot(223)
#ax4 = P.subplot(224)

for i in range(len(fname)):
    z, Omega_HI, Omega_HI_ab, bias, Tb = np.genfromtxt(root+fname[i], skip_header=7).T
    ax1.plot(z, Omega_HI * 1e4, color=colours[i], lw=1.8, label=names[i])
    ax2.plot(z, bias, color=colours[i], lw=1.8, label=names[i])
    #ax3.plot(z, bias, color=colours[i], lw=1.5)

#P.xlabel(r"$z$", fontdict={'fontsize':'20'})
#P.ylabel(r"$T_b(z)$", fontdict={'fontsize':'20'})

#ax1.set_ylabel(r"$T_b(z)$")
ax1.set_xlabel(r"$z$", fontsize=20)
ax1.set_ylabel(r"$\Omega_\mathrm{HI}(z) / 10^{-4}$", fontsize=20)
ax2.set_xlabel(r"$z$", fontsize=20)
ax2.set_ylabel(r"$b_\mathrm{HI}(z)$", fontsize=20)

ax1.set_ylim((0., 13.1))
ax2.legend(loc='upper left', prop={'size':'large'})

# Display options
fontsize = 18.
for ax in [ax1, ax2]:
    for tick in ax.yaxis.get_major_ticks():
      tick.label1.set_fontsize(fontsize)
    for tick in ax.xaxis.get_major_ticks():
      tick.label1.set_fontsize(fontsize)
P.tight_layout()
P.show()
