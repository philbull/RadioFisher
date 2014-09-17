#!/usr/bin/python
"""
Plot transverse beams of rf.experiments.
"""
import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
from units import *
from mpi4py import MPI

import os
import euclid

e = rf.experiments.
cosmo = rf.experiments.cosmo

#expts = [ 
#  e.exptS, e.exptM, e.exptL, e.exptL, e.exptL,
#  e.GBT, e.Parkes, e.GMRT, e.WSRT, e.APERTIF,
#  e.VLBA, e.JVLA, e.JVLA, e.BINGO, e.BAOBAB32,
#  e.BAOBAB128, e.CHIME, e.AERA3, e.KAT7, e.KAT7, 
#  e.KAT7, e.MeerKATb1, e.MeerKATb1, e.MeerKATb1, e.MeerKATb2, 
#  e.MeerKATb2, e.MeerKATb2, e.ASKAP, e.SKA1MIDbase1, e.SKA1MIDbase1, 
#  e.SKA1MIDbase1, e.SKA1MIDbase2, e.SKA1MIDbase2, e.SKA1MIDbase2, e.SKA1MIDfull1, 
#  e.SKA1MIDfull1, e.SKA1MIDfull1, e.SKA1MIDfull2, e.SKA1MIDfull2, e.SKA1MIDfull2,
#  e.SKA1SURbase1, e.SKA1SURbase2, e.SKA1SURfull1, e.SKA1SURfull2 ]

# FIXME: Which band to use for SKA rf.experiments. Changes T_inst.
expts = [ e.CHIME, e.MeerKATb1, e.SKA1MIDfull1, e.SKA1MIDfull1, ]
labels = ['CHIME Full', 'MeerKAT B1', 'SKA1-MID Full B1 (Int.)', 'SKA1-MID Full B1 (Dish)']
mode = ['c', 'i', 'i', 's']

#colours = ['#CC0000', '#ED5F21', '#FAE300', '#5B9C0A', '#1619A1', '#56129F', '#990A9C']
colours = ['#5B9C0A', '#990A9C', '#1619A1', '#CC0000', '#FAE300', 'c']
linestyle = ['solid', 'dashdot', 'dashed', 'solid', 'solid', 'dotted']
lws = [1.2, 1.8, 1.8, 2.1]

# FIXME
expts = [ e.SKA1MIDfull1, e.SKA1MIDfull1, ]
labels = ['SKA1-MID Full B1 (Int.)', 'SKA1-MID Full B1 (Dish)']
mode = ['i', 's']
colours = ['#1619A1', '#CC0000', '#FAE300', 'c']
linestyle = ['solid', 'solid', 'dashed', 'solid', 'solid', 'dotted']
lws = [2.1, 2.1]

# Set cosmo values at fixed redshift
cosmo = rf.experiments.cosmo
cosmo_fns = rf.background_evolution_splines(cosmo)
H, r, D, f = cosmo_fns

z = 1. #2.55
cosmo['z'] = z
cosmo['r'] = r(z)
cosmo['rnu'] = C*(1.+z)**2. / H(z)

# Set k_perp
kperp = np.logspace(-3., 2., 2000)
q = r(z) * kperp

# Set-up plots
fig = P.figure()
ax2 = fig.add_subplot(211)
ax = fig.add_subplot(212)

for k in range(len(labels)):
    expt = expts[k]
    
    # Noise prefactor
    Tsky = 60e3 * (300.*(1.+z)/expt['nu_line'])**2.55 # Foreground sky signal (mK)
    Tsys = expt['Tinst'] + Tsky
    noise = (Tsys/1e3)**2.
    
    if mode[k] == 'i' or mode[k] == 'c':
        if mode[k] == 'i': expt['mode'] = 'interferom'
        if mode[k] == 'c': expt['mode'] = 'cylinder'
        
        # FOV calculation
        nu = expt['nu_line'] / (1. + z)
        l = 3e8 / (nu*1e6)
        if 'cyl' in expt['mode']:
            expt['fov'] = np.pi * (l / expt['Ddish']) # Cylinder mode, 180deg * theta
        else:
            expt['fov'] = (l / expt['Ddish'])**2.
        
        # Load n(u) interpolation function, if needed
        if ( 'int' in expt['mode'] or 'cyl' in expt['mode'] or 
             'comb' in expt['mode']) and 'n(x)' in expt.keys():
            expt['n(x)_file'] = expt['n(x)']
            expt['n(x)'] = rf.load_interferom_file(expt['n(x)'])
        
        # Transverse beam
        noise *= rf.interferometer_response(q, y=np.zeros(q.shape), 
                                                   cosmo=cosmo, expt=expt)
    else:
        noise *= rf.dish_response(q, y=np.zeros(q.shape), 
                                                   cosmo=cosmo, expt=expt)
    
    ax.plot( kperp, noise, color=colours[k], lw=lws[k], label=labels[k], 
            ls=linestyle[k] )


# Foregrounds
#cfg = rf.Cfg(q, y=np.zeros(q.shape), cosmo=cosmo, expt=expt)
#P.plot(kperp, cfg*1e8, 'k-')


############################
# Get f_bao(k) function
cosmo = rf.load_power_spectrum(cosmo, "cache_pk.dat", force_load=True)
fbao = cosmo['fbao']

kk = np.logspace(-4., 2., 1000)
ax2.plot(kk, fbao(kk), 'k-', lw=1.8)
ax2.set_xscale('log')

# BAO scales
#ax2.axvline(0.023, color='k', lw=2.8, ls='dashed', alpha=0.4)
#ax2.axvline(0.39, color='k', lw=2.8, ls='dashed', alpha=0.4)

ax2.fill_betweenx([-1,1], 0.023, 0.39, color='k', alpha=0.05)
ax.fill_betweenx([1e-3,1e6], 0.023, 0.39, color='k', alpha=0.05, zorder=-1)

#ax.axvline(0.023, color='k', lw=2.8, ls='dashed', alpha=0.4)
#ax.axvline(0.39, color='k', lw=2.8, ls='dashed', alpha=0.4)

############################


ax.legend(loc='lower right', prop={'size':'medium'}, frameon=False)

# Scale
ax.set_xscale('log')
ax.set_yscale('log')

#ax.set_ylim((6e-3, 1e7))
ax.set_ylim((1e-1, 5e5))
ax2.set_ylim((-0.06, 0.06))

ax.set_xlim((1e-3, 3e1))
ax2.set_xlim((1e-3, 3e1))

"""
# y labels
P.ylabel(r"$T^2_\mathrm{sys}\, \mathcal{I} B^{-2}_\perp \, [K^2]$", fontdict={'fontsize':'20'})
P.xlabel(r"$k_\perp \,[\mathrm{Mpc}^{-1}]$", fontdict={'fontsize':'20'})
P.legend(loc='lower center', prop={'size':'medium'}, ncol=2)

P.tick_params(axis='both', which='major', labelsize=20, size=8., width=1.5, pad=8.)
P.tick_params(axis='both', which='minor', labelsize=20, size=3., width=1.5, pad=8.)
"""

ax.tick_params(axis='both', which='major', labelsize=20, size=8., width=1.5, pad=8.)
ax.tick_params(axis='both', which='minor', size=4., width=1.5)

ax.set_xlabel(r"$k_\perp \, [\mathrm{Mpc}^{-1}]$", fontdict={'fontsize':'xx-large'}, labelpad=8.)
ax.set_ylabel(r"$T^2_\mathrm{sys}\, \mathcal{I} B^{-2}_\perp \, [K^2]$", fontdict={'fontsize':'xx-large'}, labelpad=4.)


ax2.tick_params(axis='both', which='major', size=8., width=1.5, pad=8., labelbottom='off', labelleft='off')
ax2.tick_params(axis='both', which='minor', size=4., width=1.5)


# Set positions
ax.set_position([0.15, 0.17, 0.8, 0.58])
ax2.set_position([0.15, 0.75, 0.8, 0.2])

#P.tight_layout()
# Set size
#P.gcf().set_size_inches(8.5, 7.)
#P.savefig("pub-beams-%2.2f.pdf" % z, dpi=100)
#P.savefig("pub-beams.pdf", dpi=100)
P.savefig("pub-beams-ska.pdf", dpi=100)
P.show()
