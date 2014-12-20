#!/usr/bin/python
"""
Compare the transverse beams for a set of experiments. (Fig. 30)
"""
import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
import os
from radiofisher import euclid
from radiofisher.units import *

e = rf.experiments
cosmo = rf.experiments.cosmo

# FIXME: Which band to use for SKA experiments. Changes T_inst.
expts = [ e.MeerKATb1, e.MeerKATb1, e.SKA1MID350, e.SKA1MID350, ]
labels = ['MeerKAT B1 (Int.)', 'MeerKAT B1 (Dish)', 'SKA1-MID B1 (Int.)', 'SKA1-MID B1 (Dish)']
mode = ['int', 'sd', 'int', 'sd']

#colours = ['#CC0000', '#ED5F21', '#FAE300', '#5B9C0A', '#1619A1', '#56129F', '#990A9C']
colours = ['#ED5F21', '#990A9C', '#1619A1', '#CC0000', '#FAE300', 'c']
linestyle = ['dashdot', 'dashed', 'solid', 'solid', 'dotted']
lws = [1.8, 1.8, 1.8, 2.1]

"""
# FIXME
expts = [ e.SKA1MIDfull1, e.SKA1MIDfull1, ]
labels = ['SKA1-MID Full B1 (Int.)', 'SKA1-MID Full B1 (Dish)']
mode = ['i', 's']
colours = ['#1619A1', '#CC0000', '#FAE300', 'c']
linestyle = ['solid', 'solid', 'dashed', 'solid', 'solid', 'dotted']
lws = [2.1, 2.1]
"""

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
y = np.zeros(q.shape)

# Set-up plots
fig = P.figure()
ax2 = fig.add_subplot(211)
ax = fig.add_subplot(212)

for k in range(len(labels)):
    expt = expts[k]
    
    # Noise prefactor
    Tsky = 60e3 * (300.*(1.+z)/expt['nu_line'])**2.55 # Foreground sky signal (mK)
    Tsys = expt['Tinst'] + Tsky
    effic = 0.7
    noise = (Tsys/1e3)**2.
    nu = expt['nu_line'] / (1. + z)
    l = 3e8 / (nu * 1e6) # Wavelength (m)
    
    if 'sd' not in mode[k]:
        # Interferometer mode
        print "\tInterferometer mode",
        
        # Default effective area / beam FWHM
        Aeff = effic * 0.25 * np.pi * expt['Ddish']**2. \
               if 'Aeff' not in expt.keys() else expt['Aeff']
        theta_b = l / expt['Ddish']
        
        # Load n(u) interpolation function, if needed
        if 'n(x)' in expt.keys():
            expt['n(x)_file'] = expt['n(x)']
            expt['n(x)'] = rf.load_interferom_file(expt['n(x)'])
        
        # Evaluate at critical freq.
        if 'nu_crit' in expt.keys():
            nu_crit = expt['nu_crit']
            l_crit = 3e8 / (nu_crit * 1e6)
            theta_b_crit = l_crit / expt['Ddish']
            
        # Choose specific mode
        if 'cyl' in mode[k]:
            # Cylinder interferometer
            print "(cylinder)"
            Aeff = effic * expt['Ncyl'] * expt['cyl_area'] / expt['Ndish']
            theta_b = np.sqrt( 0.5 * np.pi * l / expt['Ddish'] )
        elif 'paf' in mode[k]:
            # PAF interferometer
            print "(PAF)"
            theta_b = theta_b_crit * (nu_crit / nu) if nu > nu_crit else 1.
        elif 'aa' in mode[k]:
            # Aperture array interferometer
            print "(aperture array)"
            Aeff *= (expt['nu_crit'] / nu)**2. if nu > nu_crit else 1.
            theta_b = theta_b_crit * (nu_crit / nu)
        else:
            # Standard dish interferometer
            print "(dish)"
        
        noise *= rf.interferometer_response(q, y, cosmo, expt)
        noise *= l**4. / (expt['Nbeam'] * (Aeff * theta_b)**2.)
    else:
        Aeff = effic * 0.25 * np.pi * expt['Ddish']**2. \
               if 'Aeff' not in expt.keys() else expt['Aeff']
        theta_b = l / expt['Ddish']
        
        if 'paf' in mode[k]:
            # PAF autocorrelation mode
            print "(PAF)"
            noise *= l**4. / (Aeff**2. * theta_b**4.)
            noise *= 1. if nu > expt['nu_crit'] else (expt['nu_crit'] / nu)**2.
        else:
            # Standard dish autocorrelation mode
            print "(dish)"
            noise *= l**4. / (Aeff**2. * theta_b**4.)
        
        noise *= 1. / (expt['Ndish'] * expt['Nbeam'])
        noise *= rf.dish_response(q, y, cosmo, expt)
        
    """
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
    """
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
ax.set_ylim((5e0, 5e5))
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

# Set size
#P.savefig("pub-beams-%2.2f.pdf" % z, dpi=100)
P.savefig("fig30-beams.pdf", dpi=100)
P.show()
