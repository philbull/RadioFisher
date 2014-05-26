#!/usr/bin/python
"""
Plot dn/dz for SKA for several flux cuts (Mario Santos, email from 2014-05-07).
"""
import numpy as np
import pylab as P
import scipy.interpolate
import scipy.integrate
import baofisher
from units import *
import experiments as e
import scipy.integrate

dat = np.genfromtxt("SKA_HIdndzb.txt", skip_header=1).T
z = dat[0]

fluxes = [-1, 0., 1., 3., 5., 6., 7.3, 10., 23., 40., 70., 100., 150., 200.] # Flux RMS
labels = ['7.3 $\mu\mathrm{Jy}$ (SKA 2)', '100 $\mu\mathrm{Jy}$ (SKA 1)', '150 $\mu\mathrm{Jy}$']

colours = ['#CC0000', '#1619A1', '#5B9C0A', '#990A9C'] # DETF/F/M/S

# Survey binning and area
#zbins = np.linspace(0., 2., 21) # dz = 0.1
zbins = np.linspace(0., 2., 11) # dz = 0.2
Sarea = 30e3 # deg^2

################################################################################
# Calculate background redshift evolution
cosmo_fns = baofisher.background_evolution_splines(e.cosmo)
H, r, D, f = cosmo_fns

def V(zmin, zmax):
    _z = np.linspace(zmin, zmax, 1000)
    Vsurvey = C * scipy.integrate.simps(r(_z)**2. / H(_z), _z)
    Vsurvey *= 4. * np.pi
    #print "\tSurvey volume: %3.2f Gpc^3" % (Vsurvey/1e9)
    return Vsurvey

zc = [0.5*(zbins[k] + zbins[k+1]) for k in range(zbins.size - 1)]
zmin = np.array([zbins[i] for i in range(zbins.size - 1)])
zmax = np.array([zbins[i+1] for i in range(zbins.size - 1)])
dV = [V(zbins[k], zbins[k+1]) for k in range(zbins.size - 1)] # Volume

################################################################################

P.subplot(111)
idxs = [6, 11, 12] # Indexes to plot
for j in range(len(idxs)):
    i = idxs[j]
    
    # Interpolate number count, N_gal / deg^2 / dz
    dndz = scipy.interpolate.interp1d(z, dat[i], kind='linear', bounds_error=False, fill_value=0.)
    
    # Integrate in each z bin
    Nz = []
    for k in range(zbins.size - 1):
        zz = np.linspace(zbins[k], zbins[k+1], 1000)
        I = scipy.integrate.simps(dndz(zz), zz)
        Nz.append(I)
    
    P.plot(zc, Nz, lw=1.5, marker='.', color=colours[j], label=labels[j])
    
    print "[%3.1f] N_tot = %1.1e" % (fluxes[i], np.sum(Nz)*1e3)
    
    # Calculate number density [Mpc^-3]
    nn = np.array(Nz) * ( 4.*np.pi * (180./np.pi)**2. ) / np.array(dV)
    
    # Output n(z) [Mpc^-3]
    ii = np.where(nn > 0.) # Keep only bins with n > 0
    np.savetxt("ska_hi_nz_dz02_%d.dat" % fluxes[i], np.column_stack((zmin[ii], zmax[ii], nn[ii])))
    
    for k in range(len(Nz)):
        print "%2.2f %5.1f %3.2e" % (zc[k], Nz[k] * 1e3, nn[k]) # z, (10^3 deg^2)^-1, Mpc^-3
    print "-"*50

#P.xscale('log')
#P.yscale('log')

#P.ylim((2., 1e5))
P.xlim((0., 2.1))

P.gca().tick_params(axis='both', which='major', labelsize=14, width=1.5, size=8.)
P.gca().tick_params(axis='both', which='minor', labelsize=14, width=1.5, size=5.)
        
P.xlabel('$z$', labelpad=10., fontdict={'fontsize':'xx-large'})
P.ylabel('$N(z) \,[\mathrm{deg}^{-2}]$', labelpad=10., fontdict={'fontsize':'xx-large'})

P.legend(loc='upper right', prop={'size':'x-large'}, frameon=False)

P.tight_layout()
P.savefig("ska-hi-nz.pdf", transparent=True)
P.show()
