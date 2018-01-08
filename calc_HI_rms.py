#!/usr/bin/python
"""
Calculate signal power as a function of frequency.
"""
import numpy as np
import pylab as P
import scipy.integrate
import baofisher
import experiments
from experiments import cosmo
from units import *
import copy

nu21 = 1420.                # Line frequency at z=0

# Pre-calculate cosmological quantities
k, pk = np.genfromtxt("cache_pk.dat")[:-1].T
H, r, D, f = baofisher.background_evolution_splines(cosmo)

def W_tophat(k, r):
    return 3. * ( np.sin(k * r) - k * r * np.cos(k * r) ) / ((k * r)**3.)

def calculate_rms(z, expt):
    """
    Calculate RMS of HI signal at a given redshift.
    """
    theta_b = 3e8 * (1. + z) / (1e6 * expt['nu_line']) / expt['Ddish'] # Beam FWHM
    rnu = C * (1.+z)**2. / H(z)
    Tb = baofisher.Tb(z, cosmo)
    bHI = 1. #baofisher.bias_HI(z, cosmo)

    # Calculate pixel volume at given redshift
    Vpix = (r(z) * theta_b)**2. * rnu * expt['dnu'] / nu21
    Rpix = Vpix**(1./3.)

    # Integrate P(k) to get correlation fn. averaged in a ball, xi(Rpix)
    y = k**2. * pk * W_tophat(k, Rpix)
    xi = scipy.integrate.simps(y, k) / (2. * np.pi**2.)
    
    # Return rms HI fluctuation
    return Tb * D(z) * bHI * np.sqrt(xi) # in mK

# Choose experiment
e = experiments
expts = [ e.SKA1MIDbase1, e.SKA1MIDbase2, e.SKA1MIDfull1, e.SKA1MIDfull2, 
          e.SKA1SURbase1, e.SKA1SURbase2, e.SKA1SURfull1, e.SKA1SURfull2 ]
names = [ 'SKA1MIDbase1', 'SKA1MIDbase2', 'SKA1MIDfull1', 'SKA1MIDfull2', 
          'SKA1SURbase1', 'SKA1SURbase2', 'SKA1SURfull1', 'SKA1SURfull2' ]

# Calculate sigma_HI for a range of redshift
#z = np.linspace(1e-2, 3., 100)
#Tb = baofisher.Tb(z, cosmo)
#sigma_HI = np.array([calculate_rms(zz, expt) for zz in z])

# Output noise per voxel (single-dish)
for j in range(len(expts)):
    expt = expts[j]
    zs, zc = baofisher.zbins_const_dnu(expt, cosmo, dnu=60.)
    dnu = expt['dnu']
    
    sigma_T = baofisher.noise_rms_per_voxel(zc, expt)
    expt2 = copy.copy(expt)
    expt2['dnu'] = 60. # 60 MHz
    sigma_60 = baofisher.noise_rms_per_voxel(zc, expt2)
    
    # Output data
    print ""
    print "-"*40
    print names[j]
    print "-"*40
    print " zc / dz / sigma_T [uK] / sigma_T [uK]"
    print " -- / -- / (%2.2f MHz)   / (60 MHz)" % dnu
    print "-"*40
    for i in range(zc.size):
        #sigma_HI = calculate_rms(zc[i], expt)
        print "%2.2f %4.4f %4.4f %4.4f" % (zc[i], zs[i+1]-zs[i], 1e3*sigma_T[i], 1e3*sigma_60[i])



expts = [ e.SKA1MIDbase1, e.SKA1MIDbase2, e.SKA1MIDfull1, e.SKA1MIDfull2 ]
names = [ 'SKA1MIDbase1', 'SKA1MIDbase2', 'SKA1MIDfull1', 'SKA1MIDfull2' ]


# Output noise per voxel (interferom.)
for j in range(len(expts)):
    expt = expts[j]
    zs, zc = baofisher.zbins_const_dnu(expt, cosmo, dnu=60.)
    dnu = expt['dnu']
    
    expt['Sarea'] = 100.*(D2RAD)**2.
    sigma_T = baofisher.noise_rms_per_voxel_interferom(zc, expt)
    expt['dnu'] = 60. # 60 MHz
    sigma_60 = baofisher.noise_rms_per_voxel_interferom(zc, expt)
    
    #n_x = load_interferom_file(expt['n(x)'])
    #x = u / nu  # x = u / (freq [MHz])
    #n_u = n_x(x) / nu**2. # n(x) = n(u) * nu^2
    
    # Output data
    print ""
    print "-"*40
    print names[j], "(INTERFEROMETER)"
    print "-"*40
    print " zc / dz / sqrt[n(u)] * sigma_T [uK] / sqrt[n(u)] * sigma_T [uK] / lambda [m] / Tsys [K]"
    print " -- / -- /       (%2.2f MHz)          /       (60 MHz)" % dnu
    print "-"*40
    for i in range(zc.size):
        # Calculate quantities from Eq. 9.38 of Rohlfs & Wilson (5th Ed.)
        l = 3e8 * (1. + zc[i]) / 1420.e6
        Ddish = expt['Ddish']
        Tsky = 60e3 * (300.*(1.+zc[i])/expt['nu_line'])**2.55 # Foreground sky signal (mK)
        Tsys = expt['Tinst'] + Tsky
        
        #sigma_HI = calculate_rms(zc[i], expt)
        print "%2.2f %4.4f %8.8f %8.8f %4.4f %4.4f" % \
              (zc[i], zs[i+1]-zs[i], 1e3*sigma_T[i], 1e3*sigma_60[i], 
               l, Tsys/1e3)


exit()
# Plot results
P.subplot(111)
P.plot(z, Tb*1e3, lw=1.4, label="$T_b(z)$")
P.plot(z, sigma_HI*1e3, lw=1.4, label="$\sigma_\mathrm{HI}(z)$")
P.plot(z, sigma_T*1e3, lw=1.4, label="$\sigma_T(z)$")
P.plot([0.5, 1., 1.5, 2.], [155.8, 210.9, 245.6, 260.8], 'bo') # mean Tb, from SKA RFC
P.plot([0.5, 1., 1.5, 2.], [40.1, 28.0, 20.9, 16.4], 'go') # rms Tb, from SKA RFC
P.xlabel("z")
P.ylabel("uK")
P.legend(loc='upper left')
P.show()
