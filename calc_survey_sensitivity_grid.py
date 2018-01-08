#!/usr/bin/python
"""
Calculate dn/dz for SKA HI galaxy redshift surveys, using dn/dz curves for 
given flux thresholds (from Mario Santos) and flux scalings with redshift for 
specific arrays.

Use these to find the flux rms / survey area / time needed to get a certain 
sensitivity at a given redshift, and plot in a grid.
"""
import numpy as np
import pylab as P
import scipy.interpolate
import scipy.ndimage
import scipy.optimize
import radiofisher as rf
import radiofisher.experiments as experiments
from radiofisher.units import *
import sys

try:
    EXPT_ID = int(sys.argv[1])
except:
    print "Expects one argument: expt_id(int)"
    sys.exit(1)

DEBUG_PLOT = True # Whether to plot fitting functions or not

CALCULATE = False # Whether to update the cache or not
if CALCULATE:
    print "\nCalculating sensitivity grids...\n"
else:
    print "\nLoading sensitivity grids from file...\n"

NU_LINE = 1.420 # HI emission line freq. in GHz
FULLSKY = (4.*np.pi * (180./np.pi)**2.) # deg^2 in the full sky
NGAL_MIN = 1e3 # Min. no. of galaxies to tolerate in a redshift bin
CBM = 1. #np.sqrt(1.57) # Correction factor due to effective beam for MID/MK (OBSOLETE)
CTH = 0.5 # Correction factor due to taking 5 sigma (not 10 sigma) cuts for SKA1
SBIG = 500. # Flux rms to extrapolate dn/dz out to (constrains behaviour at large Srms)

#-------------------------------------------------------------------------------
# Survey specifications
#-------------------------------------------------------------------------------

#name = "SKA1-MID B2 x"
#numin = 700. #350. #785.
#numax = 1420.
#Sarea = 5e3
#Tinst = 20.
#Ddish = 15.
#Ndish = 200.
#ttot = 1e3
#dnu = 0.01
#effic = 0.8
#Nsig = 5.

#-------------------------------------------------------------------------------
# Combined survey specifications
#-------------------------------------------------------------------------------

# SKA1-MID B2 Rebaselined
if EXPT_ID == 0:
    name = "SKA1-MID B2 Rebase. + MK"
    numin = 900.
    numax = 1420.
    Sarea = 5e3
    ttot = 1e4
    dnu = 0.01
    Nsig = 5.

    # Tinst1, Ddish1, Ndish1, effic1, numin1, numax1
    expt1 = (15.5, 15., 130., 0.85, 950., 1420.) # MID B2 Rebase.
    expt2 = (30., 13.5, 64., 0.85, 900., 1420.) # MeerKAT

if EXPT_ID == 1:
    # SKA1-MID B2 Alternative
    name = "SKA1-MID B2 Alt. + MK"
    numin = 795.
    numax = 1420.
    Sarea = 5e3
    ttot = 1e4
    dnu = 0.01
    Nsig = 5.

    # Tinst1, Ddish1, Ndish1, effic1, numin1, numax1
    expt1 = (15.5, 15., 130., 0.85, 795., 1420.) # MID B2 Alt.
    expt2 = (30., 13.5, 64., 0.85, 900., 1420.) # MeerKAT

#-------------------------------------------------------------------------------

# Fitting coeffs. from Table 3 in v1 of Yahya et al. paper
Srms = np.array([0., 1., 3., 5., 6., 7.3, 10., 23., 40., 70., 100., 150., 200.])
c1 = [6.21, 6.55, 6.53, 6.55, 6.58, 6.55, 6.44, 6.02, 5.74, 5.62, 5.63, 5.48, 5.0]
c2 = [1.72, 2.02, 1.93, 1.93, 1.95, 1.92, 1.83, 1.43, 1.22, 1.11, 1.41, 1.33, 1.04]
c3 = [0.79, 3.81, 5.22, 6.22, 6.69, 7.08, 7.59, 9.03, 10.58, 13.03, 15.49, 16.62, 17.52]
c4 = [0.5874, 0.4968, 0.5302, 0.5504, 0.5466, 0.5623, 0.5928, 0.6069, 0.628, 
      0.6094, 0.6052, 0.6365, 1., 1.] # Needs end padding 1
c5 = [0.3577, 0.7206, 0.7809, 0.8015, 0.8294, 0.8233, 0.8072, 0.8521, 0.8442, 
      0.9293, 1.0859, 0.965, 0., 0.] # Needs end padding 0
c1 = np.array(c1); c2 = np.array(c2); c3 = np.array(c3)
c4 = np.array(c4); c5 = np.array(c5)
Smax = np.max(Srms)

# Calculate cosmo. functions
cosmo_fns = rf.background_evolution_splines(experiments.cosmo)
H, r, D, f = cosmo_fns


def fluxrms(nu, Tinst, Ddish, Ndish, Sarea, ttot, dnu=0.01, effic=0.7):
    """
    Calculate the flux rms [uJy] for a given array config., using expression 
    from Yahya et al. (2015), near Eq. 3.
    """
    Tsys = Tinst + 60. * (300./nu)**2.55 # [K]
    Aeff = effic * Ndish * np.pi * (Ddish/2.)**2. # [m^2]
    fov = (np.pi/8.) * (1.3 * 3e8 / (nu*1e6 * Ddish))**2. * (180./np.pi)**2. # [deg^2]
    tp = ttot * (fov / Sarea)
    Srms = 260. * (Tsys/20.) * (25e3 / Aeff) * np.sqrt( (0.01/dnu) * (1./tp) )
    return Srms

def fluxrms_combined(nu, expt1, expt2, Sarea, ttot, dnu=0.01):
    """
    Calculate the flux rms [uJy] for a combination of arrays, using expression 
    from Yahya et al. (2015), near Eq. 3.
    """
    Tinst1, Ddish1, Ndish1, effic1, numin1, numax1 = expt1
    Tinst2, Ddish2, Ndish2, effic2, numin2, numax2 = expt2
    
    # Calculate Aeff / Tsys for each sub-array
    Tsys1 = Tinst1 + 60. * (300./nu)**2.55 # [K]
    Tsys2 = Tinst2 + 60. * (300./nu)**2.55 # [K]
    Aeff1 = effic1 * Ndish1 * np.pi * (Ddish1/2.)**2. # [m^2]
    Aeff2 = effic2 * Ndish2 * np.pi * (Ddish2/2.)**2. # [m^2]
    
    # Define band masks
    msk1 = np.zeros(nu.shape); msk2 = np.zeros(nu.shape)
    msk1[np.where(np.logical_and(nu >= numin1, nu <= numax1))] = 1.
    msk2[np.where(np.logical_and(nu >= numin2, nu <= numax2))] = 1.
    
    # Calculate combined Aeff / Tsys
    Aeff_over_Tsys = Aeff1/Tsys1*msk1 + Aeff2/Tsys2*msk2
    
    # Calculate mean FOV
    fov1 = (np.pi/8.) * (1.3 * 3e8 / (nu*1e6 * Ddish1))**2.
    fov2 = (np.pi/8.) * (1.3 * 3e8 / (nu*1e6 * Ddish1))**2.
    fov = (Ndish1 * fov1 + Ndish2 * fov2) / float(Ndish1 + Ndish2)
    fov *= (180./np.pi)**2. # [deg^2]
    
    # Calculate time per pointing and overall sensitivity
    tp = ttot * (fov / Sarea)
    Srms = 260. * (25e3/20.) / Aeff_over_Tsys * np.sqrt( (0.01/dnu) * (1./tp) )
    return Srms

def extend_with_linear_interp(xnew, x, y):
    """
    Extend an array using a linear interpolation from the last two points.
    """
    dx = x[-1] - x[-2]
    dy = y[-1] - y[-2]
    ynew = y[-1] + dy * (xnew - x[-1]) / dx
    y = np.concatenate((y, [ynew,]))
    return y

def n_bin(zmin, zmax, dndz, bias=None):
    """
    Number density of galaxies in a given z bin (assumes full sky). Also 
    returns volume of bin. dndz argument expects an interpolation fn. in units 
    of deg^-2.
    """
    _z = np.linspace(zmin, zmax, 500)
    vol = 4.*np.pi*C * scipy.integrate.simps(r(_z)**2. / H(_z), _z)
    N_bin = FULLSKY * scipy.integrate.simps(dndz(_z), _z)
    nz = N_bin / vol
    
    # Calculate mean bias (weighted by number density)
    if bias is not None:
        b = scipy.integrate.simps(bias(_z)*dndz(_z), _z) / (N_bin / FULLSKY)
        #b = bias(0.5*(zmin+zmax))
        return nz, vol, b
    return nz, vol

def redshift_bins(dz=0.1, Nbins=None):
    """
    Calculate redshift bins.
    """
    zmin = NU_LINE*1e3 / numax - 1.
    zmax = NU_LINE*1e3 / numin - 1.
    if zmin < 0.: zmin = 0.
    if Nbins is not None:
        zbins = np.linspace(zmin, zmax, Nbins+1)
    else:
        Nbins = np.floor((zmax - zmin) / dz)
        zbins = np.linspace(zmin, zmin + dz*Nbins, Nbins+1)
        if zmax - np.max(zbins) > 0.04:
            zbins = np.concatenate((zbins, [zmax,]))
    return zbins


# Extrapolate fitting functions to high flux rms
c1 = extend_with_linear_interp(SBIG, Srms, c1)
c2 = np.concatenate((c2, [1.,])) # Asymptote to linear fn. of redshift
c3 = extend_with_linear_interp(SBIG, Srms, c3)
Srms = np.concatenate((Srms, [SBIG,]))

# Construct grid of dn/dz (deg^-2) as a function of flux rms and redshift and 
# then construct 2D interpolator
z = np.linspace(0., 4., 1400)
nu = NU_LINE / (1. + z)
_dndz = np.array([10.**c1[j] * z**c2[j] * np.exp(-c3[j]*z) for j in range(Srms.size)])
_bias = np.array([c4[j] * np.exp(c5[j]*z) for j in range(Srms.size)])
dndz = scipy.interpolate.RectBivariateSpline(Srms, z, _dndz, kx=1, ky=1)
bias = scipy.interpolate.RectBivariateSpline(Srms, z, _bias, kx=1, ky=1)


def nz_for_specs(Sarea, ttot):
    """
    Calculate n(z) for a given set of specs.
    """
    # Construct dndz(z) interpolation fn. for the sensitivity of actual experiment
    fsky = Sarea / FULLSKY
    nu = 1420. / (1. + z)

    # Calculate flux
    #Sz = (Nsig/10.) * fluxrms(nu, Tinst, Ddish, Ndish, Sarea, ttot, dnu, effic) # 5 sigma
    Sz = (Nsig/10.) * fluxrms_combined(nu, expt1, expt2, Sarea, ttot, dnu)
    #Sref = fluxrms(1000., Tinst, Ddish, Ndish, Sarea, ttot, dnu, effic)
    #print "Srms = %3.1f uJy [%d sigma]" % (Sref, Nsig)

    dndz_expt = scipy.interpolate.interp1d(z, dndz.ev(Sz, z))
    bias_expt = scipy.interpolate.interp1d(z, bias.ev(Sz, z))

    # Calculate binned number densities
    nz, vol, b = np.array( [n_bin(zbins[i], zbins[i+1], dndz_expt, bias_expt) 
                            for i in range(zbins.size-1)] ).T
    vol *= fsky
    return nz, b


# Define redshift bins
#zbins = redshift_bins(dz=0.1)
zbins = redshift_bins(dz=0.01)
zc = np.array([0.5*(zbins[i] + zbins[i+1]) for i in range(zbins.size-1)])

# Load P(k)
k, pk = np.genfromtxt("cache_pk.dat").T

# Get n(z) as a function of survey area
sareas = np.logspace(2., np.log10(30e3), 30)
ttots = np.logspace(np.log10(500.), np.log10(2e4), 30)
X, Y = np.meshgrid(ttots, sareas)

# Hi-res grid, for interpolation
hsareas = np.logspace(2., np.log10(30e3), 60)
httots = np.logspace(2., np.log10(2e4), 60)
hX, hY = np.meshgrid(httots, hsareas)


# Calculate grid
if CALCULATE:
    zmaxes = np.zeros((sareas.size, ttots.size))
    zmaxes2 = np.zeros((sareas.size, ttots.size))
    vols = np.zeros((sareas.size, ttots.size))
    for i in range(sareas.size):
        for j in range(ttots.size):
            # Get number density
            nzs, _b = nz_for_specs(sareas[i], ttots[j])
            
            # Find z_max at which n(z) > 5e-4
            #if np.where(nzs >= 5e-4)[0].size > 0:
            #    idx = np.argmin(np.abs(nzs - 5e-4))
            #    zmaxes[i, j] = zc[idx]
            #    
            #    # Get volume for z_max
            #    _z = np.linspace(0., zc[idx], 200)
            #    vols[i, j] = C * scipy.integrate.simps(r(_z)**2. / H(_z), _z)
            #    vols[i, j] *= sareas[i] * (np.pi/180.)**2.
            
            # Find 1/(b^2 P(k_NL)) and use it to calculate z_max
            knl = 0.14 * (1. + zc)**(2./(2. + experiments.cosmo['ns']))
            pk02 = scipy.interpolate.interp1d(k, pk, kind='linear')(knl)
            pkinv = 1./ ( pk02 * (D(zc) * _b)**2. )
            
            # Calculate zmax (only if a solution nP=1 exists)
            if np.where(nzs - pkinv >= 0.)[0].size > 0:
                zmaxes[i, j] = zc[np.argmin(np.abs(nzs - pkinv))]
                
                # Get volume for z_max
                _z = np.linspace(0., zmaxes[i, j], 200)
                vols[i, j] = C * scipy.integrate.simps(r(_z)**2. / H(_z), _z)
                vols[i, j] *= sareas[i] * (np.pi/180.)**2.

    # Save and then re-load
    np.save("survey_sens_grid", zmaxes)
    np.save("survey_sens_grid_vol", vols)

# Load data from cache
zmaxes = np.load("survey_sens_grid.npy")
vols = np.load("survey_sens_grid_vol.npy")

# Interpolate grid (for smoothing)
#interp_grid = scipy.interpolate.interp2d(np.log(X).flatten(), np.log(Y).flatten(), zmaxes.flatten(), kind='cubic')
#h_zmaxes = interp_grid(np.log(httots), np.log(hsareas))
#hX, hY = np.meshgrid(httots, hsareas)

"""
######################
# VOLUME
######################
vols2 = scipy.ndimage.gaussian_filter(vols/1e9, sigma=1.5, order=0)
#ctr = P.contour(X, Y, vols/1e9, colors='r', linewidths=2., levels=[0.1, 0.2, 0.4, 0.6, 0.8, 1.2, 1.6])
ctr = P.contour(X, Y, vols2, colors='k', linewidths=2., levels=[0.1, 0.2, 0.4, 0.6, 0.8, 1.2, 1.6,])
clbls = P.clabel(ctr, ctr.levels, fmt="%1.1f ", fontsize=18, inline=False, use_clabeltext=True, background='r')

txt = P.figtext(0.18, 0.90, r"${\rm V}_{\rm sur}(z_{\rm max})$", fontsize=24, backgroundcolor='w', bbox={'alpha':0.15, 'linewidth':0.})

clbls[0].set_position((1450., 185.))

FIG_NAME = "ska1gs-vol.pdf"

"""
######################
# Z_MAX
######################

#ctr = P.contour(X, Y, zmaxes, colors='k', linewidths=2., levels=[0.1, 0.205, 0.3, 0.4, 0.5, 0.6, 0.8, 1.])
zmaxes2 = scipy.ndimage.gaussian_filter(zmaxes, sigma=0.8, order=0)
ctr = P.contour(X, Y, zmaxes2, colors='k', linewidths=2., levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.])

#ctr2 = P.contour(X, Y, zmaxes2, colors='r', linewidths=2., levels=[0.2, 0.3, 0.4, 0.6, 0.8, 1.])
#ctr2 = P.contour(hX, hY, h_zmaxes, colors='k', linewidths=2.)

#print ctr.levels
clbls = P.clabel(ctr, ctr.levels, fmt="%1.1f ", fontsize=18, inline=False, use_clabeltext=True, background='r')

P.figtext(0.20, 0.89, r"$z_{\rm max}$", fontsize=28, backgroundcolor='w')

P.annotate('SKA1 GS', xy=(7.5e3, 6.9e3),  xycoords='data',
           horizontalalignment='center', verticalalignment='center', 
           fontsize=20., color='r')
FIG_NAME = "ska1gs-zmax.pdf"

#clbls[6].set_position((16500., 135.))
#clbls[6].set_bbox(dict(alpha=0.0,))

######################


# Manually set label positions and rotations
#clbls[0].set_position((440., 3130.))
#clbls[1].set_position((650., 1820.))
#clbls[2].set_position((1910., 845.))

for l in clbls:
    l.set_rotation(0.)
    l.set_backgroundcolor('w')

# Lines and markings
#P.axvline(1e3, color='k', ls='dashed', lw=1.5, alpha=0.5)
#P.axvline(1e4, color='k', ls='dashed', lw=1.5, alpha=0.5)
P.plot(1e4, 5e3, 'rx', ms=15., mew=3.) # SKA1 5,000 deg^2 survey


#P.grid(True)
P.xscale('log')
P.yscale('log')

P.tick_params(axis='both', which='major', labelsize=20, width=1.5, size=8., pad=10)
P.tick_params(axis='both', which='minor', labelsize=20, width=1.5, size=5.)

P.ylabel(r'$S_{\rm area}$ $[{\rm deg}^{2}]$', labelpad=10., fontdict={'fontsize':'xx-large'})
P.xlabel(r'$t_{\rm tot}$ $[{\rm hrs}]$', labelpad=5., fontdict={'fontsize':'xx-large'})

P.tight_layout()

P.savefig(FIG_NAME)
P.show()
