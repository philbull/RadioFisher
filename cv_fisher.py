#!/usr/bin/python
"""
Calculate Fisher matrix and P(k) constraints for all redshift bins for a given 
experiment.
"""
import numpy as np
import pylab as P
import scipy.spatial, scipy.integrate, scipy.interpolate
from scipy.integrate import simps
import radiofisher as rf
from radiofisher.units import *
from radiofisher.experiments import USE, foregrounds
from mpi4py import MPI

comm = MPI.COMM_WORLD
myid = comm.Get_rank()
size = comm.Get_size()

################################################################################
# Set-up experiment parameters
################################################################################

# Define cosmology and experiment settings
survey_name = "ExptA"
root = "output/" + survey_name

# Planck 2015 base_plikHM_TTTEEE_lowTEB_post_BAO
cosmo = {
    'omega_M_0':        0.3108,
    'omega_lambda_0':   0.6892,
    'omega_b_0':        0.04883,
    'omega_HI_0':       4.86e-4,
    'N_eff':            3.046,
    'h':                0.6761,
    'ns':               0.96708,
    'sigma_8':          0.8344,
    'w0':               -1.,
    'wa':               0.,
    'mnu':              0.,
    'k_piv':            0.05,
    'aperp':            1.,
    'apar':             1.,
    'bHI0':             0.677,
    'sigma_nl':         1e-8, #7., # FIXME
    'mnu':              0.,
    'gamma':            0.55,
    'foregrounds':      foregrounds,
}


# Experimental setup A
expt = {
    'mode':             'idish',       # Interferometer or single dish
    'Ndish':            32**2,         # No. of dishes
    'Nbeam':            1,             # No. of beams
    'Ddish':            10.,           # Single dish diameter [m]
    'Tinst':            10.*(1e3),     # Receiver temp. [mK]
    'survey_dnutot':    1000.,         # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1420.,         # Max. freq. of survey
    'dnu':              0.2,           # Bandwidth of single channel [MHz]
    'Sarea':            2.*np.pi,      # Total survey area [radians^2]
    'epsilon_fg':       1e-14,         # Foreground amplitude
    'ttot':             43829.*HRS_MHZ, # Total integration time [MHz^-1]
    'nu_line':          1420.406,      # Rest-frame freq. of emission line [MHz]
    'epsilon_fg':       1e-12,         # FG subtraction residual amplitude
    'use':              USE            # Which constraints to use/ignore
}


def baseline_dist(Nx, Ny, Ddish, nu=1420.):
    """
    Creates interpolation function for (circularised) baseline density n(d), 
    assuming a regular grid.
    """
    # Generate regular grid
    y = np.arange(Ny, step=Ddish)
    x, y = np.meshgrid( np.arange(Nx) * Ddish, 
                        np.arange(Ny) * Ddish )
    
    # Calculate baseline separations
    d = scipy.spatial.distance.pdist(np.column_stack((x.flatten(), 
                                                      y.flatten())) ).flatten()
    
    # Calculate FOV and sensible uv-plane bin size
    Ndish = Nx * Ny
    l = 3e8 / (nu*1e6)
    fov = 180. * 1.22 * (l/Ddish) * (np.pi/180.)**2.
    du = 1. / np.sqrt(fov) # 1.5 / ...

    # Remove D < Ddish baselines
    d = d[np.where(d > Ddish)] # Cut sub-FOV baselines
    d /= l # Rescale into u = d / lambda
    
    # Calculate bin edges
    imax = int(np.max(d) / du) + 1
    edges = np.linspace(0., imax * du, imax+1)
    edges = np.arange(0., imax * du, 1.)# FIXME
    print edges[1] - edges[0]

    # Calculate histogram (no. baselines in each ring of width du)
    bins, edges = np.histogram(d, edges)
    u = np.array([ 0.5*(edges[i+1] + edges[i]) 
                   for i in range(edges.size-1) ]) # Centroids

    # Convert to a density, n(u)
    nn = bins / (2. * np.pi * u * du)

    # Integrate n(u) to find norm. (should give 1 if no baseline cuts used)
    norm = scipy.integrate.simps(2.*np.pi*nn*u, u)
    #print "n(u) renorm. factor:", 0.5 * Ndish * (Ndish - 1) / norm

    # Convert to freq.-independent expression, n(x) = n(u) * nu^2,
    # where nu is in MHz.
    n_x = nn * nu**2.
    x = u / nu
    
    
    # Plot n(u) as a fn. of k_perp
    kperp = 2.*np.pi*u / (0.5*(2733 + 1620.)) # @ avg. of z = 0.42, 0.78
    P.plot(kperp, n_x / 900.**2., lw=1.8, color='r')
    #P.xscale('log')
    
    P.ylabel("$n(u)$", fontsize=18)
    P.xlabel(r"$k_\perp$ ${\rm Mpc}^{-1}$", fontsize=18)
    
    P.gca().tick_params(axis='both', which='major', labelsize=20, size=8., 
                   width=1.5, pad=8.)
    P.gca().tick_params(axis='both', which='minor', labelsize=20, size=5., 
                   width=1.5, pad=8.)
    
    P.tight_layout()
    P.show()
    exit()
    
    return scipy.interpolate.interp1d(x, n_x, kind='linear', 
                                      bounds_error=False, fill_value=0.)

# Set baseline density
expt['n(x)'] = baseline_dist(32, 32, 10.) # Interferometer antenna density

# Define redshift bins
dat = np.genfromtxt("slosar_background_zlow.dat").T
zmin = dat[0]
bias = dat[4]
#zs = np.concatenate((zmin, [zmin[1] - zmin[0],]))
#zc = 0.5 * (zs[:-1] + zs[1:])

# Single bin between 800 - 1000 MHz
zs = np.array([1420./1000. - 1., 1420./800. - 1.])
zc = 0.5 * (zs[:-1] + zs[1:])

# Define kbins (used for output)
kbins = np.arange(0., 5.*cosmo['h'], 0.1*cosmo['h']) # Bins of 0.1 h/Mpc

################################################################################

# Precompute cosmological functions and P(k)
cosmo_fns = rf.background_evolution_splines(cosmo)
# Load P(k) and split into smooth P(k) and BAO wiggle function

k_in, pk_in = np.genfromtxt("slosar_pk_z0.dat").T # Already in non-h^-1 units
cosmo['pk_nobao'], cosmo['fbao'] = rf.spline_pk_nobao(k_in, pk_in)
cosmo['k_in_max'] = np.max(k_in)
cosmo['k_in_min'] = np.min(k_in)


# Switch-off massive neutrinos, fNL, MG etc.
mnu_fn = None
transfer_fn = None
Neff_fn = None
switches = []

H, r, D, f = cosmo_fns

################################################################################
# Compare Anze's functions with the ones we calculate internally
################################################################################

"""
# Distance, r(z) [Mpc]
zz = dat[0]
P.plot(zz, dat[1], 'b-', lw=1.8)
P.plot(zz, (1.+zz)*r(zz), 'y--', lw=1.8)

# Growth (normalised to 1 at z=0)
P.plot(zz, dat[2], 'r-', lw=1.8)
P.plot(zz, D(zz)/D(0.), 'y--', lw=1.8)

# Growth rate, f(z)
P.plot(zz, dat[3], 'g-', lw=1.8)
P.plot(zz, f(zz), 'y--', lw=1.8)

P.show()
exit()
"""

################################################################################
# Loop through redshift bins, assigning them to each process
################################################################################

for i in range(zs.size-1):
    if i % size != myid:
      continue
    print ">>> %2d working on redshift bin %2d -- z = %3.3f" % (myid, i, zc[i])
    
    # Calculate bandwidth
    numin = expt['nu_line'] / (1. + zs[i+1])
    numax = expt['nu_line'] / (1. + zs[i])
    expt['dnutot'] = numax - numin
    z = zc[i]
    
    # Pack values and functions into the dictionaries cosmo, expt
    HH, rr, DD, ff = cosmo_fns
    
    cosmo['A'] = 1.
    cosmo['omega_HI'] = rf.omega_HI(z, cosmo)
    cosmo['bHI'] = rf.bias_HI(z, cosmo) # FIXME
    cosmo['btot'] = cosmo['bHI']
    cosmo['Tb'] = rf.Tb(z, cosmo)
    cosmo['z'] = z; cosmo['D'] = DD(z)
    cosmo['f'] = ff(z)
    cosmo['r'] = rr(z); cosmo['rnu'] = C*(1.+z)**2. / HH(z)
    cosmo['switches'] = switches
    
    # Physical volume (in rad^2 Mpc^3) (note factor of nu_line in here)
    Vphys = expt['Sarea'] * (expt['dnutot']/expt['nu_line']) \
          * cosmo['r']**2. * cosmo['rnu']
    print "Vphys = %3.3e Mpc^3" % Vphys
    
    #---------------------------------------------------------------------------
    # Noise power spectrum
    #---------------------------------------------------------------------------
    
    # Get grid of (q,y) coordinates
    kgrid = np.linspace(1e-4, 5.*cosmo['h'], 500)
    KPAR, KPERP = np.meshgrid(kgrid, kgrid)
    y = cosmo['rnu'] * KPAR
    q = cosmo['r'] * KPERP
    
    # Get noise power spectrum (units ~ mK^2)
    cn = rf.Cnoise(q, y, cosmo, expt) * cosmo['r']**2. * cosmo['rnu'] \
                                      * cosmo['h']**3. \
                                      * 0.1**3. # FIXME: Fudge factor to get in 
                                                # the same ballpark!
    
    print "%3.3e Mpc^3" % (cosmo['r']**2. * cosmo['rnu'])
    
    # Plot noise power spectrum
    fig, ax = P.subplots(1)
    ax.set_aspect('equal')
    mat = ax.matshow(np.log10(cn).T, origin='lower',
                     extent=[0., np.max(kgrid)/cosmo['h'], 
                             0., np.max(kgrid)/cosmo['h']], 
                     aspect='auto', vmin=-3.7, vmax=-2.)
    
    # Lines of constant |k|
    from matplotlib.patches import Circle
    for n in range(1, 6):
        ax.add_patch( Circle((0., 0.), n, fc='none', ec='w', alpha=0.5, lw=2.2) )
    
    P.xlabel(r"$k_\perp$ $[h/{\rm Mpc}]$", fontsize=18)
    P.ylabel(r"$k_\parallel$ $[h/{\rm Mpc}]$", fontsize=18)
    
    # Colour bar
    clr = P.colorbar(mat)
    clr.set_label(r"$\log_{10}[P_N(k_\perp, k_\parallel)]$ $[{\rm mK}^2 {\rm Mpc}^3]$", fontsize=18)
    
    # Tweak tick labels
    P.gca().tick_params(axis='both', which='major', labelsize=20, size=8., 
                   width=1.5, pad=8.)
    P.gca().tick_params(axis='both', which='minor', labelsize=20, size=5., 
                   width=1.5, pad=8.)
    
    P.show()
    exit()
    #---------------------------------------------------------------------------
    
    
    
    
    
    
    # Set binning
    Nperp = 50
    Npar = 45
    dk = 0.1 * cosmo['h'] # k bin size
    
    # Loop over bins
    dP = np.zeros((Nperp, Npar))
    for ii in range(Nperp):
        kperp_min = 1e-4 + ii*dk
        kperp_max = kperp_min + dk
        kperp = np.logspace(np.log10(kperp_min), np.log10(kperp_max), 80)
        #kperp = np.linspace(kperp_min, kperp_max, 120)
        
        for jj in range(Npar):
            kpar_min = 1e-4 + jj*dk
            kpar_max = kpar_min + dk
            kpar = np.logspace(np.log10(kpar_min), np.log10(kpar_max), 40)
            #kpar = np.linspace(kpar_min, kpar_max, 80)
            
            # Get grid of (q,y) coordinates
            KPAR, KPERP = np.meshgrid(kpar, kperp)
            y = cosmo['rnu'] * KPAR
            q = cosmo['r'] * KPERP
            
            # Calculate integrand
            cs = rf.Csignal(q, y, cosmo, expt)
            cn = rf.Cnoise(q, y, cosmo, expt)
            integrand = KPERP * (cs / (cs + cn))**2.
    
            # Do double integration
            Ik = [simps(integrand.T[i], kperp) for i in range(kpar.size)]
            dP[ii,jj] = simps(Ik, kpar)
    
    # Rescale deltaP/P
    dP *= Vphys / (8. * np.pi**2.)
    dP = 1. / np.sqrt(dP)
    
    fig, ax = P.subplots(1)
    ax.set_aspect('equal')
    mat = ax.matshow(np.log10(dP).T, vmin=-3.7, vmax=-2., origin='lower',
                     extent=[0., Nperp*0.1, 0., Npar*0.1], aspect='auto')
    
    from matplotlib.patches import Circle
    for n in range(1, 6):
        ax.add_patch( Circle((0., 0.), n, fc='none', ec='w', alpha=0.5, lw=2.2) )
    
    P.xlabel(r"$k_\perp$ $[h/{\rm Mpc}]$", fontsize=18)
    P.ylabel(r"$k_\parallel$ $[h/{\rm Mpc}]$", fontsize=18)
    #P.yscale('log')
    #P.xscale('log')
    clr = P.colorbar(mat)
    clr.set_label("$\log_{10}[\sigma_P / P\,(k_\perp, k_\parallel)]$", fontsize=18)
    
    P.gca().tick_params(axis='both', which='major', labelsize=20, size=8., 
                   width=1.5, pad=8.)
    P.gca().tick_params(axis='both', which='minor', labelsize=20, size=5., 
                   width=1.5, pad=8.)
    
    #P.tight_layout()
    P.show()
    exit()
    
    # Evaluate at output grid points
    #Ikpar(kgrid)
    
    #cumtrapz(Ik, kgrid, initial=0.)

comm.barrier()
if myid == 0: print "Finished."
