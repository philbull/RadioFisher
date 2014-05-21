#!/usr/bin/python
"""
Perform HI survey Fisher forecast based on Pedro's formalism (see notes from 
August 2013).

Requires up-to-date NumPy, SciPy (tested with version 0.11.0) and matplotlib. 
A number of functions can optionally use MPI (mpi4py).

(Phil Bull & Pedro G. Ferreira, 2013--2014)
"""
import numpy as np
import scipy.integrate
import scipy.interpolate
from scipy.misc import derivative
import pylab as P
import matplotlib.patches
import matplotlib.cm
from units import *
import uuid, os, sys, copy, md5
import camb_wrapper as camb
from tempfile import gettempdir

# No. of samples in log space in each dimension. 300 seems stable for (AA).
NSAMP_K = 500 # 1000
NSAMP_U = 1500 # 3000

# Debug settings (set all to False for normal operation)
DBG_PLOT_CUMUL_INTEGRAND = False # Plot k-space integrand of the dP/P integral
INF_NOISE = 1e200 # Very large finite no. used to denote infinite noise
EXP_OVERFLOW_VAL = 250. # Max. value of exponent for np.exp() before assuming overflow

# Decide which RSD function to use (N.B. interpretation of sigma_NL changes 
# slightly depending on option)
RSD_FUNCTION = 'kaiser'
#RSD_FUNCTION = 'loeb'

# Location of CAMB fiducial P(k) file
# NOTE: Currently expects CAMB P(k) needs to be at chosen z value (z=0 here).
CAMB_MATTERPOWER = "/home/phil/oslo/iswfunction/cosmomc/camb/testX_matterpower.dat"
CAMB_KMAX = 130. / 0.7 # Max. k for CAMB, in h Mpc^-1
CAMB_EXEC = "/home/phil/oslo/bao21cm/camb" # Directory containing camb executable


################################################################################
# Plotting functions
################################################################################

def figure_of_merit(p1, p2, F, cov=None):
    """
    DETF Figure of Merit, defined as the area inside the 95% contour of w0,wa.
    
    fom = 1 / [ 4 * sqrt( |cov(w0, wa)| ) ], where cov = F^-1, and cov(w0, wa) 
    is the w0,wa 2x2 sub-matrix of the covmat. The factor of 4 comes from 
    looking at the 95% (2-sigma) contours.
    """
    if cov == None: cov = np.linalg.inv(F)
    
    # Calculate determinant
    c11 = cov[p1,p1]
    c22 = cov[p2,p2]
    c12 = cov[p1,p2]
    det = c11*c22 - c12**2.
    
    fom = 0.25 / np.sqrt(det)
    return fom

def ellipse_for_fisher_params(p1, p2, F, Finv=None):
    """
    Return covariance ellipse parameters (width, height, angle) from 
    Fisher matrix F, for parameters in the matrix with indices p1 and p2.
    
    See arXiv:0906.4123 for expressions.
    """
    if Finv is not None:
        cov = Finv
    else:
        cov = np.linalg.inv(F)
    c11 = cov[p1,p1]
    c22 = cov[p2,p2]
    c12 = cov[p1,p2]
    
    # Calculate ellipse parameters (Eqs. 2-4 of Coe, arXiv:0906.4123)
    y1 = 0.5*(c11 + c22)
    y2 = np.sqrt( 0.25*(c11 - c22)**2. + c12**2. )
    a = 2. * np.sqrt(y1 + y2) # Factor of 2 because def. is *total* width of ellipse
    b = 2. * np.sqrt(y1 - y2)
    
    # Flip major/minor axis depending on which parameter is dominant
    if c11 >= c22:
        w = a; h = b
    else:
        w = b; h = a
    
    # Handle c11==c22 case for angle calculation
    if c11 != c22:
        ang = 0.5*np.arctan( 2.*c12 / (c11 - c22) )
    else:
        ang = 0.5*np.arctan( 2.*c12 / 1e-20 ) # Sign sensitivity here
    
    # Factors to use if 1,2,3-sigma contours required
    alpha = [1.52, 2.48, 3.44]
    
    return w, h, ang * 180./np.pi, alpha

def plot_ellipse(F, p1, p2, fiducial, names, ax=None):
    """
    Show error ellipse for 2 parameters from Fisher matrix.
    """
    alpha = [1.52, 2.48, 3.44] # 1/2/3-sigma scalings, from Table 1 of arXiv:0906.4123
    
    # Get ellipse parameters
    x, y = fiducial
    a, b, ang = ellipse_for_fisher_params(p1, p2, F)
    
    # Get 1,2,3-sigma ellipses and plot
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[k]*b, 
                 height=alpha[k]*a, angle=ang, fc='none') for k in range(0, 2)]
    if ax is None: ax = P.subplot(111)
    for e in ellipses: ax.add_patch(e)
    ax.plot(x, y, 'bx')
    
    if ax is None:
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
        P.show()


def triangle_plot(fiducial, F, names, priors=None, skip=None):
    """
    Show triangle plot of 2D error ellipses from Fisher matrix.
    """
    alpha = [1.52, 2.48, 3.44] # 1/2/3-sigma scalings, from Table 1 of arXiv:0906.4123
    
    # Calculate covmat
    N = len(fiducial)
    Finv = np.linalg.inv(F)
    
    # Remove unwanted variables (after marginalisation though)
    if skip is not None:
        Finv = fisher_with_excluded_params(Finv, skip)
    
    # Loop through elements of matrix, plotting 2D contours or 1D marginals
    for i in range(N):
      for j in range(N):
        x = fiducial[i]; y = fiducial[j]
        
        # Plot 2D contours
        if j > i:
          a, b, ang = ellipse_for_fisher_params(i, j, F=None, Finv=Finv)
            
          # Get 1,2,3-sigma ellipses and plot
          ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[k]*b, 
                       height=alpha[k]*a, angle=ang, fc='none') for k in range(0, 2)]
          ax = P.subplot(N, N, N*j + i + 1)
          for e in ellipses: ax.add_patch(e)
          P.plot(x, y, 'bx')
          ax.tick_params(axis='both', which='major', labelsize=8)
          ax.tick_params(axis='both', which='minor', labelsize=8)
          
        # Plot 1D Gaussian
        if i==j:
          ax = P.subplot(N, N, N*i + j + 1)
          P.title(names[i])
          
          u = fiducial[i]
          std = np.sqrt(Finv[i,i])
          _x = np.linspace(u - 4.*std, u + 4.*std, 500)
          _y = np.exp(-0.5*(_x - u)**2. / std**2.) \
                     / np.sqrt(2.*np.pi*std**2.)
          P.plot(_x, _y, 'r-')
          
          # Add priors to plot
          if priors is not None:
            if not np.isinf(priors[i]):
              std = priors[i]
              yp = np.exp(-0.5*(_x - u)**2. / priors[i]**2.) \
                     / np.sqrt(2.*np.pi*std**2.)
              P.plot(_x, yp, 'b-', alpha=0.5)
          
          ax.tick_params(axis='both', which='major', labelsize=8)
          ax.tick_params(axis='both', which='minor', labelsize=8)
        if i==0: P.ylabel(names[j])
    P.show()

def fix_log_plot(y, yerr):
    """
    Corrects errorbars for log plot.
    """
    yup = yerr.copy()
    ylow = yerr.copy()
    ylow[np.where(y - yerr <= 0.)] = y[np.where(y - yerr <= 0.)]*0.99999
    
    # Correct inf's too
    ylow[np.isinf(yerr)] = y[np.isinf(yerr)]*0.99999
    yup[np.isinf(yerr)] = y[np.isinf(yup)]*1e5
    return yup, ylow

def plot_corrmat(F, names):
    """
    Plot the correlation matrix for a given Fisher matrix.
    
    Returns matplotlib Figure object.
    """
    # Construct correlation matrix
    F_corr = np.zeros(F.shape)
    for ii in range(F.shape[0]):
        for jj in range(F.shape[0]):
            F_corr[ii,jj] = F[ii, jj] / np.sqrt(F[ii,ii] * F[jj,jj])
    
    # Plot corrmat
    fig = P.figure()
    ax = fig.add_subplot(111)
    
    #F_corr = F_corr**3.
    matshow = ax.matshow(F_corr, vmin=-1., vmax=1., cmap=matplotlib.cm.get_cmap("RdBu"))
    #ax.title("z = %3.3f" % zc[i])
    fig.colorbar(matshow)
    
    """
    # Label the ticks correctly
    locs, labels = (ax.get_xticks(), ax.get_xticklabels())
    print labels, len(labels)
    print locs, len(locs)
    #labels = ['',]
    labels = ['',] + names
    new_labels = [x.format(locs[ii]) for ii,x in enumerate(labels)]
    ax.set_xticks(locs, new_labels)
    locs, labels = (ax.get_yticks(), ax.get_yticklabels())
    ax.set_yticks(locs, new_labels)
    
    ax.xlim((-0.5, 5.5))
    ax.ylim((5.5, -0.5))
    """
    lbls = ["%d:%s" % (i, names[i]) for i in range(len(names))]
    ax.set_xlabel(lbls)
    
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_markersize(0)
        tick.tick2line.set_markersize(0)
        tick.label1.set_horizontalalignment('center')
    
    fig.show()
    P.show()
    #P.savefig("corrcoeff-%s-%3.3f.png" % (names[k], zc[i]))


################################################################################
# Binning strategies
################################################################################

def zbins_equal_spaced(expt, bins=None, dz=None):
    """
    Return redshift bin edges and centroids for an equally-spaced redshift 
    binning.
    """
    if (bins is not None) and (dz is not None):
        raise ValueError("Specify either bins *or* dz; you can't specify both.")
    
    # Get redshift ranges
    zmin = expt['nu_line'] / expt['survey_numax'] - 1.
    zmax = expt['nu_line'] / (expt['survey_numax'] - expt['survey_dnutot']) - 1.
    
    # Set number of bins
    if dz is not None:
        bins = int((zmax - zmin) / dz)
    
    # Return bin edges and centroids
    zs = np.linspace(zmin, zmax, bins+1)
    zc = [0.5*(zs[i+1] + zs[i]) for i in range(zs.size - 1)]
    return zs, np.array(zc)

def zbins_const_dr(expt, cosmo, bins=None, nsamples=500, initial_dz=None):
    """
    Return redshift bin edges and centroids for bins that are equally-spaced 
    in r(z). Will either split the full range into some number of bins, or else 
    fill the range using bins of const. dr. set from an initial bin delta_z.
    """
    # Get redshift range
    zmin = expt['nu_line'] / expt['survey_numax'] - 1.
    zmax = expt['nu_line'] / (expt['survey_numax'] - expt['survey_dnutot']) - 1.
    
    # Fiducial cosmological values
    _z = np.linspace(0., zmax, nsamples)
    a = 1. / (1. + _z)
    H0 = (100.*cosmo['h']); w0 = cosmo['w0']; wa = cosmo['wa']
    om = cosmo['omega_M_0']; ol = cosmo['omega_lambda_0']
    ok = 1. - om - ol
    
    # Sample comoving dist. r(z) at discrete points (assumes ok~0 in last step)
    omegaDE = ol * np.exp(3.*wa*(a - 1.)) / a**(3.*(1. + w0 + wa))
    E = np.sqrt( om * a**(-3.) + ok * a**(-2.) + omegaDE )
    _r = (C/H0) * np.concatenate( ([0.], scipy.integrate.cumtrapz(1./E, _z)) )
    
    # Interpolate to get z(r) and r(z)
    r_z = scipy.interpolate.interp1d(_z, _r, kind='linear')
    z_r = scipy.interpolate.interp1d(_r, _z, kind='linear')
    
    # Return bin edges and centroids
    if bins is not None:
        # Get certain no. of bins
        rbins = np.linspace(r_z(zmin), r_z(zmax), bins+1)
        zbins = z_r(rbins)
    else:
        # Get bins with const. dr from initial delta_z
        rmin = r_z(zmin)
        rmax = r_z(zmax)
        dr = r_z(zmin + initial_dz) - rmin
        print "Const. dr =", dr, "Mpc"
        
        # Loop through range, filling with const. dr bins as far as possible
        rtot = rmin
        zbins = []
        while rtot < rmax:
            zbins.append(z_r(rtot))
            rtot += dr
        zbins.append(z_r(rmax))
        zbins = np.array(zbins)
        
    zc = [0.5*(zbins[i+1] + zbins[i]) for i in range(zbins.size - 1)]
    return zbins, np.array(zc)

def zbins_const_dnu(expt, cosmo, bins=None, dnu=None, initial_dz=None):
    """
    Return redshift bin edges and centroids for bins that are equally-spaced 
    in frequency (nu). Will either split the full range into some number of 
    bins, or else fill the range using bins of const. dnu set from an initial 
    bin delta_z.
    """
    # Get redshift range
    zmin = expt['nu_line'] / expt['survey_numax'] - 1.
    zmax = expt['nu_line'] / (expt['survey_numax'] - expt['survey_dnutot']) - 1.
    numax = expt['nu_line'] / (1. + zmin)
    numin = expt['nu_line'] / (1. + zmax)
    
    # nu as a function of z
    nu_z = lambda zz: expt['nu_line'] / (1. + zz)
    z_nu = lambda f: expt['nu_line'] / f - 1.
    
    # Return bin edges and centroids
    if bins is not None:
        # Get certain no. of bins
        nubins = np.linspace(numax, numin, bins+1)
        zbins = z_nu(nubins)
    else:
        # Get bins with const. dr from initial delta_z
        if dnu is None:
            dnu = nu_z(zmin + initial_dz) - nu_z(zmin)
        dnu = -1. * np.abs(dnu) # dnu negative, to go from highest to lowest freq.
        
        # Loop through range, filling with const. dr bins as far as possible
        nu = numax
        zbins = []
        while nu > numin:
            zbins.append(z_nu(nu))
            nu += dnu
        zbins.append(z_nu(numin))
        
        # Check we haven't exactly hit the edge (to ensure no zero-width bins)
        if np.abs(zbins[-1] - zbins[-2]) < 1e-4:
            del zbins[-2] # Remove interstitial bin edge
        zbins = np.array(zbins)
        
    zc = [0.5*(zbins[i+1] + zbins[i]) for i in range(zbins.size - 1)]
    return zbins, np.array(zc)

################################################################################
# Experiment specification handler functions
################################################################################

def load_interferom_file(fname):
    """
    Load n(u) file for interferometer and return linear interpolation function.
    """
    x, _nx = np.genfromtxt(fname).T
    interp_nx = scipy.interpolate.interp1d( x, _nx, kind='linear', 
                                            bounds_error=False, fill_value=1./INF_NOISE )
    return interp_nx

def overlapping_expts(expt_in, zlow=None, zhigh=None):
    """
    Get "effective" experimental parameters in a redshift bin for two 
    experiments that are joined together.
    
    If the bins completely overlap in frequency, takes the (dish*beam)-weighted 
    mean of {Tinst, Ddish}, the max. value of dnu, and the range in freq. 
    (survey_dnutot and survey_numax) covered by either experiment (not 
    necessarily both). Uses the survey parameters of expt1.
    
    Returns a dict of effective experimental parameters for the bin.
    """
    # Check to see if experiment is made up of overlapping instruments
    # Returns the original expt dict if no overlap is defined
    if 'overlap' in expt_in.keys():
        expt1, expt2 = expt_in['overlap']
    else:
        return expt_in
    
    # Copy everything over from expt_in
    expt = {}
    for key in expt_in.keys():
        if key is not 'overlap': expt[key] = expt_in[key]
    
    # If no low/high is specified, just return extended freq. range (useful for 
    # redshift binning)
    nu1 = [expt1['survey_numax'] - expt1['survey_dnutot'], expt1['survey_numax']]
    nu2 = [expt2['survey_numax'] - expt2['survey_dnutot'], expt2['survey_numax']]
    if zlow is None or zhigh is None:
        expt['survey_numax'] = np.max((expt1['survey_numax'], expt2['survey_numax']))
        expt['survey_dnutot'] = expt['survey_numax'] - np.min((nu1, nu2))
        expt['nu_line'] = expt1['nu_line']
        return expt
        
    # Calculate bin min/max freqs.
    nu_high = expt1['nu_line'] / (1. + zlow)
    nu_low  = expt1['nu_line'] / (1. + zhigh)
    
    # Get weighting factors (zero if expt. doesn't completely overlap with bin)
    N1 = expt1['Ndish'] * expt1['Nbeam']
    N2 = expt2['Ndish'] * expt2['Nbeam']
    if nu_low < 0.9999*nu1[0] or nu_high > 1.0001*nu1[1]: N1 = 0
    if nu_low < 0.9999*nu2[0] or nu_high > 1.0001*nu2[1]: N2 = 0
    assert(N1 + N2 != 0)
    
    f1 = N1 / float(N1 + N2)
    f2 = 1. - f1
    
    # Calculate effective parameters
    expt['Ndish'] = N1 + N2
    expt['Nbeam'] = 1
    expt['Tinst'] = f1 * expt1['Tinst'] + f2 * expt2['Tinst']
    expt['Ddish'] = f1 * expt1['Ddish'] + f2 * expt2['Ddish']
    
    # Full frequency range covered by either experiment (not used for noise calc.)
    expt['survey_numax'] = np.max((expt1['survey_numax'], expt2['survey_numax']))
    expt['survey_dnutot'] = expt['survey_numax'] - np.min((nu1, nu2))
    expt['dnu'] = np.max((expt1['dnu'], expt2['dnu']))
    
    # Establish common parameters
    expt['ttot'] = expt1['ttot']
    expt['Sarea'] = expt1['Sarea']
    expt['nu_line'] = expt1['nu_line']
    expt['epsilon_fg'] = expt1['epsilon_fg']
    expt['use'] = expt1['use']
    
    # Flag any keys that we didn't transfer over
    for key in expt1.keys():
        if key not in expt:
            print "\toverlapping_expts: Key '%s' from expt1 ignored." % key
    for key in expt2.keys():
        if key not in expt:
            print "\toverlapping_expts: Key '%s' from expt2 ignored." % key
    return expt


################################################################################
# Cosmology functions
################################################################################

def load_power_spectrum( cosmo, cachefile, kmax=CAMB_KMAX, comm=None, 
                         force=False, force_load=False ):
    """
    Precompute a number of quantities for Fisher analysis:
      - f_bao(k), P_smooth(k) interpolation functions (spline_pk_nobao)
    
    Parameters
    ----------
    
    cosmo : dict
        Dictionary of cosmological parameters
    
    cachefile : string
        Path to a cached CAMB matter powerspectrum output file.
        
        N.B. Ensure that k_max of the CAMB output is bigger than k_max for the 
        Fisher analysis; otherwise, P(k) will be truncated.
    
    Returns
    -------
    
    cosmo : dict
        Input cosmo dict, but with pk_nobao(k), fbao(k) added.
    """
    # MPI set-up
    myid = 0; size = 1
    if comm is not None:
        myid = comm.Get_rank()
        size = comm.Get_size()
    
    # Set-up CAMB parameters
    p = convert_to_camb(cosmo)
    p['transfer_kmax'] = kmax / cosmo['h']
    p['transfer_high_precision'] = 'T'
    p['transfer_k_per_logint'] = 250 #1000
    
    # Check for massive neutrinos
    if cosmo['mnu'] > 0.001:
        p['omnuh2'] = cosmo['mnu'] / 93.04
        p['massless_neutrinos'] = 2.046
        p['massive_neutrinos'] = "2 1"
        p['nu_mass_eigenstates'] = 1
    
    # Only let one MPI worker do the calculation, then let all other workers 
    # load the result from cache
    if myid == 0:
        print "\tprecompute_for_fisher(): Loading matter P(k)..."
        dat = cached_camb_output(p, cachefile, mode='matterpower', force=force,
                                 force_load=force_load)
    if comm is not None: comm.barrier()
    if myid != 0:
        dat = cached_camb_output(p, cachefile, mode='matterpower', force=force, 
                                 force_load=force_load)
    
    # Load P(k) and split into smooth P(k) and BAO wiggle function
    k_in, pk_in = dat
    cosmo['pk_nobao'], cosmo['fbao'] = spline_pk_nobao(k_in, pk_in)
    cosmo['k_in_max'] = np.max(k_in)
    cosmo['k_in_min'] = np.min(k_in)
    return cosmo

def background_evolution_splines(cosmo, zmax=10., nsamples=500):
    """
    Get interpolation functions for background functions of redshift:
      * H(z), Hubble rate in km/s/Mpc
      * r(z), comoving distance in Mpc
      * D(z), linear growth factor
      * f(z), linear growth rate
    """
    _z = np.linspace(0., zmax, nsamples)
    a = 1. / (1. + _z)
    H0 = (100.*cosmo['h']); w0 = cosmo['w0']; wa = cosmo['wa']
    om = cosmo['omega_M_0']; ol = cosmo['omega_lambda_0']
    ok = 1. - om - ol
    
    # Sample Hubble rate H(z) and comoving dist. r(z) at discrete points
    omegaDE = ol * np.exp(3.*wa*(a - 1.)) / a**(3.*(1. + w0 + wa))
    E = np.sqrt( om * a**(-3.) + ok * a**(-2.) + omegaDE )
    _H = H0 * E
    
    r_c = np.concatenate( ([0.], scipy.integrate.cumtrapz(1./E, _z)) )
    if ok > 0.:
        _r = C/(H0*np.sqrt(ok)) * np.sinh(r_c * np.sqrt(ok))
    elif ok < 0.:
        _r = C/(H0*np.sqrt(-ok)) * np.sin(r_c * np.sqrt(-ok))
    else:
        _r = (C/H0) * r_c
    
    # Integrate linear growth rate to find linear growth factor, D(z)
    # N.B. D(z=0) = 1.
    a = 1. / (1. + _z)
    Oma = cosmo['omega_M_0'] * (1.+_z)**3. * (100.*cosmo['h']/_H)**2.
    _f = Oma**cosmo['gamma']
    _D = np.concatenate( ([0.,], scipy.integrate.cumtrapz(_f, np.log(a))) )
    _D = np.exp(_D)
    
    # Construct interpolating functions and return
    r = scipy.interpolate.interp1d(_z, _r, kind='linear', bounds_error=False)
    H = scipy.interpolate.interp1d(_z, _H, kind='linear', bounds_error=False)
    D = scipy.interpolate.interp1d(_z, _D, kind='linear', bounds_error=False)
    f = scipy.interpolate.interp1d(_z, _f, kind='linear', bounds_error=False)
    return H, r, D, f

def Tb(z, cosmo, formula='chang'):
    """
    Brightness temperature Tb(z), in mK. Several different expressions for the 
    21cm line brightness temperature are available:
    
     * 'santos': obtained using a simple power-law fit to Mario's data.
       (Best-fit Tb: 0.1376)
     * 'hall': from Hall, Bonvin, and Challinor.
     * 'chang': from Chang et al. (Default)
    """
    omegaHI = omega_HI(z, cosmo)
    if formula == 'santos':
        Tb = 0.1376 * (1. + 1.44*z - 0.277*z**2.)
    elif formula == 'chang':
        Tb = 0.3 * (omegaHI/1e-3) * np.sqrt(0.29*(1.+z)**3.)
        Tb *= np.sqrt((1.+z)/2.5)
        Tb /= np.sqrt(cosmo['omega_M_0']*(1.+z)**3. + cosmo['omega_lambda_0'])
    else:
        # Hall et al.
        om = cosmo['omega_M_0']; ol = cosmo['omega_lambda_0']
        ok = 1. - om - ol
        E = np.sqrt(om*(1.+z)**3. + ok*(1.+z)**2. + ol)
        Tb = 188. * cosmo['h'] * omegaHI * (1.+z)**2. / E
    return Tb

def bias_HI(z, cosmo):
    """
    b_HI(z), obtained using a simple power-law fit to Mario's data.
    (Best-fit b_HI: 0.702)
    """
    return cosmo['bHI0'] * (1. + 3.80e-1*z + 6.7e-2*z**2.)

def omega_HI(z, cosmo):
    """
    Omega_HI(z), obtained using a simple power-law fit to Mario's data.
    (Best-fit omega_HI_0: 9.395e-04)
    """
    return cosmo['omega_HI_0'] * (1. + 4.77e-2*z - 3.72e-2*z**2.)

def omegaM_z(z, cosmo):
    """
    Matter density as a function of redshift.
    """
    om = cosmo['omega_M_0']; ol = cosmo['omega_lambda_0']
    ok = 1. - om - ol
    E = np.sqrt(om*(1.+z)**3. + ok*(1.+z)**2. + ol)
    return om * (1. + z)**3. * E**-2.

def convert_to_camb(cosmo):
    """
    Convert cosmological parameters to CAMB parameters.
    (N.B. CAMB derives Omega_Lambda from other density parameters)
    """
    p = {}
    p['hubble'] = 100.*cosmo['h']
    p['omch2'] = (cosmo['omega_M_0'] - cosmo['omega_b_0']) * cosmo['h']**2.
    p['ombh2'] = cosmo['omega_b_0'] * cosmo['h']**2.
    p['omk'] = 1. - (cosmo['omega_M_0'] + cosmo['omega_lambda_0'])
    p['scalar_spectral_index__1___'] = cosmo['ns']
    p['w'] = cosmo['w0']
    p['wa'] = cosmo['wa']
    return p

def cached_camb_output(p, cachefile, cosmo=None, mode='matterpower', 
                       force=False, force_load=False):
    """
    Load P(k) or T(k) from cache, or else use CAMB to recompute it.
    """
    # Create hash of input cosmo parameters
    m = md5.new()
    keys = p.keys()
    keys.sort()
    for key in keys:
        if key is not "output_root":
            m.update("%s:%s" % (key, p[key]))
    in_hash = m.hexdigest()
    
    # Check if cached version exists; if so, make sure its hash matches
    try:
        # Get hash from header of file
        f = open(cachefile, 'r')
        header = f.readline()
        f.close()
        hash = header.split("#")[1].strip()
        
        # Compare with input hash; quit if hash doesn't match (unless force=True)
        if in_hash != hash and not force and not force_load:
            print "\tcached_camb_output: Hashes do not match; delete the cache file and run again to update."
            print "\t\tFile: %s" % cachefile
            print "\t\tHash in file:  %s" % hash
            print "\t\tHash of input: %s" % in_hash
            raise ValueError()
        
        # Check if recalculation has been forced; throw fake IOError if it has
        if force:
            print "\tcached_camb_output: Forcing recalculation of P(k)."
            raise IOError
        
        # If everything was OK, try to load from cache file, then return
        dat = np.genfromtxt(cachefile).T
        return dat
    except IOError:
        pass # Need to recompute
    except:
        raise
    
    # Set output directory to /tmp and check that paramfiles directory exists
    root = gettempdir() + "/"
    if not os.path.exists("paramfiles/"):
        os.makedirs("paramfiles/")
        print "\tcached_camb_output: Created paramfiles/ directory."
    
    # Generate unique filename, create parameter file, and run CAMB
    fname = str(uuid.uuid4())
    p['output_root'] = root + fname
    camb.camb_params("%s.ini" % fname, **p)
    output = camb.run_camb("%s.ini" % fname, camb_exec_dir=CAMB_EXEC)
    
    # Get values of cosmo. params for rescaling CAMB output
    if cosmo is not None and mode != 'cl':
        h = cosmo['h']
        sigma_8_in = cosmo['sigma_8']
        sigma8 = output['sigma8']
    elif mode != 'cl':
        h = p['hubble'] / 100.
        sigma_8_in = output['sigma8']
        sigma8 = output['sigma8']
    
    # Load requested datafile (matterpower or transfer)
    if mode == 'transfer':
        # Transfer function, T(k)
        dat = np.genfromtxt("%s%s_transfer_out.dat" % (root, fname)).T
        dat[0] *= h # Convert h^-1 Mpc => Mpc
    elif mode == 'cl' or mode == 'cls':
        # CMB C_l's
        dat = np.genfromtxt("%s%s_scalCls.dat" % (root, fname)).T
    else:
        # Matter power spectrum, P(k) [renormalised to input sigma_8]
        dat = np.genfromtxt("%s%s_matterpower.dat" % (root, fname)).T
        dat[0] *= h # Convert h^-1 Mpc => Mpc
        dat[1] *= (sigma_8_in/sigma8)**2. / h**3.
        
    # Save data to file, adding hash to header
    print "\tcached_camb_output: Saving '%s' to %s" % (mode, cachefile)
    hdr = "%s#" % in_hash
    np.savetxt(cachefile, dat.T, header=hdr)
    
    # Reload data from file (to check it's OK) and return
    dat = np.genfromtxt(cachefile).T
    return dat

def deriv_transfer(cosmo, cachefile, kmax=CAMB_KMAX, kref=1e-3, comm=None, force=False):
    """
    Return matter transfer function term used in non-Gaussianity calculation, 
    1/(T(k) k^2), and its k derivative using CAMB.
    
    (N.B. should use CAMB parameters transfer_high_precision = T and 
    transfer_k_per_logint = 1000, for accuracy)
    
    Parameters
    ----------
    
    fname : str
        Path to CAMB transfer_out.dat file.
    
    cosmo : dict
    """
    # MPI set-up
    myid = 0; size = 1
    if comm is not None:
        myid = comm.Get_rank()
        size = comm.Get_size()
    
    # Set-up CAMB parameters
    p = convert_to_camb(cosmo)
    p['transfer_kmax'] = kmax
    p['transfer_high_precision'] = 'T'
    p['transfer_k_per_logint'] = 1000
    
    # Only let one MPI worker do the calculation, then let all other workers 
    # load the result from cache
    if myid == 0:
        print "\tderiv_transfer(): Loading transfer function..."
        dat = cached_camb_output(p, cachefile, mode='transfer', force=force)
    if comm is not None: comm.barrier()
    if myid != 0:
        dat = cached_camb_output(p, cachefile, mode='transfer', force=force)
    
    # Normalise so that T(k) -> 1 as k -> 0 (see Jeong and Komatsu 2009)
    # (Actually, since it's in synch. gauge, we take T(k) = 1 at k = kref Mpc^-1; 
    # synch. gauge T(k) doesn't tend to a constant at low k as in Newtonian gauge)
    k = dat[0]; Tk = dat[6]
    idx = np.argmin(np.abs(k - kref))
    Tk /= Tk[idx]
    
    # Interpolate weighted inverse transfer fn. in rescaled (non-h^-1) units
    scale_fn = 1. / (Tk * k**2.)
    iscalefn = scipy.interpolate.interp1d( k, scale_fn, kind='linear',
                                           bounds_error=False ) #fill_value=0.)
    
    # Calculate derivative w.r.t k and interpolate
    dk = 1e-7 # Small-ish, to prevent spikes in derivative nr. boundaries
    dscalefn = (iscalefn(k + 0.5*dk) - iscalefn(k - 0.5*dk)) / dk
    idscalefn_dk = scipy.interpolate.interp1d( k, dscalefn, kind='linear',
                                               bounds_error=False )
    # TODO: Calculate derivative w.r.t. M_nu
    idscalefn_dMnu = None
    return iscalefn, idscalefn_dk, idscalefn_dMnu

def deriv_neutrinos(cosmo, cacheroot, kmax=CAMB_KMAX, force=False, 
                          mnu=0., dmnu=0.01, Neff=3.046, dNeff=0.1, comm=None):
    """
    Return numerical derivative for massive neutrinos, dlog[P(k)] / d(Sum m_nu), 
    or Neff, dlog[P(k)] / d(N_eff), using CAMB.
    
    Uses MPI if available, and caches the result. Assumes a single 
    massive neutrino species. dmnu ~ 0.01 seems to give good convergence to 
    M_nu derivative.
    
    Parameters
    ----------
    
    cosmo : dict
        Dictionary of cosmological parameters.
    
    cacheroot : str
        Filename root. Three cache files will be produced, with this string at 
        the beginning of their names, followed by numbered suffixes.
    
    kmax : float, optional
        Max. k value that CAMB should calculate
    
    force : bool, optional (default: False)
        If a cache file already exists, whether to force the calculation to be 
        re-done.
    
    mnu : float, optional
        Fiducial neutrino mass, in eV. If this is non-zero, the derivative of 
        log(P(k)) with respect to neutrino mass will be returned.
    
    dmnu : float, optional (default: 0.01)
        Finite difference for neutrino derivative (in eV).
    
    Neff : float, optional (default: 3.046)
        The fiducial effective no. of neutrino species, N_eff, to use. If mnu is 
        set to zero, the derivative with respect to Neff will be returned, 
        assuming no massive species.
        
        Otherwise, it will be assumed that there is one massive species, and 
        (Neff - 1) massless species.
    
    dNeff : float, optional (default: 0.1)
        Finite difference for Neff derivative.
    
    Returns
    -------
    
    dlogpk_dp(k) : interpolation function
        Interpolation function as a function of k, for either the M_nu or 
        N_eff derivative.
    """
    # MPI set-up
    myid = 0; size = 1
    if comm is not None:
        myid = comm.Get_rank()
        size = comm.Get_size()
    if myid == 0: print "\tderiv_logpk_neutrinos(): Loading P(k) data for derivs..."
    
    # Set finite difference values and other CAMB parameters
    p = convert_to_camb(cosmo)
    p['transfer_kmax'] = kmax
    if mnu != 0.:
        # Neutrino mass derivative
        # Set neutrino density and choose one massive neutrino species
        # (Converts Sum(m_nu) [in eV] into Omega_nu h^2, using expression from 
        # p5 of Planck 2013 XVI.)
        p['omnuh2'] = mnu / 93.04
        p['massless_neutrinos'] = Neff - 1.
        p['massive_neutrinos'] = "2 1"
        p['nu_mass_eigenstates'] = 1.
        deriv_param = 'omnuh2'
        x = p['omnuh2']
        dx = dmnu / 93.04
        dp = dmnu
    else:
        # Neff derivative
        deriv_param = 'massless_neutrinos'
        x = Neff
        dx = dNeff
        dp = dNeff
    
    # Set finite difference derivative values and load/calculate (MPI)
    xvals = [x-dx, x, x+dx]
    dat = [0 for i in range(len(xvals))] # Empty list for each worker
    for i in range(len(xvals)):
        if i % size == myid:
            fname = "%s-%d.dat" % (cacheroot, i)
            p[deriv_param] = xvals[i]
            try:
                dat[i] = cached_camb_output(p, fname, mode='matterpower', force=force)
            except:
                if comm is not None:
                    print "cached_camb_output() failed."
                    comm.Abort(errorcode=10)
                sys.exit()
    
    # Spread all of dat[] contents to every worker
    if comm is not None:
        for i in range(len(xvals)):
            src = i % size
            dat[i] = comm.bcast(dat[i], root=src)
    
    # Sanity check to make sure k values match up
    idxmin = np.min([dat[i][0].size for i in range(3)]) # Max. common idx for k array
    for i in range(len(xvals)):
        diff = np.sum(np.abs(dat[0][0][:idxmin] - dat[i][0][:idxmin]))
        if diff != 0.:
            print dat[0][0][:10]
            raise ValueError("k arrays do not match up. Summed difference: %f" % diff)
    
    # Take finite difference of P(k)
    dPk_dp = (dat[2][1][:idxmin] - dat[0][1][:idxmin]) / (2.*dp)
    dlogPk_dp = dPk_dp / dat[1][1][:idxmin]
    k = dat[0][0][:idxmin]
    
    # Interpolate result
    idlogpk = scipy.interpolate.interp1d( k, dlogPk_dp, kind='linear',
                                 bounds_error=False, fill_value=dlogPk_dp[-1] )
    return idlogpk

def logpk_derivative(pk, kgrid):
    """
    Calculate the first derivative of the (log) power spectrum, 
    d log P(k) / d k. Sets the derivative to zero wherever P(k) is not defined.
    
    Parameters
    ----------
    
    pk : function
        Callable function (usually an interpolation fn.) for P(k)
    
    kgrid : array_like
        Array of k values on which the integral will be computed.
    """
    # Calculate dlog(P(k))/dk using central difference technique
    # (Sets lowest-k values to zero since P(k) not defined there)
    # (Suppresses div/0 error temporarily)
    dk = 1e-7
    np.seterr(invalid='ignore')
    dP = pk(kgrid + 0.5*dk) / pk(kgrid - 0.5*dk)
    np.seterr(invalid=None)
    dP[np.where(np.isnan(dP))] = 1. # Set NaN values to 1 (sets deriv. to zero)
    dlogpk_dk = np.log(dP) / dk
    return dlogpk_dk
    
def fbao_derivative(fbao, kgrid, kref=[3e-2, 4.5e-1]):
    """
    Calculate the first derivative of the fbao(k) function.
    
    Parameters
    ----------
    
    fbao : function
        Callable function (usually an interpolation fn.) for fbao(k).
    
    kgrid : array_like
        Array of k values on which the integral will be computed.
    
    kref : array_like, size 2
        k range for which f_bao(k) was defined.
    """
    # Calculate dfbao/dk using central difference technique
    # (Sets lowest-k values to zero since P(k) not defined there)
    dk = 1e-7
    dfbao_dk = (fbao(kgrid + 0.5*dk) - fbao(kgrid - 0.5*dk)) / dk
    
    # Deal with sharp edge effect because of bracket over which fbao(k) was 
    # defined (zero derivative near to edge)
    idxs = np.where(kgrid <= kref[0])
    if len(idxs[0]) > 0:
        dfbao_dk[idxs] = 0.
        dfbao_dk[np.max(idxs) + 1] = 0.
    
    # Interpolate
    idfbao_dk = scipy.interpolate.interp1d( kgrid, dfbao_dk, kind='linear',
                                            bounds_error=False, fill_value=0. )
    return idfbao_dk
    
# 2.15e-2!!!!
def spline_pk_nobao(k_in, pk_in, kref=[2.15e-2, 4.5e-1]): #kref=[3e-2, 4.5e-1]):
    """
    Construct a smooth power spectrum with BAOs removed, and a corresponding 
    BAO template function, by using a two-stage splining process.
    """
    # Get interpolating function for input P(k) in log-log space
    _interp_pk = scipy.interpolate.interp1d( np.log(k_in), np.log(pk_in), 
                                        kind='quadratic', bounds_error=False )
    interp_pk = lambda x: np.exp(_interp_pk(np.log(x)))
    
    # Spline all (log-log) points except those in user-defined "wiggle region",
    # and then get derivatives of result
    idxs = np.where(np.logical_or(k_in <= kref[0], k_in >= kref[1]))
    _pk_smooth = scipy.interpolate.UnivariateSpline( np.log(k_in[idxs]), 
                                                    np.log(pk_in[idxs]), k=3, s=0 )
    pk_smooth = lambda x: np.exp(_pk_smooth(np.log(x)))

    # Construct "wiggles" function using spline as a reference, then spline it 
    # and find its 2nd derivative
    fwiggle = scipy.interpolate.UnivariateSpline(k_in, pk_in / pk_smooth(k_in), k=3, s=0)
    derivs = np.array([fwiggle.derivatives(_k) for _k in k_in]).T
    d2 = scipy.interpolate.UnivariateSpline(k_in, derivs[2], k=3, s=1.0) #s=1.
    # (s=1 to get sensible smoothing)
    
    # Find maxima and minima of the gradient (zeros of 2nd deriv.), then put a
    # low-order spline through zeros to subtract smooth trend from wiggles fn.
    wzeros = d2.roots()
    wzeros = wzeros[np.where(np.logical_and(wzeros >= kref[0], wzeros <= kref[1]))]
    wzeros = np.concatenate((wzeros, [kref[1],]))
    wtrend = scipy.interpolate.UnivariateSpline(wzeros, fwiggle(wzeros), k=3, s=0)
    
    # Construct smooth "no-bao" function by summing the original splined function and 
    # the wiggles trend function
    idxs = np.where(np.logical_and(k_in > kref[0], k_in < kref[1]))
    pk_nobao = pk_smooth(k_in)
    pk_nobao[idxs] *= wtrend(k_in[idxs])
    fk = (pk_in - pk_nobao)/pk_nobao
    
    # Construct interpolating functions
    ipk = scipy.interpolate.interp1d( k_in, pk_nobao, kind='linear',
                                      bounds_error=False, fill_value=0. )
    ifk = scipy.interpolate.interp1d( k_in, fk, kind='linear',
                                      bounds_error=False, fill_value=0. )
    return ipk, ifk


################################################################################
# Integration helper functions
################################################################################

def integrate_grid(integrand, kgrid, ugrid):
    """
    Integrate over a 2D grid of sample points using the Simpson rule.
    """
    Ik = [scipy.integrate.simps(integrand.T[i], ugrid) for i in range(kgrid.size)]
    return scipy.integrate.simps(Ik, kgrid)

def integrate_grid_cumulative(integrand, kgrid, ugrid):
    """
    Integrate over a 2D grid of sample points using the Simpson rule.
    """
    Ik = [scipy.integrate.simps(integrand.T[i], ugrid) for i in range(kgrid.size)]
    
    # Debugging plot of integrand I(k)
    if DBG_PLOT_CUMUL_INTEGRAND:
        P.subplot(111)
        P.plot(kgrid, Ik)
        P.xscale('log')
        P.show()
    return scipy.integrate.cumtrapz(Ik, kgrid, initial=0.)

def integrate_fisher_elements(derivs, kgrid, ugrid):
    """
    Construct a Fisher matrix by performing the integrals for each element and 
    populating a matrix with the results.
    """
    Nparams = len(derivs)
    K, U = np.meshgrid(kgrid, ugrid)
    
    # Loop through matrix elements, performing the appropriate integral for each
    F = np.zeros((Nparams, Nparams))
    for i in range(Nparams):
      for j in range(Nparams):
        if j >= i:
          F[i,j] = integrate_grid(K**2. * derivs[i]*derivs[j], kgrid, ugrid)
          F[j,i] = F[i,j]
    return F

def integrate_fisher_elements_cumulative(pbinned, derivs, kgrid, ugrid):
    """
    Cumulatively integrate cross-terms between a set of parameters and a 
    parameter that is binned in k.
    
    Returns the raw cumulative integral as a function of k (which can then be 
    interpolated and chopped up into bins).
    
    Parameters
    ----------
    
    pbinned : int
        Index of parameter that will be binned in k
    
    derivs : list of array_like
        List of Fisher derivative terms for each parameter
    
    kgrid, ugrid : array_like (1D)
        Mesh of k and mu values to integrate over.
    """
    p = pbinned
    Nparams = len(derivs)
    K, U = np.meshgrid(kgrid, ugrid)
    
    # Loop through parameters, performing cumulative integral for each
    integ = [ integrate_grid_cumulative(K**2. * derivs[i]*derivs[p], kgrid, ugrid) 
              for i in range(Nparams) ]
    return integ

def bin_cumulative_integrals(cumul, kgrid, kedges):
    """
    Apply a binning scheme to a set of cumulative integrals.
    
    Parameters
    ----------
    
    cumul : list of array_like
        List of cumulative integrals, returned by 
        integrate_fisher_elements_cumulative()
    
    kgrid : array_like
        Array of k values for the samples of the cumulative integral
    
    kedges : array_like
        Array of bin edges in k.
    """
    Nparams = len(cumul)
    pbins = []
    for i in range(Nparams):
        # Duplicate final element of cumulative integral array and assign to a 
        # large k value. This ensures that the interpolation behaves properly 
        # for k > kmax.
        _kgrid = np.concatenate((kgrid, [1e4,]))
        _cumul = np.concatenate((cumul[i], [cumul[i][-1],]))
        
        # Interpolate cumulative integral and get values at bin edges
        y_k = scipy.interpolate.interp1d( _kgrid, _cumul, kind='linear', 
                                          bounds_error=False, fill_value=0. )
        ybins = y_k(kedges)
        
        # Subtract between bin edges to get value for bin
        yc = np.array( [ybins[i+1] - ybins[i] for i in range(ybins.size-1)] )
        pbins.append(yc)
    
    # k bin centroids
    kc = np.array([0.5*(kedges[i+1] + kedges[i]) for i in range(kedges.size-1)])
    return kc, pbins


def expand_fisher_with_kbinned_parameter(F_old, pbins, pnew):
    """
    Add a parameter that has been binned in k to a Fisher matrix (including all 
    cross terms).
    
    The new (binned) parameter will be added to the end of the matrix. All 
    other parameters in pbins are assumed to be sorted in the same order as the 
    original Fisher matrix.
    """
    Nold = F_old.shape[0]
    Nbins = len(pbins[0])
    
    # Sanity check: Length of pbins should be no. old params + 1
    assert Nold == len(pbins) - 1, "Length of pbins should be no. old params + 1"
    
    # Calculate indices of the old parameters in the pbins list
    pidxs = [i for i in range(Nold+1)]
    pidxs.remove(pnew)
    
    # Construct new, extended matrix
    Fnew = np.zeros((Nold+Nbins, Nold+Nbins))
    Fnew[:Nold,:Nold] = F_old # Old Fisher matrix (old param. x old param.)
    for i in range(Nold): # Off-diagonals (old param. x binned param.)
        Fnew[i,Nold:Nold+Nbins] = pbins[pidxs[i]]
        Fnew[Nold:Nold+Nbins,i] = pbins[pidxs[i]]
    for i in range(Nbins): # Diagonal (binned param. x binned param.)
        j = Nold + i
        Fnew[j,j] = pbins[pnew][i]
    return Fnew


################################################################################
# Noise and signal covariances
################################################################################

def noise_rms_per_voxel(z, expt):
    """
    Convenience function for calculating the rms noise per voxel,
    (sigma_T)^2 = Tsys^2 . Sarea / (dnu . t_tot . theta_b^2 . Nd . Nb )
    """
    Tsky = 60e3 * (300.*(1. + z)/expt['nu_line'])**2.55 # Foreground sky signal (mK)
    Tsys = expt['Tinst'] + Tsky # System temperature
    I = 1. / (expt['Ndish'] * expt['Nbeam']) # Dish multiplicity
    theta_fwhm = 3e8 * (1. + z) / (1e6 * expt['nu_line']) / expt['Ddish'] # Beam FWHM
    
    sigma_T = Tsys * np.sqrt(expt['Sarea'] * I) \
            / np.sqrt(expt['dnu'] * expt['ttot'] * theta_fwhm**2.)
    return sigma_T

def noise_rms_per_voxel_interferom(z, expt):
    """
    Convenience function for calculating the rms noise per voxel,
    (sigma_T)^2 = Tsys^2 . Sarea / (dnu . t_tot . n(u)) * FOV^2
    """
    Tsky = 60e3 * (300.*(1. + z)/expt['nu_line'])**2.55 # Foreground sky signal (mK)
    Tsys = expt['Tinst'] + Tsky # System temperature
    
    nu = expt['nu_line'] / (1. + z)
    l = 3e8 / (nu*1e6)
    fov = (l / expt['Ddish'])**2.
    
    sigma_T = Tsys * np.sqrt(expt['Sarea'] * fov**2.) \
            / np.sqrt(expt['dnu'] * expt['ttot'])
    return sigma_T

def interferometer_response(q, y, cosmo, expt):
    """
    Dish multiplicity and beam factors (I * B_perp * B_par) for the noise 
    covariance of an interferometer.
    """
    c = cosmo
    kperp = q / (c['aperp']*c['r'])
    kpar = y / (c['apar']*c['rnu'])
    
    # Mario's interferometer noise calculation
    u = kperp * c['r'] / (2. * np.pi) # UV plane: |u| = d / lambda
    nu = expt['nu_line'] / (1. + c['z'])
    l = 3e8 / (nu * 1e6) # Wavelength (m)
    
    # Calculate interferometer baseline density, n(u)
    use_nx = False
    if "n(x)" in expt.keys():
        if expt['n(x)'] is not None: use_nx = True
    if use_nx:
        # Rescale n(x) with freq.-dependence
        print "\tUsing user-specified baseline density, n(u)"
        x = u / nu  # x = u / (freq [MHz])
        n_u = expt['n(x)'](x) / nu**2. # n(x) = n(u) * nu^2
        n_u[np.where(n_u == 0.)] = 1. / INF_NOISE
    else:
        # Approximate expression for n(u), assuming uniform density in UV plane
        print "\tUsing uniform baseline density, n(u) ~ const."
        u_min = expt['Dmin'] / l
        u_max = expt['Dmax'] / l
        
        # Sanity check: The physical area of the array must be greater than the 
        # combined area of the dishes
        ff = expt['Ndish'] * (expt['Ddish'] / expt['Dmax'])**2. # Filling factor
        print "\tArray filling factor: %3.3f" % ff
        if ff > 1.:
            raise ValueError("Filling factor is > 1; dishes are too big to fit in specified area (out to Dmax).")
        
        #n_u = (l*expt['Ndish']/expt['Dmax'])**2. / (2.*np.pi) * np.ones(u.shape)
        #n_u[np.where(u < u_min)] = 1. / INF_NOISE
        #n_u[np.where(u > u_max)] = 1. / INF_NOISE
        
        # New calc.
        n_u = expt['Ndish']*(expt['Ndish'] - 1.) * l**2. * np.ones(u.shape) \
              / (2. * np.pi * (expt['Dmax']**2. - expt['Dmin']**2.) )
        n_u[np.where(u < u_min)] = 1. / INF_NOISE
        n_u[np.where(u > u_max)] = 1. / INF_NOISE
    
    # FOV cut-off (DISABLED)
    #l = 3e8 / (nu * 1e6) # Wavelength (m)
    #u_fov = 1. / np.sqrt(expt['fov'])
    #n_u[np.where(u < u_fov)] = 1. / INF_NOISE
    
    # Interferometer multiplicity factor, /I/
    I = 4./9. * expt['fov'] / n_u
    
    # Gaussian in parallel direction. Perp. direction already accounted for by 
    # n(u) factor in multiplicity (I)
    sigma_kpar = np.sqrt(16.*np.log(2)) * expt['nu_line'] / (expt['dnu'] * c['rnu'])
    B_par = (y/(c['rnu']*sigma_kpar))**2.
    B_par[np.where(B_par > EXP_OVERFLOW_VAL)] = EXP_OVERFLOW_VAL
    invbeam2 = np.exp(B_par)
    
    return I * invbeam2


def dish_response(q, y, cosmo, expt):
    """
    Dish multiplicity and beam factors (I * B_perp * B_par) for the noise 
    covariance of a single-dish mode instrument.
    """
    c = cosmo
    kperp = q / (c['aperp']*c['r'])
    kpar = y / (c['apar']*c['rnu'])
    
    # Dish-mode multiplicity factor, /I/
    I = 1. / (expt['Ndish'] * expt['Nbeam'])
    
    # Define parallel/perp. beam scales
    l = 3e8 * (1. + c['z']) / (1e6 * expt['nu_line'])
    theta_fwhm = l / expt['Ddish']
    sigma_kpar = np.sqrt(16.*np.log(2)) * expt['nu_line'] / (expt['dnu'] * c['rnu'])
    sigma_kperp = np.sqrt(16.*np.log(2)) / (c['r'] * theta_fwhm)
    #            np.sqrt(2.) * expt['Ddish'] * expt['nu_line'] \
    #             / (c['r'] * D0 * (1.+c['z']))
    
    # Sanity check: Require that Sarea > Nbeam * (beam)^2
    if (expt['Sarea'] < expt['Nbeam'] / (sigma_kperp * c['r'])**2.):
        raise ValueError("Sarea is less than (Nbeam * beam^2)")
    
    # Single-dish experiment has Gaussian beams in perp. and par. directions
    # (N.B. Check for overflow values and trim them.)
    B_tot = (q/(c['r']*sigma_kperp))**2. + (y/(c['rnu']*sigma_kpar))**2.
    B_tot[np.where(B_tot > EXP_OVERFLOW_VAL)] = EXP_OVERFLOW_VAL
    invbeam2 = np.exp(B_tot)
    
    return I * invbeam2


def Cnoise(q, y, cosmo, expt, cv=False):
    """
    Noise covariance matrix, from last equality in Eq. 25 of Pedro's notes.
    A Fourier-space beam has been applied to act as a filter over the survey 
    volume. Units: mK^2.
    """
    c = cosmo
    kperp = q / (c['aperp']*c['r'])
    kpar = y / (c['apar']*c['rnu'])
    
    # Calculate noise properties
    Vsurvey = expt['Sarea'] * expt['dnutot'] / expt['nu_line']
    Tsky = 60e3 * (300.*(1.+c['z'])/expt['nu_line'])**2.55 # Foreground sky signal (mK)
    Tsys = expt['Tinst'] + Tsky
    noise = Tsys**2. * Vsurvey / (expt['ttot'] * expt['dnutot'])
    if cv: noise = 1. # Cosmic variance-limited calc.
    
    # Apply multiplicity/beam response for 
    if 'int' in expt['mode']:
        # Interferometer mode
        print "\tInterferometer mode."
        noise *= interferometer_response(q, y, cosmo, expt)
    elif 'cyl' in expt['mode']:
        # Cylinder mode
        print "\tCylinder (interferometer) mode."
        noise *= interferometer_response(q, y, cosmo, expt)
    elif 'dish' in expt['mode']:
        # Dish (autocorrelation) mode
        print "\tSingle-dish mode."
        noise *= dish_response(q, y, cosmo, expt)
    elif 'comb' in expt['mode']:
        print "\tCombined interferometer + single-dish mode"
        # Combined dish + interferom. mode
        # N.B. For each voxel, this takes the response which has the lowest 
        # noise (*either* interferom. of single dish), rather than adding the 
        # inverse noise terms together. This is correct in the CV-limited 
        # regime, since it prevents double counting of photons, but is 
        # pessimistic in the noise-dominated regime, since it throwws information away
        r_int = interferometer_response(q, y, cosmo, expt)
        r_dish = dish_response(q, y, cosmo, expt)
        #noise *= np.minimum(r_int, r_dish) # Taking the elementwise minimum
        noise *= 1./(1./r_int + 1./r_dish) # Adding in quadrature
    else:
        # Mode not recognised (safest just to raise an error)
        raise ValueError("Experiment mode not recognised. Choose 'interferom', 'dish', or 'combined'.")
    
    # Cut-off in parallel direction due to (freq.-dep.) foreground subtraction
    kfg = 2.*np.pi * expt['nu_line'] / (expt['survey_dnutot'] * c['rnu'])
    #if 'kfg_fac' in expt.keys(): kfg *= expt['kfg_fac']
    noise[np.where(kpar < kfg)] = INF_NOISE
    return noise


def Csignal(q, y, cosmo, expt):
    """
    Get (q,y)-dependent factors of the signal covariance matrix. Units: mK^2.
    """
    c = cosmo
    
    # Wavenumber and mu = cos(theta)
    kperp = q / (c['aperp']*c['r'])
    kpar = y / (c['apar']*c['rnu'])
    k = np.sqrt(kpar**2. + kperp**2.)
    u2 = (kpar / k)**2.
    
    # RSD function (bias 'btot' already includes scale-dep. bias/non-Gaussianity)
    if RSD_FUNCTION == 'kaiser':
        # Pedro's notes, Eq. 7
        Frsd = (c['btot'] + c['f']*u2)**2. * np.exp(-u2*(k*c['sigma_nl'])**2.)
    else:
        # arXiv:0812.0419, Eq. 5
        sigma_nl2_eff = (c['D'] * c['sigma_nl'])**2. * (1. - u2 + u2*(1.+c['f'])**2.)
        Frsd = (c['btot'] + c['f']*u2)**2. * np.exp(-0.5 * k**2. * sigma_nl2_eff)
    
    # Construct signal covariance and return
    cs = Frsd * (1. + c['A'] * c['fbao'](k)) * c['D']**2. * c['pk_nobao'](k)
    cs *= c['aperp']**2. * c['apar']
    return cs * c['Tb']**2. / (c['r']**2. * c['rnu'])


def Cfg(q, y, cosmo, expt):
    """
    Foreground removal noise covariance, from p13 (and Table 1) of Pedro's 
    notes. [mK^2]
    """
    fg = cosmo['foregrounds']
    n_fg = len(fg['A']) # No. of foreground components
    nu = expt['nu_line'] / (1. + cosmo['z']) # Centre freq. of redshift bin
    
    # Replace zeros in q to prevent div/0
    _q = q.copy()
    _q[np.where(q == 0.)] = 1e-100 # Set to a very small (but non-zero) value
    
    # Sum foreground noise contributions
    Cfg = 0
    for i in range(n_fg):
        Cx = fg['A'][i] * (fg['l_p'] / (2.*np.pi*_q))**fg['nx'][i] \
                        * (nu / fg['nu_p'])**fg['mx'][i]
        Cx[np.where(q == 0.)] = INF_NOISE # Mask values where q is zero
        Cfg += Cx
    
    # Scale by FG subtraction residual amplitude and return
    return expt['epsilon_fg']**2. * Cfg


def n_IM(kgrid, ugrid, cosmo, expt):
    """
    Effective "galaxy redshift survey" number density for IM survey. Unlike for 
    a galaxy redshift survey, this is actually a (strong) function of (k, mu), 
    so simply integrating over it to give n(z) doesn't necessarily give 
    sensible results.
    """
    # Convert (k, u) into (q, y)
    K, U = np.meshgrid(kgrid, ugrid)
    y = cosmo['rnu'] * K * U
    q = cosmo['r'] * K * np.sqrt(1. - U**2.)
    dn_dk = K**2. / ( Cnoise(q, y, cosmo, expt) + Cfg(q, y, cosmo, expt) )
    n_z = integrate_grid(dn_dk, kgrid, ugrid) / (2.*np.pi)**2.
    n_z *= cosmo['Tb']**2. / (cosmo['r']**2. * cosmo['rnu'])
    
    # Calculate Vsurvey
    _z = np.linspace(zmin, zmax, 1000)
    Vsurvey = C * scipy.integrate.simps(rr(_z)**2. / HH(_z), _z)
    Vsurvey *= (4.*np.pi) * expt['Sarea'] / (4.*np.pi)
    
    return n_z, Vsurvey
    

################################################################################
# Fisher matrix calculation and integrands
################################################################################

def fisher_integrands( kgrid, ugrid, cosmo, expt, massive_nu_fn=None, 
                       Neff_fn=None, transfer_fn=None, cv_limited=False,
                       galaxy_survey=False, cs_galaxy=None ):
    """
    Return integrands over (k, u) for the Fisher matrix, for all parameters.
    Order: ( A, bHI, Tb, sig2, sigma8, ns, f, aperp, apar, [Mnu], [fNL], pk )
    """
    c = cosmo
    use = expt['use']
    r = c['r']; rnu = c['rnu']
    aperp = c['aperp']; apar = c['apar']
    a = 1. / (1. + c['z'])
    
    # Convert (k, u) into (q, y)
    K, U = np.meshgrid(kgrid, ugrid)
    y = rnu * K * U
    q = r * K * np.sqrt(1. - U**2.)
    
    # Calculate k, mu
    k = np.sqrt( (aperp*q/r)**2. + (apar*y/rnu)**2. )
    u2 = y**2. / ( y**2. + (q * rnu/r * aperp/apar)**2. )
    
    # Calculate bias (incl. non-Gaussianity, if requested)
    # FIXME: Should calculate bias only in the functions that need it 
    # (i.e. shouldn't be global)
    s_k = 1. + c['beta_1'] * k + c['beta_2'] * k**2.
    if transfer_fn is None:
        b = c['btot'] = c['bHI'] * s_k
    else:
        # Get values/derivatives of matter transfer function term, 1/(T(k) k^2)
        _Tfn, _dTfn_dk, _dTfn_dmnu = transfer_fn
        Tfn = _Tfn(k); dTfn_dk = _dTfn_dk(k)
        
        # Calculate non-Gaussian bias
        fNL = c['fNL']
        delta_c = 1.686 / c['D'] # LCDM critical overdensity (e.g. Eke et al. 1996)
        alpha = (100.*c['h'])**2. * c['omega_M_0'] * delta_c * Tfn / (C**2. * c['D'])
        b = c['btot'] = c['bHI'] * s_k + 3.*(c['bHI'] - 1.) * alpha * fNL
    
    # Choose signal/noise models depending on whether galaxy or HI survey
    if galaxy_survey:
        # Calculate Csignal, Cnoise for galaxy survey
        cs = cs_galaxy(q, y, cosmo, expt)
        cn = 1./cosmo['ngal']
        ctot = cs + cn
    else:
        # Calculate Csignal, Cnoise, Cfg for HI
        cs = Csignal(q, y, cosmo, expt)
        cn = Cnoise(q, y, cosmo, expt)
        cf = Cfg(q, y, cosmo, expt)
        if not cv_limited:
            ctot = cs + cn + cf
        else:
            # Cosmic variance-limited calculation
            print "Calculation is CV-limited."
            ctot = cs + 1e-10 * (cn + cf) # Just shrink the foregrounds etc.
    
    """
    #########################################
    # FIXME
    # Output V_eff = cs/cn / (1 + cs/cn)
    kperp = np.logspace(-5., 3., 600)
    kpar = np.logspace(-5., 1., 500)
    Kperp, Kpar = np.meshgrid(kperp, kpar)
    qq = c['r'] * Kperp
    yy = c['rnu'] * Kpar
    cosmo['btot'] = cosmo['btot'][0,0] # Remove scale-dep. of bias
    
    _cs = Csignal(qq, yy, cosmo, expt)
    _cn = Cnoise(qq, yy, cosmo, expt)
    _cf = Cfg(qq, yy, cosmo, expt)
    snr = _cs / (_cn + _cf)
    Veff = snr / (1. + snr)
    mode = "sd" #"int" #"sd"
    np.save("snr-Veff-"+mode, Veff)
    np.save("snr-kperp-"+mode, kperp)
    np.save("snr-kpar-"+mode, kpar)
    np.save("snr-cn-"+mode, _cn)
    np.save("snr-cs-"+mode, _cs)
    np.save("snr-cf-"+mode, _cf)
    exit()
    #########################################
    """
    
    # Calculate derivatives of the RSD function we are using
    # (Ignores scale-dep. of bias in drsd_sk; that's included later)
    if RSD_FUNCTION == 'kaiser':
        drsd_df = 2. * u2 / (b + c['f']*u2)
        drsd_dfsig8 = 2. * u2 / (b*c['sigma_8']*c['D'] + c['f']*c['sigma_8']*c['D']*u2)
        drsd_dsig2 = -k**2. * u2
        drsd_du2 = 2.*c['f'] / (b + c['f']*u2) - (k * c['sigma_nl'])**2.
        drsd_dk = -2. * k * u2 * c['sigma_nl']**2.
    else:
        drsd_df = u2 * ( 2. * u2 / (b + c['f']*u2) \
                       - (1. + c['f']) * (k*c['sigma_nl']*c['D'])**2. )
        drsd_dsig2 = -0.5 * (k * c['D'])**2. * ( 1. - u2 + u2*(1. + c['f'])**2. )
        drsd_du2 = 2. * c['f'] / (b + c['f']*u2) \
                 - 0.5 * (k*c['sigma_nl']*c['D'])**2. * ((1. + c['f'])**2. - 1.)
        drsd_dk = -k*(c['D']*c['sigma_nl'])**2. * (1. - u2 + u2*(1. + c['f'])**2.)
    
    # Evaluate derivatives for non-Gaussian bias
    if transfer_fn is not None:
        # Prefactor for NG bias deriv. terms
        fac = 2. / (b + c['f']*u2)
        
        # Derivatives of b_NG
        dbng_fNL = 3.*(b - 1.) * alpha
        dbng_bHI = s_k + 3.*alpha*fNL # Used in deriv_bHI
        dbng_k = -3.*(b - 1.) * alpha * fNL * dTfn_dk #(2./k + dTk_k/Tk)
        dbng_f = -6.*(b - 1.) * alpha * fNL * np.log(a)
        
        # Extra terms used in EOS expansion
        dbng_omegak = -3.*(b - 1.) * alpha * fNL / c['omega_M_0']
        dbng_omegaDE = dbng_omegak
        
        # Massive neutrinos
        dbng_dmnu_term = 0.
        if massive_nu_fn is not None:
            print "\tWARNING: (M_nu, f_NL) crossterm not currently supported. Setting to 0."
            """
            # TODO: Implement dTk_mnu(k) function.
            dTk_mnu = _dTk_mnu(k)
            dbng_mnu = -3.*(b - 1.) * alpha * fNL * dTk_mnu/Tk
            dbng_dmnu_term = dbng_mnu * fac # Used in deriv_mnu
            """
            #dbng_dmnu_term = 0.
        
        # Derivatives of C_S
        dbias_k = 2. * c['bHI'] * (c['beta_1'] + 2.*k*c['beta_2']) / (b + c['f']*u2) \
                 + dbng_k * fac
        dbng_df_term = dbng_f * fac # Used in deriv_f
        
        deriv_fNL = dbng_fNL * fac * cs / ctot
        deriv_omegak_ng = dbng_omegak * fac * cs / ctot
        deriv_omegaDE_ng = dbng_omegaDE * fac * cs / ctot
    else:
        dbng_bHI = s_k
        dbias_k = 2. * c['bHI'] * (c['beta_1'] + 2.*k*c['beta_2']) / (b + c['f']*u2)
        dbng_df_term = 0.
        dbng_dmnu_term = 0.
        deriv_omegak_ng = 0.
        deriv_omegaDE_ng = 0.
    
    # Get analytic log-derivatives for parameters
    # FIXME: (f sigma8) and (b sigma8) terms don't handle NG properly
    deriv_A   = c['fbao'](k) / (1. + c['A']*c['fbao'](k)) * cs / ctot
    deriv_bHI = 2. * dbng_bHI / (b + c['f']*u2) * cs / ctot
    deriv_f   = ( use['f_rsd'] * drsd_df \
                + use['f_growthfactor'] * 2.*np.log(a) \
                + dbng_df_term ) * cs / ctot
    deriv_bsig8 = 2. * dbng_bHI / (b*c['sigma_8']*c['D'] + c['f']*c['sigma_8']*c['D']*u2) * cs / ctot
    deriv_fsig8 = drsd_dfsig8 * cs / ctot
    deriv_sig2 = drsd_dsig2 * cs / ctot
    deriv_pk = cs / ctot
    
    # Analytic log-derivatives for scale-dependent bias parameters
    deriv_beta1 = 2. * k * c['bHI'] / (b + c['f']*u2) * cs / ctot
    deriv_beta2 = 2. * k**2. * c['bHI'] / (b + c['f']*u2) * cs / ctot
    
    # Get analytic log-derivatives for new parameters
    deriv_sigma8 = (2. / c['sigma_8']) * cs / ctot
    deriv_ns = np.log(k) * cs / ctot
    deriv_Tb = (2. / c['Tb']) * cs / ctot if not galaxy_survey else 0.
    
    # Evaluate derivatives for (apar, aperp) parameters
    dlogpk_dk = logpk_derivative(c['pk_nobao'], k) # Numerical deriv.
    daperp_u2 = -2. * (rnu/r * q/y * aperp/apar * u2)**2. / aperp
    dapar_u2 =   2. * (rnu/r * q/y * aperp/apar * u2)**2. / apar
    daperp_k = (aperp*q/r)**2. / (k*aperp)
    dapar_k  = (apar*y/rnu)**2. / (k*apar)
    
    # Construct alpha derivatives
    if use['alpha_all']:
        deriv_aperp = ( (2./aperp) + drsd_du2 * daperp_u2 \
                       + (dlogpk_dk + drsd_dk + dbias_k)*daperp_k ) * cs / ctot
        deriv_apar =  ( (1./apar)  + drsd_du2 * dapar_u2 \
                       + (dlogpk_dk + drsd_dk + dbias_k)*dapar_k  ) * cs / ctot
    else:
        # Split-out alpha terms so that they can be switched on and off
        dfbao_dk = c['dfbao_dk'](k)
        t_term = [0,] * 5; r_term = [0,] * 5
        use_term = [ use['alpha_volume'], use['alpha_rsd_angle'], 
                     use['alpha_rsd_shift'], use['alpha_bao_shift'],
                     use['alpha_pk_shift'] ]
        
        # (t = transverse)
        t_term[0] = (2./aperp)
        t_term[1] = drsd_du2 * daperp_u2
        t_term[2] = drsd_dk * daperp_k
        t_term[3] = c['A'] * dfbao_dk / (1. + c['A'] * c['fbao'](k)) * daperp_k
        t_term[4] = (dlogpk_dk * daperp_k) - t_term[3]
        # t_term[3] = dlogpk_dk * daperp_k # Total P(k) shift term
        
        # (r = radial)
        r_term[0] = (1./apar)
        r_term[1] = drsd_du2 * dapar_u2
        r_term[2] = drsd_dk * dapar_k
        r_term[3] = c['A'] * dfbao_dk / (1. + c['A'] * c['fbao'](k)) * dapar_k
        r_term[4] = (dlogpk_dk * dapar_k) - r_term[3]
        # r_term[3] = dlogpk_dk * dapar_k # Total P(k) shift term
        
        # Sum-up all terms
        deriv_aperp = 0; deriv_apar = 0
        for i in range(5):
            deriv_aperp += use_term[i] * t_term[i]
            deriv_apar  += use_term[i] * r_term[i]
        deriv_aperp *= cs / ctot
        deriv_apar  *= cs / ctot
    
    # Make list of (non-optional) derivatives
    deriv_list = [ deriv_A, deriv_bHI, deriv_Tb, deriv_sig2, deriv_sigma8, 
                   deriv_ns, deriv_f, deriv_aperp, deriv_apar, 
                   deriv_bsig8, deriv_fsig8 ]
    paramnames = ['A', 'b_HI', 'Tb', 'sigma_NL', 'sigma8', 'n_s', 'f', 
                  'aperp', 'apar', 'fs8', 'bs8']
    # FIXME: Add d_beta1, d_beta2
    
    # Evaluate derivatives for massive neutrinos and add to list
    if massive_nu_fn is not None:
        deriv_mnu = massive_nu_fn(k) * cs / ctot
        deriv_list.append(deriv_mnu)
        paramnames.append('Mnu')
   
    # Evaluate derivatives for Neff and add to list
    if Neff_fn is not None:
        deriv_Neff = Neff_fn(k) * cs / ctot
        deriv_list.append(deriv_Neff)
        paramnames.append('N_eff')
   
    # Add f_NL deriv. to list
    if transfer_fn is not None:
        deriv_list.append(deriv_fNL)
        #deriv_list.append(deriv_omegak_ng) # FIXME: Currently ignored
        #deriv_list.append(deriv_omegaDE_ng)
        paramnames.append('f_NL')
    
    # Add deriv_pk to list (always assumed to be last in the list)
    deriv_list.append(deriv_pk)
    paramnames.append('pk')
    
    # Return derivs. Order is:
    # (A, bHI, Tb, sig2, sigma8, ns, f, aperp, apar, [Mnu], [Neff], [fNL], pk)
    return deriv_list, paramnames


def eos_fisher_matrix_derivs(cosmo, cosmo_fns):
    """
    Pre-calculate derivatives required to transform (aperp, apar) into dark 
    energy parameters (Omega_k, Omega_DE, w0, wa, h, gamma).
    
    Returns interpolation functions for d(f,a_perp,par)/d(DE params) as fn. of a.
    """
    w0 = cosmo['w0']; wa = cosmo['wa']
    om = cosmo['omega_M_0']; ol = cosmo['omega_lambda_0']
    ok = 1. - om - ol
    
    # Omega_DE(a) and E(a) functions
    omegaDE = lambda a: ol * np.exp(3.*wa*(a - 1.)) / a**(3.*(1. + w0 + wa))
    E = lambda a: np.sqrt( om * a**(-3.) + ok * a**(-2.) + omegaDE(a) )
    
    # Derivatives of E(z) w.r.t. parameters
    #dE_omegaM = lambda a: 0.5 * a**(-3.) / E(a)
    if np.abs(ok) < 1e-7: # Effectively zero
        dE_omegak = lambda a: 0.5 * a**(-2.) / E(a)
    else:
        dE_omegak = lambda a: 0.5 * a**(-2.) / E(a) * (1. - 1./a)
    dE_omegaM = lambda a: 0.5 * a**(-3.) / E(a)
    dE_omegaDE = lambda a: 0.5 / E(a) * (1. - 1./a**3.)
    dE_w0 = lambda a: -1.5 * omegaDE(a) * np.log(a) / E(a)
    dE_wa = lambda a: -1.5 * omegaDE(a) * (np.log(a) + 1. - a) / E(a)
    
    # Bundle functions into list (for performing repetitive operations with them)
    fns = [dE_omegak, dE_omegaDE, dE_w0, dE_wa]
    
    # Set sampling of scale factor, and precompute some values
    HH, rr, DD, ff = cosmo_fns
    aa = np.linspace(1., 1e-4, 500)
    zz = 1./aa - 1.
    EE = E(aa); fz = ff(aa)
    gamma = cosmo['gamma']; H0 = 100. * cosmo['h']; h = cosmo['h']
    
    # Derivatives of apar w.r.t. parameters
    derivs_apar = [f(aa)/EE for f in fns]
    
    # Derivatives of f(z) w.r.t. parameters
    f_fac = -gamma * fz / EE
    df_domegak  = f_fac * (EE/om + dE_omegak(aa))
    df_domegaDE = f_fac * (EE/om + dE_omegaDE(aa))
    df_w0 = f_fac * dE_w0(aa)
    df_wa = f_fac * dE_wa(aa)
    df_dh = np.zeros(aa.shape)
    df_dgamma = fz * np.log(omegaM_z(zz, cosmo))
    derivs_f = [df_domegak, df_domegaDE, df_w0, df_wa, df_dh, df_dgamma]
    
    # Calculate comoving distance (including curvature)
    r_c = scipy.integrate.cumtrapz(1./(aa**2. * EE))
    r_c = np.concatenate(([0.], r_c))
    if ok > 0.:
        r = C/(H0*np.sqrt(ok)) * np.sinh(r_c * np.sqrt(ok))
    elif ok < 0.:
        r = C/(H0*np.sqrt(-ok)) * np.sin(r_c * np.sqrt(-ok))
    else:
        r = C/H0 * r_c
    
    # Perform integrals needed to calculate derivs. of aperp
    derivs_aperp = [(C/H0)/r[1:] * scipy.integrate.cumtrapz( f(aa)/(aa * EE)**2.) 
                        for f in fns]
    
    # Add additional term to curvature integral (idx 1)
    # N.B. I think Pedro's result is wrong (for fiducial Omega_k=0 at least), 
    # so I'm commenting it out
    #derivs_aperp[1] -= (H0 * r[1:] / C)**2. / 6.
    
    # Add initial values (to deal with 1/(r=0) at origin)
    inivals = [0.5, 0.5, 0., 0.] # FIXME: Are these OK?
    derivs_aperp = [ np.concatenate(([inivals[i]], derivs_aperp[i])) 
                     for i in range(len(derivs_aperp)) ]
    
    # Add (h, gamma) derivs to aperp,apar
    derivs_aperp += [np.ones(aa.shape)/h, np.zeros(aa.shape)]
    derivs_apar  += [np.ones(aa.shape)/h, np.zeros(aa.shape)]
    
    # Construct interpolation functions
    interp_f     = [scipy.interpolate.interp1d(aa[::-1], d[::-1], 
                    kind='linear', bounds_error=False) for d in derivs_f]
    interp_apar  = [scipy.interpolate.interp1d(aa[::-1], d[::-1], 
                    kind='linear', bounds_error=False) for d in derivs_apar]
    interp_aperp = [scipy.interpolate.interp1d(aa[::-1], d[::-1], 
                    kind='linear', bounds_error=False) for d in derivs_aperp]
    return [interp_f, interp_aperp, interp_apar]


def indexes_for_sampled_fns(p, Nbins, zfns):
    """
    Return indices of all rows of a matrix for a parameter that has been 
    expanded as a function of z.
    
    Parameters
    ----------

    p : int
        The ID of the parameter to return the indices for (i.e. its ID in the 
        original, unexpanded matrix)
    
    Nbins : int
        No. of redshift bins used
    
    zfns : list
        List of IDs for the parameters that have been expanded as a fn. of z 
        (this should also be the IDs in the original unexpanded matrix, not the 
        expanded one)
    """
    ids = []
    for i in range(p+1):
        if i in zfns:
            ids += [i,]*Nbins
        else:
            ids += [i,]
    idxs = np.where(np.array(ids) == p)[0]
    return idxs

def expand_matrix_for_sampled_fn(Fold, idx_fn, Nsamp, isamp, names):
    """
    Expand a Fisher matrix to include a row/column for each sample of a 
    function. The function sample rows are all set to zero, except the one with 
    ID 'isamp' (which takes the values given in the input Fisher matrix)
    
    Parameters
    ----------
    
    Fold : array_like
        The input Fisher matrix for a single sample of the function fn(z).
    
    idx_fn : int
        Index (in Fold) of the row corresponding to the sample of fn(z).
    
    Nsamp : int
        Number of samples of fn(z).
    
    isamp : int
        The ID of the sample of the function that Fold corresponds to, e.g. for 
        5 samples of the function, valid values are {0..4}.
        
        To get a total Fisher matrix for a given set of samples, call this 
        function Nsamples times, incrementing isamp after each call.
    
    names : list
        Parameter names.
    """
    idx = idx_fn
    n = Fold.shape[0]
    F = np.zeros((n+(Nsamp-1), n+(Nsamp-1)))
    
    # Construct matrix for elements that aren't fns. of z
    Fnew = np.zeros((n+(Nsamp-1), n+(Nsamp-1)))
    Fnew[:idx,:idx] = Fold[:idx, :idx]
    Fnew[idx+Nsamp:,idx+Nsamp:] = Fold[idx+1:, idx+1:]
    Fnew[idx+Nsamp:,:idx] = Fold[idx+1:, :idx]
    Fnew[:idx,idx+Nsamp:] = Fold[:idx, idx+1:]
    
    # Add fn(z) elements into array in the correct place (given by isamp)
    Fnew[idx+isamp,idx+isamp] = Fold[idx,idx]
    Fnew[idx+isamp,:idx] = Fold[idx,:idx]
    Fnew[:idx,idx+isamp] = Fold[:idx,idx]
    Fnew[idx+isamp,idx+Nsamp:] = Fold[idx,idx+1:]
    Fnew[idx+Nsamp:,idx+isamp] = Fold[idx+1:,idx]
    
    # Update list of parameter names
    new_names = names[:idx]
    new_names += ["%s%d" % (names[idx], i) for i in range(Nsamp)]
    new_names += names[idx+1:]
    return Fnew, new_names

def combined_fisher_matrix(F_list, names, exclude=[], expand=[]):
    """
    Combine the individual redshift bin Fisher matrices from a survey into one 
    Fisher matrix. In the process, remove unwanted parameters, and expand the 
    matrix for parameters that are to be constrained as a function of z.
    
    Parameters
    ----------
    
    F_list : list of array_like
        List of Fisher matrices, sorted by redshift bin.
    
    exclude : list (optional)
        Names of parameters of the original Fisher matrices that 
        should be removed from the final matrix.
        
    expand : list (optional)
        Names of parameters of the original Fisher matrices that 
        should be expanded as functions of redshift.
    
    names : list (optional)
        Ordered list of strings, with the names of the parameters. If this is 
        given, a list of strings will be returned with the names of the 
        parameters in the total matrix.
    
    Returns
    -------
    
    Ftot : array_like
        Combined Fisher matrix.
    
    names : list
        Updated list of parameter names.
    """
    Nbins = len(F_list)
    Nparams = F_list[0].shape[0]
    
    # Get indices for parameters that will be excluded/expanded
    exclude = indices_for_param_names(names, exclude)
    expand = indices_for_param_names(names, expand)
    
    # Calculate indices of params to expand, after excluded params are removed
    idxs = [i for i in range(Nparams)]
    for exc in exclude: idxs.remove(exc)
    new_expand = [idxs.index(exp) for exp in expand]
    new_expand.sort()
    new_expand.reverse() # Need to be in reverse order
    
    # Loop through redshift bins
    Ftot = 0
    for i in range(Nbins):
        # Exclude parameters
        F, new_names = fisher_with_excluded_params(F_list[i], exclude, names)
        # Expand fns. of z
        for idx in new_expand:
            F, new_names = expand_matrix_for_sampled_fn(F, idx, Nbins, i, new_names)
        Ftot += F
    return Ftot, new_names

def add_fisher_matrices(F1, F2, lbls1, lbls2, info=False, expand=False):
    """
    Add two Fisher matrices that may not be aligned.
    """
    # If 'expand', expand the final matrix to incorporate the union of all parameters
    if expand:
        # Find params in F2 that are missing from F1
        not_found = [l for l in lbls2 if l not in lbls1]
        lbls = lbls1 + not_found
        Nnew = len(lbls)
        print "add_fisher_matrices: Expanded output matrix to include non-overlapping params:", not_found
        
        # Construct expanded F1, with additional params at the end
        Fnew = np.zeros((Nnew, Nnew))
        Fnew[:F1.shape[0],:F1.shape[0]] = F1
        lbls1 = lbls
    else:
        # New Fisher matric is found by adding to a copy of F1
        Fnew = F1.copy()
    
    # Go through lists of params, adding matrices where they overlap
    for ii in range(len(lbls2)):
      if lbls2[ii] in lbls1:
        for jj in range(len(lbls2)):
          if lbls2[jj] in lbls1:
            _i = lbls1.index(lbls2[ii])
            _j = lbls1.index(lbls2[jj])
            Fnew[_i,_j] += F2[ii,jj]
            if info: print lbls1[_i], lbls2[ii], "//", lbls1[_j], lbls2[jj]
      if (lbls2[ii] not in lbls1) and info:
        print "add_fisher_matrices:", lbls2[ii], "not found in Fisher matrix."
    
    # Return either new Fisher matrix, or new (expanded) matrix + new labels
    if expand:
        return Fnew, lbls1
    else:
        return Fnew

def transform_to_lss_distances(z, F, paramnames, cosmo_fns=None, DA=None, H=None, 
                               rescale_da=1., rescale_h=1.):
    """
    Transform constraints on D_A(z) and H(z) into constraints on the LSS 
    distance measures, D_V(z) and F(z).
    
    Parameters
    ----------
    
    z : float
        Redshift of the current bin
    
    F : array_like
        Fisher matrix for the current bin
    
    paramnames : list
        Ordered list of parameter names for the Fisher matrix
    
    cosmo_fns : tuple of functions, optional
        Functions for H(z), r(z), D(z), f(z). If specified, these are used to 
        calculate H(z) and D_A(z).
    
    DA, H : float, optional
        Values of H(z) and D_A(z) at the current redshift. These will be used 
        if cosmo_fns is not specified.
    
    rescale_da, rescale_h : float, optional
        Scaling factors for H and D_A in the original Fisher matrix.
    
    Returns
    -------
    Fnew : array_like
        Updated Fisher matrix, with D_A replaced by D_V, and H replaced by F.
    
    newnames : list
        Updated list of parameter names.
    """
    # Calculate distances
    if cosmo_fns is not None:
        HH, rr, DD, ff = cosmo_fns
        DA = rr(z) / (1. + z)
        H = HH(z)
    DV = ((1.+z)**2. * DA**2. * C*z / H)**(1./3.)
    FF = (1.+z) * DA * H / C
    
    # Indices of parameters to be replaced (DA <-> DV, H <-> F)
    idv = ida = paramnames.index('DA')
    iff = ih = paramnames.index('H')
    
    # Construct extension operator, d(D_A, H)/d(D_V, F)
    S = np.eye(F.shape[0])
    S[ida,idv] = DA / DV        # d D_A / d D_V
    S[ida,iff] = DA / (3.*FF)   # d D_A / d F
    S[ih,idv] = - H / DV        # d H / d D_V
    S[ih,iff] = 2.*H / (3.*FF)  # d H / d F
    
    # Rescale Fisher matrix for DA, H
    F[ida,:] /= rescale_da; F[:,ida] /= rescale_da
    F[ih,:] /= rescale_h; F[:,ih] /= rescale_h
    
    # Multiply old Fisher matrix by extension operator to get new Fisher matrix
    Fnew = np.dot(S.T, np.dot(F, S))
    
    # Rename parameters
    newnames = copy.deepcopy(paramnames)
    newnames[ida] = 'DV'
    newnames[ih] = 'F'
    return Fnew, newnames
    
    

def expand_fisher_matrix(z, derivs, F, names, exclude=[]):
    """
    Transform Fisher matrix to with (f, aperp, apar) parameters into one with 
    dark energy EOS parameters (Omega_k, Omega_DE, w0, wa, h, gamma) instead.
    
    Parameters
    ----------
    
    z : float
        Central redshift of the survey.
    
    derivs : 2D list of interp. fns.
        Array of interpolation functions used to evaluate derivatives needed to 
        transform to new parameter set. Precompute this using the 
        eos_fisher_matrix_derivs() function.
    
    F : array_like
        Fisher matrix for the old parameters.
    
    names : list
        List of names of the parameters in the current Fisher matrix.
    
    exclude : list, optional
        Prevent a subset of the functions [f, aperp, apar] from being converted 
        to EOS parameters. e.g. exclude = [1,] will prevent aperp from 
        contributing to the EOS parameter constraints.
    
    Returns
    -------
    
    Fnew : array_like
        Fisher matrix for the new parameters.
    
    paramnames : list, optional
        Names parameters in the expanded Fisher matrix.
    """
    a = 1. / (1. + z)
    
    # Define mapping between old and new Fisher matrices (including expanded P(k) terms)
    old = copy.deepcopy(names)
    Nold = len(old)
    oldidxs = [old.index(p) for p in ['f', 'aperp', 'apar']]
    
    # Insert new parameters immediately after 'apar'
    new_params = ['omegak', 'omegaDE', 'w0', 'wa', 'h', 'gamma']
    new = old[:old.index('apar')+1]
    new += new_params
    new += old[old.index('apar')+1:]
    newidxs = [new.index(p) for p in new_params]
    Nnew = len(new)
    
    # Construct extension operator, d(f,aperp,par)/d(beta)
    S = np.zeros((Nold, Nnew))
    for i in range(Nold):
      for j in range(Nnew):
        # Check if this is one of the indices that is being replaced
        if i in oldidxs and j in newidxs:
            # Old parameter is being replaced
            ii = oldidxs.index(i) # newidxs
            jj = newidxs.index(j)
            if ii not in exclude:
                S[i,j] = derivs[ii][jj](a)
        else:
            if old[i] == new[j]: S[i,j] = 1.
    
    # Multiply old Fisher matrix by extension operator to get new Fisher matrix
    Fnew = np.dot(S.T, np.dot(F, S))
    return Fnew, new


def fisher_with_excluded_params(F, excl, names):
    """
    Return Fisher matrix with certain rows/columns excluded
    """
    # Remove excluded parameters from Fisher matrix
    Nnew = F.shape[0] - len(excl)
    msk = [i not in excl for i in range(F.shape[0])] # 1D mask
    Fnew = F[np.outer(msk,msk)].reshape((Nnew, Nnew))
    
    # Remove names of excluded parameters
    excl_names = [names[i] for i in excl]
    new_names = copy.deepcopy(names)
    for n in excl_names: del new_names[new_names.index(n)]
    return Fnew, new_names

def indices_for_param_names(paramnames, params, warn=True):
    """
    Returns indices of parameters 
    
    Parameters
    ----------
    
    paramnames : list
        Full (ordered) list of parameters used in Fisher matrix.
    
    params : list
        List of parameter names to find indices for. If the name is given with 
        a trailing asterisk, e.g. 'pk*', all parameters *starting* with 'pk' will 
        be matched. Matches will not be made mid-string.
    
    warn : bool, optional
        Whether to show a warning if a parameter is not found. Default: true.
    
    Returns
    -------
    
    idxs : list
        List of indices of parameters specified in 'params'
    """
    if type(params) is str: params = [params,] # Convert single param. string to list
    idxs = []
    for p in params:
      if p[-1] == "*":
        # Wildcard; match all params
        for i in range(len(paramnames)):
          if p[:-1] == paramnames[i][:len(p[:-1])]:
            idxs.append(i)
      else:
        # Add index of parameter to list
        if p in paramnames:
          idxs.append( paramnames.index(p) )
        else:
          # Parameter not found; throw a warning
          if warn: print "\tindices_for_param_names(): %s not found in param list." % p
    return np.array(idxs)

def load_param_names(fname):
    """
    Load parameter names from Fisher matrix with a given filename.
    (Names are stored as a header comment.)
    """
    f = open(fname, 'r')
    hdr = f.readline()
    if hdr[0] == '#':
        names = hdr.split()[1:] # (trim leading #)
    else:
        raise ValueError("Unable to process line 0 of %s as a header." % fname)
    f.close()
    return names

def fisher( zmin, zmax, cosmo, expt, cosmo_fns, return_pk=False, kbins=None, 
            massive_nu_fn=None, Neff_fn=None, transfer_fn=None, cv_limited=False, 
            kmin=1e-7, kmax=130. ):
    """
    Return Fisher matrix (an binned power spectrum, with errors) for given
    fiducial cosmology and experimental settings.
    
    Parameters
    ----------
    
    zmin, zmax : float
        Redshift window of survey
    
    cosmo : dict
        Dictionary of fiducial cosmological parameters
    
    expt : dict
        Dictionary of experimental parameters
    
    cosmo_fns : tuple of functions
        Tuple of cosmology functions of redshift, {H(z), r(z), D(z), f(z), }. 
        These should all be callable functions.
    
    return_pk : bool, optional
        If set to True, returns errors and fiducial values for binned power 
        spectrum.
    
    kbins : int, optional
        If return_pk=True, defines the bin edges in k for the binned P(k).
    
    massive_nu_fn : interpolation fn.
        Interpolating function for calculating derivative of log[P(k)] with 
        respect to neutrino mass, Sum(mnu)
    
    Neff_fn : interpolation fn.
        Interpolating function for calculating derivative of log[P(k)] with 
        respect to effective number of neutrinos.
    
    transfer_fn : interpolation fn.
        Interpolating function for calculating T(k) and its derivative w.r.t k.
    
    Returns
    -------
    
    F : array_like (2D)
        Fisher matrix for the parameters (aperp, apar, bHI, A, sigma_nl).
    
    pk : array_like, optional
        If return_pk=True, returns errors and fiducial values for binned power 
        spectrum.
    """
    # Copy, to make sure we don't modify input expt or cosmo
    cosmo = copy.deepcopy(cosmo)
    expt = copy.deepcopy(expt)
    
    # Fetch/precompute cosmology functions
    HH, rr, DD, ff = cosmo_fns
    
    # Sanity check: k bins must be defined if return_pk is True
    if return_pk and kbins is None:
        raise NameError("If return_pk=True, kbins must be defined.")
    
    # Calculate survey redshift bounds, central redshift, and total bandwidth
    numin = expt['nu_line'] / (1. + zmax)
    numax = expt['nu_line'] / (1. + zmin)
    expt['dnutot'] = numax - numin
    z = 0.5 * (zmax + zmin)
    
    # Calculate FOV (only used for interferom. mode)
    # FOV in radians, with C = 3e8 m/s, freq = (nu [MHz])*1e6 Hz
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
        expt['n(x)'] = load_interferom_file(expt['n(x)'])
    
    # Pack values and functions into the dictionaries cosmo, expt
    cosmo['omega_HI'] = omega_HI(z, cosmo)
    cosmo['bHI'] = bias_HI(z, cosmo)
    cosmo['Tb'] = Tb(z, cosmo)
    cosmo['z'] = z; cosmo['f'] = ff(z); cosmo['D'] = DD(z)
    cosmo['r'] = rr(z); cosmo['rnu'] = C*(1.+z)**2. / HH(z) # Perp/par. dist. scales
    
    # Physical volume (in rad^2 Mpc^3) (note factor of nu_line in here)
    Vphys = expt['Sarea'] * (expt['dnutot']/expt['nu_line']) \
          * cosmo['r']**2. * cosmo['rnu']
    Vfac = np.pi * Vphys / (2. * np.pi)**3.
    
    # Set-up integration sample points in (k, u)-space
    ugrid = np.linspace(-1., 1., NSAMP_U) # N.B. Order of integ. limits is correct
    kgrid = np.logspace(np.log10(kmin), np.log10(kmax), NSAMP_K)
    
    # Calculate fbao(k) derivative
    cosmo['dfbao_dk'] = fbao_derivative(cosmo['fbao'], kgrid)
    
    # Output k values
    c = cosmo
    D0 = 0.5 * 1.02 * 300. / np.sqrt(2.*np.log(2.)) # Dish FWHM prefactor [metres]
    kfg = 2.*np.pi * expt['nu_line'] / (expt['survey_dnutot'] * c['rnu'])
    sigma_kpar = (2.*np.pi) * expt['nu_line'] / (expt['dnu'] * c['rnu'])
    sigma_kperp =  np.sqrt(2.) * expt['Ddish'] * expt['nu_line'] \
                 / (c['r'] * D0 * (1.+c['z']))
    print "-"*50
    print "kmin\t  %s" % kmin
    print "kmax\t  %s" % kmax
    print "kfg \t  %3.3e" % kfg
    print "skpar\t %6.3f" % sigma_kpar
    print "skprp\t %6.3f" % sigma_kperp
    print "lmin \t %4.1f" % (2.*np.pi / np.sqrt(expt['Sarea'])) # FIXME: Should be FOV for interferom.
    print "lmax \t %4.1f" % (sigma_kperp * c['r'])
    print "signl\t %4.4f" % (1./cosmo['sigma_nl'])
    print "Vphys\t %4.1f Gpc^3" % (Vphys/1e9)
    print "RSD fn\t %s" % RSD_FUNCTION
    print "-"*50
    
    # Sanity check on P(k)
    if kmax > cosmo['k_in_max']:
        raise ValueError(
          "Input P(k) only goes up to %3.2f Mpc^-1, but kmax is %3.2f Mpc^-1." \
          % (cosmo['k_in_max'], kmax) )
    
    # Get derivative terms for Fisher matrix integrands, then perform the 
    # integrals and populate the matrix
    derivs, paramnames = fisher_integrands( kgrid, ugrid, cosmo, expt, 
                                            massive_nu_fn=massive_nu_fn,
                                            Neff_fn=Neff_fn,
                                            transfer_fn=transfer_fn,
                                            cv_limited=cv_limited )
    F = Vfac * integrate_fisher_elements(derivs, kgrid, ugrid)
    
    # Calculate cross-terms between binned P(k) and other params
    if return_pk:
        # Do cumulative integrals for cross-terms with P(k)
        cumul = integrate_fisher_elements_cumulative(-1, derivs, kgrid, ugrid)
        
        # Calculate binned P(k) and cross-terms with other params
        kc, pbins = bin_cumulative_integrals(cumul, kgrid, kbins)
        
        # Add k-binned terms to Fisher matrix (remove non-binned P(k))
        pnew = len(cumul) - 1
        FF, paramnames = fisher_with_excluded_params(F, excl=[F.shape[0]-1], 
                                                     names=paramnames)
        F_pk = Vfac * expand_fisher_with_kbinned_parameter(FF / Vfac, pbins, pnew)
        
        # Construct dict. with info needed to rebin P(k) and cross-terms
        binning_info = {
          'F_base':  FF,
          'Vfac':    Vfac,
          'cumul':   cumul,
          'kgrid':   kgrid
        }
        
        # Append paramnames
        paramnames += ["pk%d" % i for i in range(kc.size)]
    
    # Return results
    if return_pk: return F_pk, kc, binning_info, paramnames
    return F, paramnames


class FisherMatrix(object):
    
    def __init__(self):
        """
        Fisher matrix object.
        """
        F = 0
    
