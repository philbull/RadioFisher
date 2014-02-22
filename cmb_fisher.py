#!/usr/bin/python
"""
Calculate basic Fisher matrixc estimate for a CMB experiment, using the results 
from Dodelson.
"""
import numpy as np
import pylab as P
import copy

from baofisher import convert_to_camb, cached_camb_output, plot_corrmat
import camb_wrapper as camb
from experiments import cosmo

fsky = 1.0 # Sky fraction
Tcmb = 2.725e6 # Monopole temp., in uK

# Planck HFI beams and noise (Table 2 [beams] and Table 6 [noise] of 
# Planck 2013 results I)
freq = [70., 100., 143., 217.] # GHz
fwhm = [13.08, 9.59, 7.18, 4.87] # arcmin
rms_px = [23.2, 11., 6., 12.] # Noise RMS/px (uK)

# From Colombo et al.
#freq = [70., 100., 143., 217.]
#fwhm = [14., 9.5, 7.1, 5.0]
#rms_px = np.array([4.7, 2.5, 2.2, 4.8]) * Tcmb

def convert_to_camb_phys(cosmo):
    """
    Convert cosmological parameters to CAMB parameters.
    (N.B. CAMB derives Omega_Lambda from other density parameters)
    """
    p = {}
    p['hubble'] = 100.*cosmo['h']
    p['omch2'] = (cosmo['omega_M_0'] - cosmo['omega_b_0']) * cosmo['h']**2.
    p['ombh2'] = cosmo['omega_b_0'] * cosmo['h']**2.
    #p['omk'] = 
    p['scalar_spectral_index__1___'] = cosmo['ns']
    p['w'] = cosmo['w0']
    p['wa'] = cosmo['wa']
    return p

def camb_deriv(paramname, dx, cosmo):
    """
    Calculate central finite difference to estimate derivative d(C_l)/d(param)
    """
    c = copy.deepcopy(cosmo)
    print paramname
    
    # Check for special density parameters, ~ Omega_X h^2
    # Make sure the derivative is corrected for them
    special_params = ['w_m', 'w_b', 'w_de']
    param_map = ['omega_M_0', 'omega_b_0', 'omega_lambda_0']
    paramname_actual = paramname
    if paramname in special_params:
        dx_deriv = dx * cosmo['h']**2.
        paramname = param_map[special_params.index(paramname)]
    else:
        dx_deriv = dx
    
    # Get CAMB output 
    x = cosmo[paramname]
    xvals = [x-dx, x+dx]
    ells = []; cls = []
    dat = [0 for i in range(len(xvals))] # Empty list for each worker
    for i in range(len(xvals)):
        c[paramname] = xvals[i]
        
        p = convert_to_camb(c)
        
        # Fix density parameters
        if paramname in ['omega_M_0', 'omega_b_0', 'w_m', 'w_b']: p['omk'] = 0.
        
        p['get_transfer'] = 'F' # Save some time by turning off P(k)
        fname = "cmbfisher/%s-%d.dat" % (paramname_actual, i)
        dat[i] = cached_camb_output(p, fname, mode='cl')
        ell = dat[i][0]; Dl = dat[i][1]
        ells.append(ell); cls.append(2.*np.pi * Dl/(ell*(ell+1.)))
    
    # Calculate central finite difference
    dCl_dx = (cls[1] - cls[0]) / (2. * dx)
    return dCl_dx

def cmb_fisher(ell, cl, rms_px, fwhm, derivs, lmax=None):
    """
    Calculate the Fisher matrix for a CMB experiment using the simple Fisher 
    presciption from Dodelson.
    
    Assumes Gaussian beams and uniform white noise.
    """
    derivs = np.array(derivs)
    
    # Convert from noise/px into noise/solid angle; convert fwhm arcmin -> rad.
    px_solid_angle = 4.*np.pi / (12. * 2048**2.) # Healpix Nside=2048
    beam_solid_angle = (fwhm * np.pi/180./60.)**2.
    noise = rms_px**2. * px_solid_angle
    sigma = fwhm * np.pi/180./60.
    
    # Model for noise/cosmic variance (Dodelson Eq. 11.111)
    delta_cl = cl + noise * np.exp(ell**2. * sigma**2.)
    delta_cl *= np.sqrt(2. / ((2.*ell + 1.) * fsky))
    
    # Construct derivative product matrix
    N = len(derivs)
    F = [[0 for i in range(N)] for j in range(N)]
    for i in range(N):
        for j in range(i, N):
            F[i][j] = derivs[i] * derivs[j] / (delta_cl)**2.
            F[j][i] = F[i][j]
    
    # Sum over ell (up to lmax)
    if lmax is not None:
        F = np.array(F)
        F = np.sum(F[:,:,:lmax], axis=2)
    else:
        F = np.sum(F, axis=2) # Sum over ell
    return F

# Get fiducial Cl's
p = convert_to_camb(cosmo)
p['get_transfer'] = 'F' # Save some time by turning off P(k)
fname = "cmbfisher/fiducial.dat"
dat = cached_camb_output(p, fname, mode='cl')
ell = dat[0]; cl = dat[1] * 2.*np.pi / (ell*(ell+1.))

# Get derivatives for various parameters
d_h  = camb_deriv("h", 0.005, cosmo)
d_om = camb_deriv("omega_M_0", 0.005, cosmo)
d_ob = camb_deriv("omega_b_0", 0.002, cosmo)
d_ol = camb_deriv("omega_lambda_0", 0.005, cosmo)
d_wm = camb_deriv("w_m", 0.003, cosmo)
d_wb = camb_deriv("w_b", 0.001, cosmo)
d_wde = camb_deriv("w_de", 0.003, cosmo)
d_ns = camb_deriv("ns", 0.005, cosmo)
d_w0 = camb_deriv("w0", 0.01, cosmo)
d_wa = camb_deriv("wa", 0.05, cosmo)
#derivs = [d_h, d_om, d_ob, d_ol, d_ns, d_w0, d_wa]
#derivs = [d_h, d_wm, d_wb, d_wde, d_ns] #, d_w0, d_wa]


"""
# Plot derivatives
delta_cl = cl + winv * np.exp(ell**2. * sigma**2.)
delta_cl *= np.sqrt(2. / ((2.*ell + 1.) * fsky))
lbl = ['h', 'om', 'ob', 'ol', 'ns', 'w0', 'wa']

P.subplot(111)
for i in range(len(derivs)):
    P.plot(ell, np.abs(derivs[i])/delta_cl, label=lbl[i])
#P.xscale('log')
P.yscale('log')
P.xlim((1., 350.))
P.ylim((1e-2, 1e2))
P.legend(loc='upper right', prop={'size':'x-small'})
P.show()
"""

# Planck prior from DETF
F_planck = np.genfromtxt("fisher_detf_planck.dat")
lbls_planck = ['n_s', 'omegaM', 'omegab', 'omegak', 'omegaDE', 'h', 'w0', 'wa', 'logA_S']

# Planck prior from Euclid
import euclid
Feuc = euclid.planck_prior_full
#['w0', 'wa', 'omegaDE', 'omegak', 'w_m', 'w_b', 'n_s']

# Fisher forecast
derivs = [d_ns, d_om, d_ob, d_ol, d_h, d_w0, d_wa, d_wm, d_wb, d_wde]
F = 0
for i in range(0,4):
    F += cmb_fisher(ell, cl, rms_px[i], fwhm[i], derivs)

# Compare
print "%8s:  %11s  %11s  %11s" % ("PARAM", "DETF", "Euclid", "Fisher")
print "%8s:  %11.4e  %11.4e  %11.4e" % ("n_s",      F_planck[0,0], Feuc[6,6], F[0,0])
print "%8s:  %11.4e  %11.4e  %11.4e" % ("omega_M",  F_planck[1,1], 0.       , F[1,1])
print "%8s:  %11.4e  %11.4e  %11.4e" % ("omega_b",  F_planck[2,2], 0.       , F[2,2])
print "%8s:  %11.4e  %11.4e  %11.4e" % ("omega_k",  F_planck[3,3], Feuc[3,3], 0.)
print "%8s:  %11.4e  %11.4e  %11.4e" % ("omega_DE", F_planck[4,4], Feuc[2,2], F[3,3])
print "%8s:  %11.4e  %11.4e  %11.4e" % ("h",        F_planck[5,5], 0.,        F[4,4])
print "%8s:  %11.4e  %11.4e  %11.4e" % ("w0",       F_planck[6,6], Feuc[0,0], F[5,5])
print "%8s:  %11.4e  %11.4e  %11.4e" % ("wa",       F_planck[7,7], Feuc[1,1], F[6,6])
print "%8s:  %11.4e  %11.4e  %11.4e" % ("omh2",     0.,            Feuc[4,4], F[7,7])
print "%8s:  %11.4e  %11.4e  %11.4e" % ("obh2",     0.,            Feuc[5,5], F[8,8])









exit()

# Construct Fisher matrix
#lbl = ['h', 'om', 'ob', 'ol', 'ns', 'w0', 'wa']
lbl = ['h', 'wm', 'wb', 'wde', 'ns'] #, 'w0', 'wa']
for j in range(4):
    F = cmb_fisher(ell, cl, rms_px[j], fwhm[j], derivs)
    cov = np.linalg.inv(F)
    for i in range(len(lbl)):
        print "%4s:  %5.5f" % (lbl[i], np.sqrt(np.diag(cov)[i]))
    print "-"*50
