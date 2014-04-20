#!/usr/bin/python
"""
Calculate basic Fisher matrix estimate for a CMB experiment, using the results 
from Dodelson.
"""
import numpy as np
import pylab as P
import copy

from baofisher import convert_to_camb, cached_camb_output, plot_corrmat
import camb_wrapper as camb
from experiments import cosmo

fsky = 0.7 # Sky fraction (DETF use f_sky = 0.7)
Tcmb = 2.725 # Monopole temp., in K

# Planck HFI beams and noise (Table 2 [beams] and Table 6 [noise] of 
# Planck 2013 results I)
freq = [70., 100., 143., 217.] # GHz (DETF use 70 - 143 GHz channels)
fwhm = [13.08, 9.59, 7.18, 4.87] # arcmin

# Noise RMS/px (uK) [Bluebook Table 1.1]
#rms_px = [[23.2, 11., 6., 12.],
#          [23.2, 11., 6., 12.]] # Noise RMS/px (uK) [Planck 2013 I]
rms_px = [ [4.7, 2.5, 2.2, 4.8],  # sqrt[(sigma_TT)^2], TT (I)
           [6.7, 4.0, 4.2, 9.8] ] # sqrt[(sigma_TT)^2], EE (Q,U)
rms_px = np.array(rms_px).T * Tcmb


def camb_deriv(paramname, dx, p):
    """
    Calculate central finite difference to estimate derivative d(C_l)/d(param)
    """
    p = copy.deepcopy(p)
    
    # Get CAMB output 
    x = p[paramname]
    xvals = [x-dx, x+dx]
    ells = []; cl_TT = []; cl_EE = []; cl_TE = []
    dat = [0 for i in range(len(xvals))] # Empty list for each worker
    for i in range(len(xvals)):
        p[paramname] = xvals[i]
        
        p['get_transfer'] = 'F' # Save some time by turning off P(k)
        fname = "cmbfisher/%s-%d.dat" % (paramname, i)
        dat[i] = cached_camb_output(p, fname, mode='cl')
        ell = dat[i][0]; Dl_TT = dat[i][1]; Dl_EE = dat[i][2]; Dl_TE = dat[i][3]
        ells.append(ell)
        cl_TT.append(2.*np.pi * Dl_TT/(ell*(ell+1.)))
        cl_EE.append(2.*np.pi * Dl_EE/(ell*(ell+1.)))
        cl_TE.append(2.*np.pi * Dl_TE/(ell*(ell+1.)))
    
    # Calculate central finite difference
    dClTT_dx = (cl_TT[1] - cl_TT[0]) / (2. * dx)
    dClEE_dx = (cl_EE[1] - cl_EE[0]) / (2. * dx)
    dClTE_dx = (cl_TE[1] - cl_TE[0]) / (2. * dx)
    return dClTT_dx, dClEE_dx, dClTE_dx

def cmb_fisher(ell, cl, rms_px, fwhm, derivs, lmax=None):
    """
    Calculate the Fisher matrix for a CMB experiment using the simple Fisher 
    prescription from Dodelson.
    
    Assumes Gaussian beams and uniform white noise.
    """
    derivs = np.array(derivs)
    
    # Fisher matrix for T/E CMB spectrum, using DETF FOMSWG technical report, Eq. A1
    # Model for noise/beam from Dodelson (around Eq. 11.111?)
    fwhm *= np.pi/180./60. # Convert arcmin -> rad
    beam_factor = ell*(ell+1.) * fwhm**2. / (8. * np.log(2.))
    beam_factor[np.where(beam_factor > 200.)] = 200. # Guard against overflow in exp()
    Bl = np.exp(beam_factor)
    
    # Signal power spectra (TT, EE, TE)
    clTT = cl[0]; clEE = cl[1]; clTE = cl[2]
    
    # Noise power spectra (assumes uniform white noise and Nside=2048)
    px_solid_angle = 4.*np.pi / (12. * 2048**2.) # Noise/px -> noise/solid angle
    nlTT = Bl * rms_px[0]**2. * px_solid_angle
    nlEE = Bl * rms_px[1]**2. * px_solid_angle
    nlTE = Bl * rms_px[0]*rms_px[1] * px_solid_angle
    
    # Construct inv. covariance, cov^-1 = (C_l + N_l)^-1, using analytic 2x2 inverse
    detW = (clEE + nlEE)*(clTT + nlTT) - (clTE + nlTE)**2.
    detW[np.where(detW == 0.)] = 1e-100 # Small, non-zero number
    Winv = [[clEE + nlEE,   -(clTE + nlTE)],
            [-(clTE + nlTE),  clTT + nlTT ]]
    Winv = np.array(Winv) / detW
    
    # Set noise to infinity for l < 30 in TE,EE, following DETF FOMSWG
    # (see p18 of technical report; it's to do with Tau uncertainties)
    Winv[1,0,:30] = 0.
    Winv[0,1,:30] = 0.
    Winv[1,1,:30] = 0.
    
    # FIXME: Disable polarisation
    #print "WARNING: Polarisation disabled."
    #Winv[1,0,:] = 0.
    #Winv[0,1,:] = 0.
    #Winv[1,1,:] = 0.
    
    # Construct derivative product matrix
    N = len(derivs)
    F = [[0 for i in range(N)] for j in range(N)]
    for i in range(N):
        for j in range(i, N):
            # Matrix of parameter derivatives of Cl's, for each ell
            D_i = np.array([[derivs[i][0], derivs[i][2]], [derivs[i][2], derivs[i][1]]])
            D_j = np.array([[derivs[j][0], derivs[j][2]], [derivs[j][2], derivs[j][1]]])
            
            # Calculate Tr[Winv dCl/dp_i Winv dCl/dp_j] for each l
            T = [ np.trace(
                  np.dot(Winv[:,:,l], 
                         np.dot(D_i[:,:,l],
                           np.dot(Winv[:,:,l], D_j[:,:,l]))))
                  for l in range(D_i.shape[-1]) ]
            T = np.array(T)
            
            # Sum over ells to get Fisher matrix
            F[i][j] = 0.5 * fsky * np.sum( ((2.*ell + 1.) * T)[2:lmax] )
            F[j][i] = F[i][j]
    F = np.array(F)
    return F

def camb_fisher_derivs(p):
    """
    Get derivatives (as a function of ell) for the full set of CAMB Fisher 
    parameters. Includes TT, EE, TE derivatives.
    """
    # Get derivatives for various parameters
    # {n_s, w0, wa, w_b, omega_k, w_cdm, h}
    d_ns = camb_deriv("scalar_spectral_index__1___", 0.004, p)
    d_w0 = camb_deriv("w", 0.01, p)
    d_wa = camb_deriv("wa", 0.05, p)
    d_wb = camb_deriv("ombh2", 0.0005, p)
    d_ok = camb_deriv("omk", 0.005, p)
    d_wc = camb_deriv("omch2", 0.0005, p)
    d_h  = camb_deriv("hubble", 0.1, p)
    derivs = [d_ns, d_w0, d_wa, d_wb, d_ok, d_wc, d_h]
    return np.array(derivs)

def camb_fiducial(p, fname="cmbfisher/fiducial.dat"):
    """
    Calculate fiducial CMB spectra.
    """
    p['get_transfer'] = 'F' # Save some time by turning off P(k)
    dat = cached_camb_output(p, fname, mode='cl')
    ell = dat[0]
    fac = 2.*np.pi / (ell*(ell+1.))
    cl_TT = dat[1] * fac
    cl_EE = dat[2] * fac
    cl_TE = dat[3] * fac
    cls = [cl_TT, cl_EE, cl_TE]
    return ell, cls

# Get fiducial Cl's
p = convert_to_camb(cosmo)
ell, cls = camb_fiducial(p, "cmbfisher/fiducial.dat")
derivs = camb_fisher_derivs(p)

# Fisher forecast
F = 0
for i in [1,2,]: # Channels
    print "Adding", freq[i], "GHz channel."
    F += cmb_fisher(ell, cls, rms_px[i], fwhm[i], derivs, lmax=1000.) # lmax=2000 for DETF
    # FIXME: Can we just add the channels like this? Surely the measurements 
    # aren't independent?! e.g. CV-limited in all channels wouldn't improve?
print F

lbls = ['n_s', 'w0', 'wa', 'w_b', 'omega_k', 'w_cdm', 'h']
sigma = 1. / np.sqrt(np.diag(F))

for i in range(len(lbls)):
    print "sigma(%7s):  %6.6f" % (lbls[i], sigma[i])

# Save Fisher matrix
np.savetxt("fisher_planck_camb.dat", F)
