#!/usr/bin/python
"""
Wrapper for calling CAMB from inside Python programs.
"""
import os
import re
import copy
import subprocess
import numpy as np
import scipy.integrate
import scipy.interpolate

C = 3e5 # km/s

def camb_params( params_fname,
                 output_root='test', calc_late_isw='F', get_scalar_cls='T', 
                 get_vector_cls='F', get_tensor_cls='F', get_transfer='T', 
                 do_lensing='F', do_nonlinear=0, l_max_scalar=2200, 
                 l_max_tensor=1500, k_eta_max_tensor=3000, use_physical='T', 
                 ombh2=0.0226, omch2=0.112, omnuh2=0, omk=0, hubble=70, 
                 w=-1, wa=0, use_tabulated_w='F', wafile='wa.dat',
                 cs2_lam=1, temp_cmb=2.726, helium_fraction=0.24, 
                 massless_neutrinos=3.046, massive_neutrinos=0, 
                 nu_mass_eigenstates=1, nu_mass_degeneracies='', 
                 nu_mass_fractions=1, num_massive=0, share_delta_neff='T',
                 initial_power_num=1, pivot_scalar=0.05, 
                 pivot_tensor=0.05, scalar_amp__1___=2.1e-9, 
                 scalar_spectral_index__1___=0.96, scalar_nrun__1___=0, 
                 tensor_spectral_index__1___=0, initial_ratio__1___=1, 
                 reionization='T', re_use_optical_depth='T', re_optical_depth=0.09, 
                 re_redshift=11, re_delta_redshift=1.5, re_ionization_frac=-1,
                 RECFAST_fudge=1.14, RECFAST_fudge_He=0.86, RECFAST_Heswitch=6,
                 RECFAST_Hswitch='T', initial_condition=1, 
                 initial_vector='-1 0 0 0 0', vector_mode=0, COBE_normalize='F', 
                 CMB_outputscale=7.4311e12, transfer_high_precision='F', 
                 transfer_kmax=2, transfer_k_per_logint=0, 
                 transfer_num_redshifts=1, transfer_interp_matterpower='T', 
                 transfer_redshift__1___=0, transfer_filename__1___='transfer_out.dat',
                 transfer_matterpower__1___='matterpower.dat', 
                 scalar_output_file='scalCls.dat', vector_output_file='vecCls.dat',
                 tensor_output_file='tensCls.dat', total_output_file='totCls.dat', 
                 lensed_output_file='lensedCls.dat', 
                 lensed_total_output_file='lensedtotCls.dat',
                 lens_potential_output_file='lenspotentialCls.dat', 
                 FITS_filename='scalCls.fits', do_lensing_bispectrum='F',
                 do_primordial_bispectrum='F', bispectrum_nfields=1, 
                 bispectrum_slice_base_L=0, bispectrum_ndelta=3, 
                 bispectrum_delta__1___=0, bispectrum_delta__2___=2, 
                 bispectrum_delta__3___=4, bispectrum_do_fisher='F', 
                 bispectrum_fisher_noise=0, bispectrum_fisher_noise_pol=0,
                 bispectrum_fisher_fwhm_arcmin=7, 
                 bispectrum_full_output_file='', 
                 bispectrum_full_output_sparse='F', 
                 bispectrum_export_alpha_beta='F', feedback_level=1, 
                 lensing_method=1, accurate_BB='F', massive_nu_approx=1, 
                 accurate_polarization='T', accurate_reionization='T', 
                 do_tensor_neutrinos='T', do_late_rad_truncation='T', 
                 number_of_threads=0, high_accuracy_default='F', 
                 accuracy_boost=1, l_accuracy_boost=1, l_sample_boost=1 ):
    """
    Define a dictionary of all parameters in CAMB, set to their default values.
    
    (N.B. Can only use 'wa' and related parameters if you compile CAMB with 
    equations_ppf.)
    """
    # Get dict. of arguments 
    args = locals()
    
    # Get all parameters into the CAMB param.ini format
    camb_params_text = ""
    for key in args:
        keyname = key
        if "__" in key: # Rename array parameters
            keyname = key.replace("___", ")").replace("__", "(")
        line_str = "=".join((keyname, str(args[key])))
        camb_params_text += line_str + "\n"
    
    # Output params file
    print "Writing parameters to", params_fname
    f = open("paramfiles/"+params_fname, 'w')
    f.write(camb_params_text)
    f.close()

def run_camb(params_fname, camb_exec_dir):
    """
    Run CAMB, using a given (pre-written) params file (see camb_params). Waits 
    for CAMB to finish before returning. Returns a dictionary of derived values 
    output by CAMB to stdout.
    """
    # Change directory and call CAMB
    cwd = os.getcwd()
    os.chdir(camb_exec_dir)
    params_path = cwd + "/paramfiles/" + params_fname
    print "Running CAMB on", params_path
    output = subprocess.check_output(["./camb", params_path])
    
    # Capture on-screen output of derived parameters
    vals = {}
    for line in output.split("\n"):
        # Special cases: sigma8 and tau_recomb
        if "sigma8" in line:
            vals['sigma8'] = float( re.findall(r'\b\d+.\d+\b', line)[1] )
        elif "tau_recomb" in line:
            tau = re.findall(r'\b\d+.\d+\b', line)
            vals['tau_recomb/Mpc'] = float(tau[0])
            vals['tau_now/Mpc'] = float(tau[1])
        elif "z_EQ" in line:
            vals['z_EQ'] = float( re.findall(r'\b\d+.\d+\b', line)[0] )
        else:
            # All other params can just be stuffed into a dictionary
            try:
                key, val = line.split("=")
                vals[key.strip()] = float(val)
            except:
                pass
    
    # Change back to the original directory
    os.chdir(cwd)
    return vals

def comoving_dist(a, cosmo):
    """
    Comoving distance. Ignores radiation, which might shift results slightly.
    """
    aa = np.logspace(np.log10(a), 0., 1000)
    zz = 1./aa - 1.
    
    # Cosmological parameters
    H0 = (100.*cosmo['h']); w0 = cosmo['w0']; wa = cosmo['wa']
    ombh2 = cosmo['omega_b_0'] * cosmo['h']**2.
    om = cosmo['omega_M_0']; ol = cosmo['omega_lambda_0']
    ogam = 2.47e-5 / cosmo['h']**2. # Rad. density fraction, from Dodelson Eq. 2.70
    Neff = 3.046
    onu = (7./8.) * (4./11.)**(4./3.) * Neff * ogam
    ok = 1. - om - ol
    
    # Omega_DE(z) and 1/E(z)
    omegaDE = ol #* np.exp(3.*wa*(aa - 1.)) / aa**(3.*(1. + w0 + wa))
    invE = 1. / np.sqrt( om*aa + ok*aa**2. + ogam + onu + omegaDE*aa**4.) # 1/(a^2 H)
    
    # Calculate r(z), with curvature-dependent parts
    r_c = scipy.integrate.simps(invE, aa)
    if ok > 0.:
        _r = C/(H0*np.sqrt(ok)) * np.sinh(r_c * np.sqrt(ok))
    elif ok < 0.:
        _r = C/(H0*np.sqrt(-ok)) * np.sin(r_c * np.sqrt(-ok))
    else:
        _r = (C/H0) * r_c
    return _r

def rsound(a, cosmo):
    """
    Calculate the sound horizon at some scale factor, a. (In Mpc)
    """
    # Uses the following expressions from Dodelson:
    # Eq. 8.19: c_s(eta) = [3 (1+R)]^(-1/2)
    # Eq. 8.22: r_s(eta) = integral_0^eta c_s(eta') d(eta')
    # p82:      R = 3/4 rho_b / rho_gamma
    # Eq. 2.71: rho_b = Omega_b a^-3 rho_cr
    # Eq. 2.69: rho_gamma = (pi^2 / 15) (T_CMB)^4
    # Eq. 2.70: Omega_gamma h^2 = 2.47e-5
    # We have also converted the integral from conformal time, deta, to da
    
    # Scale-factor samples
    aa = np.logspace(-8., np.log10(a), 1000)
    
    # Cosmological parameters
    H0 = (100.*cosmo['h']); w0 = cosmo['w0']; wa = cosmo['wa']
    ombh2 = cosmo['omega_b_0'] * cosmo['h']**2.
    om = cosmo['omega_M_0']; ol = cosmo['omega_lambda_0']
    ogam = 2.47e-5 / cosmo['h']**2. # Rad. density fraction, from Dodelson Eq. 2.70
    Neff = 3.046
    onu = (7./8.) * (4./11.)**(4./3.) * Neff * ogam
    ok = 1. - om - ol
    
    # Omega_DE(z) and E(z)
    omegaDE = ol * np.exp(3.*wa*(aa - 1.)) / aa**(3.*(1. + w0 + wa))
    
    # Integrate sound speed
    R = 3.0364e4 * ombh2 * aa # Baryon-photon ratio
    cs = np.sqrt(3. + 3.*R) # Sound speed
    rs_integ = 1. / np.sqrt( om*aa + ok*aa**2. + ogam + onu + omegaDE*aa**4.) # 1/(a^2 H)
    rs_integ /= cs
    rs = (C/H0) * scipy.integrate.simps(rs_integ, aa)
    return rs

def cmb_to_theta(cosmo, h):
    """
    Convert input CMB parameters to a theta value; taken from params_CMB.f90, 
    function CMBToTheta(CMB), which implements Hu & Sugiyama fitting formula.
    """
    # Recast cosmological parameters into CAMB parameters
    p = {}
    cosmo['h'] = h
    p['hubble'] = 100.*cosmo['h']
    p['omch2'] = (cosmo['omega_M_0'] - cosmo['omega_b_0']) * cosmo['h']**2.
    p['ombh2'] = cosmo['omega_b_0'] * cosmo['h']**2.
    p['omk'] = 1. - (cosmo['omega_M_0'] + cosmo['omega_lambda_0'])
    
    # CAMB parameters
    ombh2 = p['ombh2']; omch2 = p['omch2']
    
    # Redshift of LSS (Hu & Sugiyama fitting formula, from CAMB)
    # N.B. This is only an approximate value. CAMB can also get a more precise 
    # value. Typical difference is ~2, i.e. ~0.2%.
    zstar = 1048. * (1. + 0.00124*ombh2**-0.738) \
          * ( 1. + (0.0783*ombh2**-0.238 / (1. + 39.5*ombh2**0.763)) \
                   * (omch2 + ombh2)**(0.560/(1. + 21.1*ombh2**1.81)) )
    astar = 1. / (1. + zstar)
    
    # Get theta = rs / r (typical agreement with CAMB is ~ 0.1%)
    # (N.B. Note different definition of angular diameter distance in CAMB)
    rs = rsound(astar, cosmo)
    rstar = comoving_dist(astar, cosmo)
    theta = rs / rstar
    return theta

def find_h_for_theta100(th100, cosmo, hmin=0.5, hmax=0.85, nsamp=20):
    """
    Find the value of h that gives a particular value of 100*theta_MC in CAMB.
    Similar to the algorithm in CosmoMC:params_CMB.f90:ParamsToCMBParams().
    
    Parameters
    ----------
    th100 : float
        Target value of 100*theta_MC
    
    cosmo : dict
        Dictionary of cosmological parameters. 'h' will be ignored. For this 
        calculation, only {'omega_M_0', 'omega_b_0', 'omega_lambda_0'} matter.
    
    hmin, hmax : float, optional
        Bounds of range of h values to search inside. Default is h = [0.5, 0.85]
    
    nsamp : int, optional
        Number of samples to return for interpolation function. Default: 20.
    
    Returns
    -------
    h : float
        Value of h that gives the input value of theta_MC.
    """
    # Get samples
    c = copy.deepcopy(cosmo)
    hvals = np.linspace(hmin, hmax, nsamp) # Range of h values to scan
    th = 100. * np.array([cmb_to_theta(c, hh) for hh in hvals])
    
    # Interpolate sample points (trend is usually very smooth, so use quadratic)
    h_match = scipy.interpolate.interp1d(th, hvals, kind='quadratic')(th100)
    return h_match
    
