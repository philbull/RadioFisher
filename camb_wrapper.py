#!/usr/bin/python

import os
import subprocess

def camb_params( params_fname,
                 output_root='test', calc_late_isw='F', get_scalar_cls='T', 
                 get_vector_cls='F', get_tensor_cls='F', get_transfer='T', 
                 do_lensing='T', do_nonlinear=0, l_max_scalar=2200, 
                 l_max_tensor=1500, k_eta_max_tensor=3000, use_physical='T', 
                 ombh2=0.0226, omch2=0.112, omnuh2=0, omk=0, hubble=70, w=-1, 
                 cs2_lam=1, temp_cmb=2.726, helium_fraction=0.24, 
                 massless_neutrinos=3.046, massive_neutrinos=0, 
                 nu_mass_eigenstates=1, nu_mass_degeneracies=0, 
                 nu_mass_fractions=1, initial_power_num=1, pivot_scalar=0.05, 
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
    """Define a dictionary of all parameters in CAMB, set to their default values."""
    
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

def run_camb(params_fname, camb_exec_dir="/home/phil/oslo/iswfunction/cosmomc/camb"):
    """
    Run CAMB, using given params file.
    """
    # Change directory and call CAMB
    cwd = os.getcwd()
    os.chdir(camb_exec_dir)
    params_path = cwd + "/paramfiles/" + params_fname
    print "Running CAMB on", params_path
    subprocess.call(["./camb", params_path])
    os.chdir(cwd)

