#!/usr/bin/python
"""
Euclid covariance matrices, taken from arXiv:1206.1225
"""
import numpy as np

def covmat_for_fom(sig_x, sig_y, fom, sgn=1.):
    """
    Return covariance matrix, given sigma_x, sigma_y, and a FOM. Diagonal 
    elements are unique up to a sign (which must be input manually).
    
    (N.B. In figures, if ellipse leans to the left, sgn=-1., otherwise +1.)
    """
    sig_xy = sgn * np.sqrt((sig_x*sig_y)**2. - 1./fom**2.)
    cov = np.array( [[sig_x**2., sig_xy], [sig_xy, sig_y**2.]] )
    return cov

def add_planck_prior(F, lbls, info=False):
    """
    Add Planck prior to a given Fisher matrix.
    """
    print "WARNING: add_planck_prior() is obsolete."
    #lbls_planck = ['omegak', 'omegaDE', 'w0', 'wa']
    lbls_planck = ['w0', 'wa', 'omegaDE', 'omegak', 'w_m', 'w_b', 'n_s']
    
    Fpl = F.copy()
    for ii in range(len(lbls_planck)):
      if lbls_planck[ii] in lbls:
        for jj in range(len(lbls_planck)):
          if lbls_planck[jj] in lbls:
            _i = lbls.index(lbls_planck[ii])
            _j = lbls.index(lbls_planck[jj])
            Fpl[_i,_j] += planck_prior_full[ii,jj]
            if info: print lbls[_i], lbls_planck[ii], "//", lbls[_j], lbls_planck[jj]
      if lbls_planck[ii] not in lbls:
        if info: print "Planck prior:", lbls_planck[ii], "not found in Fisher matrix."
    return Fpl

def add_detf_planck_prior(F, lbls, info=False):
    """
    Add Planck prior from DETF. See process_detf_planck_fisher.py for details 
    of its construction.
    """
    F_planck = np.genfromtxt("fisher_detf_planck.dat")
    lbls_planck = ['n_s', 'omegaM', 'omegab', 'omegak', 'omegaDE', 
                   'h', 'w0', 'wa', 'logA_S']
    
    # FIXME: Should add nuisance parameters to Fisher matrix.
    print "FIXME: add_detf_planck_prior() should add nuisance parameters too."
    
    # Add prior
    Fpl = F.copy()
    for ii in range(len(lbls_planck)):
      if lbls_planck[ii] in lbls:
        for jj in range(len(lbls_planck)):
          if lbls_planck[jj] in lbls:
            _i = lbls.index(lbls_planck[ii])
            _j = lbls.index(lbls_planck[jj])
            Fpl[_i,_j] += F_planck[ii,jj]
            if info: print lbls[_i], lbls_planck[ii], "//", lbls[_j], lbls_planck[jj]
      if lbls_planck[ii] not in lbls:
        if info: print "Planck prior:", lbls_planck[ii], "not found in Fisher matrix."
    return Fpl

def euclid_to_baofisher(F, cosmo):
    """
    Transform Planck prior from Euclid science review, Amendola (et al. 2012), 
    Table 1.17, to our set of parameters.
    Fisher matrix available from Mukherjee et al., arXiv:0803.1616v1.
    
    Euclid:    {w0, wa, omega_DE, omega_k, w_m, w_b, n_s}
    BAOFisher: {n_s, w0, wa, omega_b, omega_k, omega_DE, h}
    """
    # Fiducial parameters from Mukherjee et al.
    h = 0.73; w_m = 0.127; w_b = 0.022
    ok = 0.; w0 = -1.; wa = 0.; n_s = 0.9602
    h2 = h**2.
    om = w_m / h2; ob = w_b / h2
    ode = 1. - om - ok
    
    # Our fiducial values
    _om = cosmo['omega_M_0']; _ob = cosmo['omega_b_0']
    _ode = cosmo['omega_lambda_0']; _h = cosmo['h']; _ns = cosmo['ns']
    _ok = 1. - _om - _ode; _w0 = -1.; _wa = 0.
    
    # Construct transformation matrix from derivatives
    dw0_dp  = [0., 1., 0., 0., 0., 0., 0.]
    dwa_dp  = [0., 0., 1., 0., 0., 0., 0.]
    doDE_dp = [0., 0., 0., 0., 0., 1., 0.]
    dok_dp  = [0., 0., 0., 0., 1., 0., 0.]
    dwm_dp  = [0., 0., 0., 0., -h2, -h2, 2.*h*om]
    dwb_dp  = [0., 0., 0., h2, 0., 0., 2.*h*ob]
    dns_dp  = [1., 0., 0., 0., 0., 0., 0.]
    M = [dw0_dp, dwa_dp, doDE_dp, dok_dp, dwm_dp, dwb_dp, dns_dp]
    M = np.array(M).T
    
    # Re-scale to our fiducial values
    M[0,:] *= n_s / _ns
    M[1,:] *= w0 / _w0
    M[3,:] *= ob / _ob
    M[5,:] *= ode / _ode
    M[6,:] *= h / _h
    
    # Transform into new set of parameters
    Fnew = np.dot(M, np.dot(F, M.T))
    return Fnew

def camb_to_baofisher(F, cosmo):
    """
    Transform Fisher matrix in CAMB parameters to our parameters (assumes the 
    same fiducial values)
    
    CAMB:      {n_s, w0, wa, w_b, omega_k, w_cdm, h}
    BAOFisher: {n_s, w0, wa, omega_b, omega_k, omega_DE, h}
    """
    h = cosmo['h']; h2 = cosmo['h']**2.
    om = cosmo['omega_M_0']; ob = cosmo['omega_b_0']
    ode = cosmo['omega_lambda_0']
    ok = 1. - om - ode
    
    # Construct transformation matrix from derivatives
    dns_dp = [1., 0., 0., 0., 0., 0., 0.]
    dw0_dp = [0., 1., 0., 0., 0., 0., 0.]
    dwa_dp = [0., 0., 1., 0., 0., 0., 0.]
    dwb_dp = [0., 0., 0., h2, 0., 0., 2.*h*ob]
    dok_dp = [0., 0., 0., 0., 1., 0., 0.]
    dwc_dp = [0., 0., 0., -h2, -h2, -h2, 2.*h*(1.-ok-ode-ob)]
    dh_dp  = [0., 0., 0., 0., 0., 0., 1.]
    M = [dns_dp, dw0_dp, dwa_dp, dwb_dp, dok_dp, dwc_dp, dh_dp]
    M = np.array(M).T
    
    # Transform into new set of parameters
    Fnew = np.dot(M, np.dot(F, M.T))
    return Fnew

def detf_to_baofisher(fname, cosmo):
    """
    Transform Planck prior from DETF to our set of parameters. DETF Fisher 
    matrix can be obtained from:
    http://c3.lbl.gov:8000/Trac.Cosmology/browser/Programs/FoMSWG/tags/original/DATA/PLANCK.dat?rev=842
    
    DETF:      {n_s, w_m, w_b, w_k, w_DE, deltaGamma, M, logG0, log A_S, {w_i}}
    BAOFisher: {n_s, w0, wa, omega_b, omega_k, omega_DE, h}
    """
    # Load DETF Planck Fisher matrix (indexes start from 0)
    dat = np.genfromtxt(fname).T
    N = np.max(dat[0]) + 1
    F = np.zeros((N,N))
    for k in range(dat.shape[1]):
        i = dat[0,k]
        j = dat[1,k]
        F[i,j] = dat[2,k]
    
    # DETF fiducial values (from p8 of arXiv:0901.0721)
    n_s = 0.963; w_m = 0.1326; w_b = 0.0227; w_k = 0.; w_DE = 0.3844; h = 0.719
    w0 = -1.; wa = 0.; h2 = h**2.
    om = w_m / h2; ob = w_b / h2; ok = w_k / h2; ode = w_DE / h2
    
    # Our fiducial values
    _om = cosmo['omega_M_0']; _ob = cosmo['omega_b_0']
    _ode = cosmo['omega_lambda_0']; _h = cosmo['h']; _ns = cosmo['ns']
    _ok = 1. - _om - _ode; _w0 = -1.; _wa = 0.
    a0 = 0.1; da = 0.025 # Scale-factor binning

    # Define transformation matrix (derivatives)
    M = np.zeros((7, N))
    
    # d(DETF)/d(n_s)
    M[0,0] = 1. * (n_s/_ns)
    
    # d(DETF)/d(w0), d(DETF)/d(wa)
    for i in range(36):
        aa = 1. - (float(i) + 0.5)*da # Centroid of 'a' bin (p8 of arXiv:0901.0721)
        M[1, 9+i] = 1. * (w0/_w0) # d(w_i)/d(w0)
        M[2, 9+i] = 1. - aa       # d(w_i)/d(wa)
    
    # d(DETF)/d(omega_b)
    M[3,2] = h2 * (ob/_ob) # d(w_b)/d(omega_b)
    
    # d(DETF)/d(omega_k)
    M[4,1] = -h2 # d(w_m)/d(omega_k)
    M[4,3] = h2  # d(w_k)/d(omega_k)
    
    # d(DETF)/d(omega_DE)
    M[5,1] = -h2 * (ode/_ode) # d(w_m)/d(omega_DE)
    M[5,4] = h2 * (ode/_ode)  # d(w_DE)/d(omega_DE)
    
    # d(DETF)/d(h)
    M[6,2] = 2.*h*ob * (h/_h)  # d(w_b)/d(h)
    M[6,1] = 2.*h*om * (h/_h)  # d(w_m)/d(h)
    M[6,3] = 2.*h*ok * (h/_h)  # d(w_k)/d(h)
    M[6,4] = 2.*h*ode * (h/_h) # d(w_DE)/d(h)
    
    # Transform into new set of parameters
    M = np.array(M)
    Fnew = np.dot(M, np.dot(F, M.T))
    return Fnew


# gamma, w0 (for fixed omega_k=0, wa=0) [Tbl 1.5, Fig 1.16]
cov_gamma_w_ref = covmat_for_fom(0.02, 0.017, 3052, sgn=-1.) # Reference
cov_gamma_w_opt = covmat_for_fom(0.02, 0.016, 3509, sgn=-1.) # Optimistic
cov_gamma_w_pes = covmat_for_fom(0.026, 0.02, 2106, sgn=-1.) # Pessimistic

# gamma, w0 (for fixed wa=0, but omega_k marginalised over) [Tbl 1.6, Fig 1.17]
cov_gamma_w_okmarg_ref = covmat_for_fom(0.03, 0.04, 1342, sgn=-1.)
cov_gamma_w_okmarg_opt = covmat_for_fom(0.03, 0.03, 1589, sgn=-1.)
cov_gamma_w_okmarg_pes = covmat_for_fom(0.04, 0.05, 864,  sgn=-1.)

# w0, w1 (for fixed gamma) [Tbl 1.11, Fig 1.20]
# (w0, w1 are the same as w0 and wa)
cov_w0_wa_fixed_gamma_ok_ref = covmat_for_fom(0.05, 0.16, 430, sgn=-1.)
cov_w0_wa_fixed_gamma_ref = covmat_for_fom(0.06, 0.26, 148, sgn=-1.)


# (z, f_g, sigma_f [ref]) [Tbl 1.4]
# Seems that omega_k, w_0, w_a are all marginalised over (but no gamma parameter)
sigma_f = np.array([
          [0.7, 0.76, 0.011],
          [0.8, 0.80, 0.010],
          [0.9, 0.82, 0.009],
          [1.0, 0.84, 0.009],
          [1.1, 0.86, 0.009],
          [1.2, 0.87, 0.009],
          [1.3, 0.88, 0.010],
          [1.4, 0.89, 0.010],
          [1.5, 0.91, 0.011],
          [1.6, 0.91, 0.012],
          [1.7, 0.92, 0.014],
          [1.8, 0.93, 0.014],
          [1.9, 0.93, 0.017],
          [2.0, 0.94, 0.023]  ] ).T

# D_A(z) and H(z) constraints for Euclid (unofficial, optimistic?)
# (z, y, sigma_y, y', sigma_y') [Tbl 1, arXiv:1311.6817]
# y \propto r(z), y' \propto 1 / H(z)
# => sig_DA/DA = sig_y / y
# => sig_H/H = - sig_y' / y'
bao_scales = np.array([
    [0.1, 2.758, 0.616, 27.153, 3.676],
    [0.25, 6.742, 0.250, 25.449, 1.477],
    [0.35, 9.214, 0.200, 24.877, 0.892],
    [0.45, 11.578, 0.180, 23.147, 0.617],
    [0.55, 13.904, 0.169, 22.347, 0.462],
    [0.65, 16.107, 0.162, 20.915, 0.364],
    [0.75, 18.105, 0.158, 19.681, 0.299],
    [0.85, 19.938, 0.156, 18.496, 0.252],
    [0.95, 21.699, 0.156, 17.347, 0.218],
    [1.05, 23.341, 0.157, 16.583, 0.191],
    [1.15, 25.138, 0.158, 15.434, 0.171],
    [1.25, 26.481, 0.160, 14.744, 0.154],
    [1.35, 27.515, 0.169, 13.815, 0.147],
    [1.45, 29.381, 0.185, 13.207, 0.145],
    [1.55, 30.963, 0.209, 12.481, 0.149],
    [1.65, 31.371, 0.240, 11.904, 0.156],
    [1.75, 32.904, 0.281, 11.217, 0.168],
    [1.85, 34.028, 0.338, 10.899, 0.186],
    [1.95, 34.790, 0.417, 10.294, 0.212],
    [2.05, 35.645, 0.529, 9.752, 0.250],
    [2.15, 37.341, 0.693, 9.344, 0.303] ]).T

# Massive neutrino constraints (unmarginalised, so v. optimistic)
# Euclid + Boss (Mnu = 0.125eV, normal hierarchy) [Tbls 1+3, arXiv:1012.2868]
# (Mnu, n_s), sigma(Mnu) = 0.1795, sigma(n_s) = 0.0314, corr = 0.717
cov_mnu_ns_euclid_boss = np.array([
  [0.1795**2., 0.717 * (0.1795 * 0.0314)], 
  [0.717 * (0.1795 * 0.0314), 0.0314**2.] ])

# Euclid + BOSS + Planck (Mnu = 0.125eV, normal hierarchy) [Tbls 1+3, arXiv:1012.2868]
# (Mnu, n_s), sigma(Mnu) = 0.0311, sigma(n_s) = 0.0022, corr = -0.034
cov_mnu_ns_euclid_boss_planck = np.array( [
  [0.0311**2., -0.034 * (0.0311 * 0.0022)],
  [-0.034 * (0.0311 * 0.0022), 0.0022**2.] ])



# Planck prior
# w0, wa, omega_DE, omega_k, w_m, w_b, n_s
# From Euclid science review, Amendola (et al. 2012), Table 1.17
planck_prior_full = np.array([
    [0.172276e6, 0.490320e5, 0.674392e6, -0.208974e7, 0.325219e7, -0.790504e7, -0.549427e5],
    [0.490320e5, 0.139551e5, 0.191940e6, -0.594767e6, 0.925615e6, -0.224987e7, -0.156374e5],
    [0.674392e6, 0.191940e6, 0.263997e7, -0.818048e7, 0.127310e8, -0.309450e8, -0.215078e6],
    [-0.208974e7, -0.594767e6, -0.818048e7, 0.253489e8, -0.394501e8, 0.958892e8, 0.666335e6],
    [0.325219e7, 0.925615e6, 0.127310e8, -0.394501e8, 0.633564e8, -0.147973e9, -0.501247e6],
    [-0.790504e7, -0.224987e7, -0.309450e8, 0.958892e8, -0.147973e9, 0.405079e9, 0.219009e7],
    [-0.549427e5, -0.156374e5, -0.215078e6, 0.666335e6, -0.501247e6, 0.219009e7, 0.242767e6] ]).T

planck_prior = np.zeros((4,4))
# Order: omega_k, omega_DE, w0, wa
_old_idxs = [3, 2, 0, 1]
for i in range(4):
    for j in range(4):
        planck_prior[i,j] = planck_prior_full[_old_idxs[i],_old_idxs[j]]
