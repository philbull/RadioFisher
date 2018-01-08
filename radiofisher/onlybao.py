#!/usr/bin/python
"""
Fisher forecasts for BAO-only, using a similar approach to the Seo & Eisenstein 
(2007) method.
"""
import numpy as np
import scipy.integrate
import baofisher
import copy, sys
from units import *
#from experiments import cosmo
import experiments_galaxy
import pylab as P
from mpi4py import MPI

# Planck 2015, synced with Danielle
cosmo = {
    'omega_M_0':        0.3142, ###
    'omega_lambda_0':   0.6858,
    'omega_b_0':        0.0491,
    'N_eff':            3.046,
    'h':                0.6726, ###
    'ns':               0.9652, ###
    'sigma_8':          0.830,
    'gamma':            0.55,
    'w0':               -1.,
    'wa':               0.,
    'fNL':              0.,
    'mnu':              0.06,
    'k_piv':            0.05, # n_s
    'aperp':            1.,
    'apar':             1.,
    'bHI0':             0.677, #0.702,
    'A':                1.,
    'sigma_nl':         7.,
    'b_1':              0.,         # Scale-dependent bias (k^2 term coeff.)
    'k0_bias':          0.1,        # Scale-dependent bias pivot scale [Mpc^-1]
    'gamma0':           0.55,
    'gamma1':           0.,
    'eta0':             0.,
    'eta1':             0.,
    'A_xi':             0.00,         # Modified gravity growth amplitude
    'logkmg':           np.log10(0.05) # New modified gravity growth scale
}


NSAMP_K = 1000
NSAMP_U = 1500

# Add massive neutrinos to cosmo parameter dict
cosmo['Mnu'] = 0.06 # Minimum neutrino mass bound, 60meV

# Set-up MPI
comm = MPI.COMM_WORLD
myid = comm.Get_rank()
size = comm.Get_size()

def dist_for_params(z, params):
    """
    Get comoving distance and expansion rate for a given set of parameters.
    """
    a = 1. / (1. + z)
    zz = np.linspace(0., 6., 200)
    aa = 1. / (1. + zz)
    
    # Set parameter values
    h = params['h']
    w0 = params['w0']
    wa = params['wa']
    oc = params['oc']
    ob = params['ob']
    onu = params['Mnu'] / (93. * h**2.) # Neutrino density scale: 93 eV
    om = oc + ob + onu
    
    ok = params['ok']
    ol = 1. - om - ok
    H0 = 100. * h
    
    # Calculate Hubble rate H(z)
    omegaDE = lambda aa: ol * np.exp(3.*wa*(aa - 1.)) / aa**(3.*(1. + w0 + wa))
    E = lambda aa: np.sqrt( om * aa**(-3.) + ok * aa**(-2.) + omegaDE(aa) )
    H = H0 * E(a)
    
    # Calculate comoving dist.
    r_c = np.concatenate( ([0.], scipy.integrate.cumtrapz(1./E(aa), zz)) )
    if ok > 0.:
        _r = C/(H0*np.sqrt(ok)) * np.sinh(r_c * np.sqrt(ok))
    elif ok < 0.:
        _r = C/(H0*np.sqrt(-ok)) * np.sin(r_c * np.sqrt(-ok))
    else:
        _r = (C/H0) * r_c
    r = scipy.interpolate.interp1d(zz, _r, kind='linear', bounds_error=False)(z)
    return H, r

def expand_fisher_matrix(F, params, new_params, derivs):
    """
    Expand a Fisher matrix with new parameters.
    """
    # Define mapping between old and new parameters
    old = copy.deepcopy(params)
    Nold = len(old)
    oldidxs = [old.index(p) for p in ['aperp', 'apar']]
    
    # Insert new parameters immediately after old Fisher matrix block
    new_params = new_params
    new = old[:old.index('apar')+1]
    new += new_params
    new += old[old.index('apar')+1:]
    newidxs = [new.index(p) for p in new_params]
    Nnew = len(new)
    
    # Construct extension operator, d(aperp,par)/d(theta)
    S = np.zeros((Nold, Nnew))
    for i in range(Nold):
      for j in range(Nnew):
        # Check if this is one of the indices that is being replaced
        if i in oldidxs and j in newidxs:
            # Old parameter is being replaced
            ii = oldidxs.index(i) # newidxs
            jj = newidxs.index(j)
            S[i,j] = derivs[ii][jj]
        else:
            if old[i] == new[j]: S[i,j] = 1.
    
    # Multiply old Fisher matrix by extension operator to get new Fisher matrix
    Fnew = np.dot(S.T, np.dot(F, S))
    return Fnew, new


def project_distances(z, F, lbls, cosmo):
    """
    Project from shift parameters, alpha, to cosmological parameters, by using 
    numerical derivatives of D_A and H.
    """
    # Neutrino density scale: 93 eV
    onu = cosmo['Mnu'] / (93. * cosmo['h']**2.)
    
    # Define cosmo parameters to project onto
    pname = ['h', 'w0', 'wa', 'oc', 'ob', 'ok', 'Mnu']
    oc = cosmo['omega_M_0'] - cosmo['omega_b_0'] - onu
    p0 = [cosmo['h'], cosmo['w0'], cosmo['wa'], oc, 
          cosmo['omega_b_0'], 0., cosmo['Mnu']]
    dp = [0.05, 0.05, 0.05, 0.01, 0.005, 0.01, 0.002]
    
    # Create new Fisher matrix with extra parameters
    Fnew = np.zeros((F.shape[0] + len(pname), F.shape[0] + len(pname)))
    Fnew[:F.shape[0],:F.shape[0]] = F
    lbls_new = lbls + pname
    
    # Fiducial parameter dict.
    params = dict( (pname[i], p0[i]) for i in range(len(pname)) )
    H, r = dist_for_params(z, params)
    
    # Calculate finite differences and project onto new params
    dHdp = []; drdp = []; derivs = [[], []] # aperp, apar
    for i in range(len(pname)):
        # +ve finite difference part
        params[pname[i]] += 0.5*dp[i]
        Hp, rp = dist_for_params(z, params)
        
        # -ve finite difference part
        params[pname[i]] -= 1.0*dp[i]
        Hm, rm = dist_for_params(z, params)
        
        # Return to fiducial value
        params[pname[i]] += 0.5*dp[i]
        
        # Calculate central finite differences
        dHdp = (Hp - Hm) / dp[i]
        drdp = (rp - rm) / dp[i]
        derivs[0].append(dHdp / H) # d(aperp)/dp
        derivs[1].append(-drdp / r) # d(apar)/dp
        
    # Project out to new parameters
    Fnew, lbls_new = expand_fisher_matrix(F, lbls, pname, derivs)
    return Fnew, lbls_new
    

def fisher_distance(zmin, zmax, fsky, nz, bz, kmin=1e-3, kmax=1., 
                    fname_pk="../cache_pk.dat"):
    """
    Construct raw Fisher matrix from BAO observable, constraining distances 
    etc. only.
    """
    kgrid = np.logspace(np.log10(kmin), np.log10(kmax), NSAMP_K)
    ugrid = np.linspace(-1., 1., NSAMP_U)
    K, U = np.meshgrid(kgrid, ugrid)
    U2 = U*U
    
    # Precompute cosmo fns.
    H, r, D, f = baofisher.background_evolution_splines(cosmo, zmax=10.)
    zc = 0.5 * (zmin + zmax)
    
    # Calculate Vsurvey
    _z = np.linspace(zmin, zmax, 500)
    Vsur = 4.*np.pi*fsky * C * scipy.integrate.simps(r(_z)**2. / H(_z), _z)
    print "\tSurvey volume: %3.2f Gpc^3" % (Vsur/1e9)
    
    # Non-linear smoothing parameters (based on conversion to Planck cosmology 
    # of values on p4 of Seo & Eisenstein 2007)
    sig_par2  = (13. * D(zc) * (1. + f(zc)))**2. # Mpc^2
    sig_perp2 = (13. * D(zc))**2. # Mpc^2
    
    # Load power spectrum and decompose into BAO + smooth parts
    print "\tLoading P(k) and extracting BAO part..."
    k_in, pk_in = np.genfromtxt(fname_pk).T
    ipk, ifk = baofisher.spline_pk_nobao(k_in, pk_in)
    idfbao_dk = baofisher.fbao_derivative(ifk, kgrid)
    print "\t  Done."
    
    # Fiducial power spectrum
    pk_smooth = D(zc)**2. * ipk(K.flatten()).reshape(K.shape)
    fbao = ifk(K.flatten()).reshape(K.shape)
    bao_smoothing = np.exp(-0.5*K**2.*(sig_perp2*(1.-U2) + sig_par2*U2))
    Ptot = (bz + f(zc) * U2)**2. * pk_smooth * (1. + fbao * bao_smoothing)
    
    # Mode weighting (effective volume factor)
    sqrtVeff = nz*Ptot / (nz*Ptot + 1.)
    
    # Fisher derivs, dlog(P_gal) / d(param)
    #fbao_fac = K * idfbao_dk(K.flatten()).reshape(K.shape) \
    #         * bao_smoothing / (1. + fbao*bao_smoothing)
    fbao_fac = K * idfbao_dk(K.flatten()).reshape(K.shape) / (1. + fbao)
    deriv_aperp = fbao_fac * (1. - U2)
    deriv_apar = fbao_fac * U2
    deriv_alpha = fbao_fac * 1.
    deriv_f = 2. * U2 / (bz + f(zc) * U2)
    deriv_b = 2. / (bz + f(zc) * U2)
    
    # Prepare list of derivs, multiply by mode weighting
    lbls = ['aperp', 'apar', 'alpha_s', 'f', 'b']
    derivs = [deriv_aperp, deriv_apar, deriv_alpha, deriv_f, deriv_b]
    derivs = [deriv * sqrtVeff for deriv in derivs]
    
    # Integrate Fisher matrix
    print "\tIntegrating Fisher matrix..."
    F = baofisher.integrate_fisher_elements(derivs, kgrid, ugrid)
    F *= Vsur / (2.*np.pi)**2. # FIXME: Factor or 2 in denom. or not?
    print "\t  Done."
    
    return F, lbls

# Choose which experiment to load
if int(sys.argv[1]) == 0:
    expt = experiments_galaxy.BOSS
    exptname = "BOSS"
elif int(sys.argv[1]) == 1:
    expt = experiments_galaxy.EuclidRef
    exptname = "EuclidRef"
elif int(sys.argv[1]) == 2:
    expt = experiments_galaxy.gSKA2MG
    exptname = "gSKA2MG"
elif int(sys.argv[1]) == 3:
    expt = experiments_galaxy.gCV_z4
    exptname = "gCVz4"
else:
    print "Need to specify experiment ID as cmdline argument."
    sys.exit(1)

# Load survey parameters
experiments_galaxy.load_expt(expt)
zmin = expt['zmin']
zmax = expt['zmax']
fsky = expt['fsky']
nz = expt['nz']
bz = expt['b']

# Loop through z bins
F_list = None
for i in range(len(zmin)):
    if i % size != myid: continue
    print "-"*40
    print "%s: Bin %d / %d on cpu %d" % (exptname, i, len(zmin), myid)
    print "-"*40

    # Calculate distance Fisher matrix for a given redshift bin
    F, lbls = fisher_distance(zmin[i], zmax[i], fsky, nz[i], bz[i], 
                              kmax=0.2, fname_pk="cache_pk.dat")

    # Project distances to cosmo parameters
    print "\tProjecting to cosmo params..."
    zc = 0.5 * (zmin[i] + zmax[i])
    F, lbls = project_distances(zc, F, lbls, cosmo)
    print "\t  Done."
    
    # Add Fisher matrix for this bin to list
    if F_list is None: F_list = np.zeros((len(zmin), F.shape[0], F.shape[1]))
    F_list[i,:,:] = F
comm.barrier()

# Reduce list of Fisher matrices to all workers
if myid == 0: print "Done Fisher calculation. Reducing..."
F_all = comm.allreduce(F_list, op=MPI.SUM)

# Combine and save Fisher matrices
if myid == 0:
    zfns = ['aperp', 'apar', 'f', 'b']
    F, lbls = baofisher.combined_fisher_matrix( F_all, expand=zfns, 
                                                names=lbls, exclude=[] )
    print lbls
    
    pmnu = lbls.index('Mnu')
    print "Neutrino mass:", 1./np.sqrt(F[pmnu,pmnu])
    
    # Save to file
    np.savetxt("fisher_bao_%s.dat" % exptname, F, header=", ".join(lbls))
    print "Saved to fisher_bao_%s.dat" % exptname
    
    # Show correlation matrix
    baofisher.plot_corrmat(F, lbls)
    P.show()
