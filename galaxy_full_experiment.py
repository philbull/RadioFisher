#!/usr/bin/env python
"""
Calculate Fisher matrix and P(k) constraints for all redshift bins for a galaxy 
redshift survey.
"""
import numpy as np
import pylab as P
import radiofisher as rf
import matplotlib.patches
from radiofisher.units import *
from mpi4py import MPI
import radiofisher.experiments as experiments
import radiofisher.experiments_galaxy as e
import sys

comm = MPI.COMM_WORLD
myid = comm.Get_rank()
size = comm.Get_size()

################################################################################
# Load experiment
################################################################################

cosmo = experiments.cosmo

# Label experiments with different settings
#EXPT_LABEL = "_mnu" #"_baoonly" #"_mg" #"_baoonly"
#EXPT_LABEL = "_mg_Dz_kmg0.01"
#EXPT_LABEL = "_paper"
#EXPT_LABEL = "_mgD_Amg0.01_kmg0.05"
#EXPT_LABEL = "_mgD_Amg0.01_kmg0.05"
#EXPT_LABEL = "_mgD_scaledep"
#EXPT_LABEL = "_rerun"
#EXPT_LABEL = "_mnu"

#EXPT_LABEL = "_mg_Axi0.01_kmg0.01"
##EXPT_LABEL = "_mgphotoz"
#EXPT_LABEL = "_mgscaledep"
#EXPT_LABEL = "_hiraxtest"
#EXPT_LABEL = "_desicv"
EXPT_LABEL = ""

cosmo['A_xi'] = 0. #0.01
cosmo['logkmg'] = np.log10(0.01)

# A_xi: 0.01 0.1
# logkmg: 0.05, 0.01, 0.005, 0.001

expt_list = [
    ( 'EuclidOpt',          e.EuclidOpt ),          # 0
    ( 'EuclidRef',          e.EuclidRef ),          # 1
    ( 'EuclidPess',         e.EuclidPess ),         # 2
    ( 'gSKAMIDMKB2',        e.SKAMIDMKB2 ),         # 3
    ( 'gSKASURASKAP',       e.SKASURASKAP ),        # 4
    ( 'gSKA2',              e.SKA2 ),               # 5
    ( 'LSST',               e.LSST ),               # 6
    ( 'BOSS',               e.BOSS ),               # 7
    ( 'WFIRST',             e.WFIRST ),             # 8
    ( 'HETDEX',             e.HETDEX ),             # 9
    ( 'WEAVEhdeep',         e.WEAVE_deep_highz ),   # 10
    ( 'WEAVEhmid',          e.WEAVE_mid_highz ),    # 11
    ( 'WEAVEhwide',         e.WEAVE_wide_highz ),   # 12
    ( 'WEAVEldeep',         e.WEAVE_deep_lowz ),    # 13
    ( 'WEAVElmid',          e.WEAVE_mid_lowz ),     # 14
    ( 'WEAVElwide',         e.WEAVE_wide_lowz ),    # 15
    ( 'gMID_B2_Base',       e.gMID_B2_Base ),       # 16 ***
    ( 'gMID_B2_Upd',        e.gMID_B2_Upd ),        # 17
    ( 'gMID_B2_Alt',        e.gMID_B2_Alt ),        # 18
    ( 'gSKA2MG',            e.gSKA2MG ),            # 19
    ( 'HETDEXdz03',         e.HETDEXdz03 ),         # 20
    ( 'gCV',                e.gCV ),                # 21
    ( 'gMIDMK_B2_Rebase',   e.gMIDMK_B2_Rebase ),   # 22
    ( 'gMIDMK_B2_Alt',      e.gMIDMK_B2_Alt ),      # 23
    ( 'gFAST20k',           e.gFAST20k ),           # 24
    ( 'gSPHEREx1',          e.SPHEREx1 ),           # 25
    ( 'gSPHEREx2',          e.SPHEREx2 ),           # 26
    ( 'gSPHEREx3',          e.SPHEREx3 ),           # 27
    ( 'gSPHEREx4',          e.SPHEREx4 ),           # 28
    ( 'gSPHEREx5',          e.SPHEREx5 ),           # 29
    ( 'gDESI_CV',           e.DESI_CV ),            # 30
    ( 'gCVLOWZ',            e.CVLOWZ ),             # 31
    ( 'SpecTel',            e.SpecTel ),            # 32
    ( 'gCVALLZ',            e.CVALLZ ),             # 33
]
names, expts = zip(*expt_list)
names = list(names); expts = list(expts)

# Take command-line argument for which survey to calculate, or set manually
if len(sys.argv) > 1:
    k = int(sys.argv[1])
else:
    raise IndexError("Need to specify ID for experiment.")
if myid == 0:
    print("="*50)
    print("Survey:", names[k])
    print("="*50)
names[k] += EXPT_LABEL
expt = expts[k]
survey_name = names[k]


"""
################################################################################
# FIXME
sarea_vals = [100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 
              6000, 7000, 8000, 9000, 10000, 12000, 15000, 17000, 20000, 22000, 
              25000, 27000, 30000]
sarea = sarea_vals[int(sys.argv[1])]
expt = e.SKA1ref
expt['fsky'] = e.sarea_to_fsky(sarea)
#expt['fname'] = "nz_SKA1-ref_%d.dat" % sarea
#survey_name = "SKA1ref_%d" % sarea

expt['fname'] = "nz_SKA1-ref_800_1300_%d.dat" % sarea
survey_name = "SKA1ref_800_1300_%d" % sarea

if myid == 0:
    print "="*50
    print survey_name
    print "="*50
################################################################################
"""

root = "output/" + survey_name

e.load_expt(expt)
zmin = expt['zmin']
zmax = expt['zmax']

# Scale-dependent growth
cosmo['fs8_kbins'] = [0., 1e-2, 1e-1, 1e0, 1e2]

switches = []
#switches = ['mg', ] #'sdbias']


################################################################################

# Define kbins (used for output)
#kbins = np.logspace(np.log10(0.001), np.log10(50.), 91)
kbins = np.logspace(np.log10(0.001), np.log10(50.), 31)

# Neutrino mass
cosmo['mnu'] = 0. #0.1
#cosmo['mnu'] = 0.06

# Precompute cosmological functions, P(k), massive neutrinos, and T(k) for f_NL
cosmo_fns =  rf.background_evolution_splines(cosmo)
if cosmo['mnu'] != 0.:
    # Massive neutrinos
    mnu_str = "mnu%03d" % (cosmo['mnu']*100.)
    fname_pk = "cache_pk_gal_%s.dat" % mnu_str
    fname_nu = "cache_%s" % mnu_str
    survey_name += mnu_str; root += mnu_str
    cosmo = rf.load_power_spectrum(cosmo, fname_pk, comm=comm)
    mnu_fn = rf.deriv_neutrinos(cosmo, fname_nu, mnu=cosmo['mnu'], comm=comm)
else:
    # Normal operation (no massive neutrinos or non-Gaussianity)
    cosmo =  rf.load_power_spectrum(cosmo, "cache_pk_gal.dat", comm=comm)
    mnu_fn = None

# Non-Gaussianity
#transfer_fn =  rf.deriv_transfer(cosmo, "cache_transfer.dat", comm=comm)
transfer_fn = None

# Effective no. neutrinos, N_eff
#Neff_fn =  rf.deriv_neutrinos(cosmo, "cache_Neff", Neff=cosmo['N_eff'], comm=comm)
Neff_fn = None

H, r, D, f = cosmo_fns
zc = 0.5 * (zmin + zmax)


################################################################################
# Store cosmological functions
################################################################################

# Store values of cosmological functions
if myid == 0:
    # Calculate cosmo fns. at redshift bin centroids and save
    _H = H(zc)
    _dA = r(zc) / (1. + zc)
    _D = D(zc)
    _f = f(zc)
    np.savetxt(root+"-cosmofns-zc.dat", np.column_stack((zc, _H, _dA, _D, _f)))
    
    # Calculate cosmo fns. as smooth fns. of z and save
    zz = np.linspace(0., 1.05*np.max(zc), 1000)
    _H = H(zz)
    _dA = r(zz) / (1. + zz)
    _D = D(zz)
    _f = f(zz)
    np.savetxt(root+"-cosmofns-smooth.dat", np.column_stack((zz, _H, _dA, _D, _f)) )

# Precompute derivs for all processes
eos_derivs = rf.eos_fisher_matrix_derivs(cosmo, cosmo_fns, fsigma8=True)

################################################################################
# Loop through redshift bins, assigning them to each process
################################################################################

Ngal = np.zeros(size)
for i in range(zmin.size):
    if i % size != myid:
      continue
    
    print(">>> %2d working on redshift bin %d / %d -- z = %3.3f" \
          % (myid, i, zmin.size, zc[i]))
    
    # Calculate basic Fisher matrix
    # (A, bHI, Tb, sigma_NL, sigma8, n_s, f, aperp, apar, [Mnu], [fNL], [pk]*Nkbins)
    F_pk, kc, binning_info, paramnames = rf.galaxy.fisher_galaxy_survey(
                                           zmin[i], zmax[i], expt['nz'][i], 
                                           expt['b'][i], cosmo, expt, cosmo_fns, 
                                           massive_nu_fn=mnu_fn, switches=switches,
                                           return_pk=True, kbins=kbins )
    # Expand Fisher matrix with EOS parameters
    ##F_eos = rf.fisher_with_excluded_params(F, [10, 11, 12]) # Exclude P(k)
    F_eos, paramnames = rf.expand_fisher_matrix(zc[i], eos_derivs, F_pk, 
                                                   exclude=[], names=paramnames, 
                                                   fsigma8=True)
    # Expand Fisher matrix for H(z), dA(z)
    # Replace aperp with dA(zi), using product rule. aperp(z) = dA(fid,z) / dA(z)
    # (And convert dA to Gpc, to help with the numerics)
    paramnames[paramnames.index('aperp')] = 'DA'
    da = r(zc[i]) / (1. + zc[i]) / 1000. # Gpc
    F_eos[7,:] *= -1. / da
    F_eos[:,7] *= -1. / da
    
    # Replace apar with H(zi)/100, using product rule. apar(z) = H(z) / H(fid,z)
    paramnames[paramnames.index('apar')] = 'H'
    F_eos[8,:] *= 1. / H(zc[i]) * 100.
    F_eos[:,8] *= 1. / H(zc[i]) * 100.
    
    # Save Fisher matrix and k bins
    np.savetxt(root+"-fisher-full-%d.dat" % i, F_eos, header=" ".join(paramnames))
    if myid == 0: np.savetxt(root+"-fisher-kc.dat", kc)
    
    # Count no. of galaxies
    Ngal[myid] += binning_info['Vsurvey'] * expt['nz'][i]
    
    # Save P(k) rebinning info
    np.savetxt(root+"-rebin-Fbase-%d.dat" % i, np.array(binning_info['F_base']) )
    np.savetxt(root+"-rebin-cumul-%d.dat" % i, np.array(binning_info['cumul']) )
    np.savetxt(root+"-rebin-kgrid-%d.dat" % i, np.array(binning_info['kgrid']) )
    np.savetxt(root+"-rebin-Vfac-%d.dat" % i, np.array([binning_info['Vsurvey'],]) )

comm.barrier()

# Count total no. of galaxies
Ngal_tot = np.zeros(size)
comm.Reduce(Ngal, Ngal_tot, op=MPI.SUM, root=0)
if myid == 0: print("Total no. of galaxies: %3.3e" % np.sum(Ngal_tot))
