#!/usr/bin/python
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
EXPT_LABEL = "_mg" #"_baoonly" #"_mg" #"_baoonly"

expt_list = [
    ( 'EuclidOpt',          e.EuclidOpt ),      # 0
    ( 'EuclidRef',          e.EuclidRef ),      # 1
    ( 'EuclidPess',         e.EuclidPess ),     # 2
    ( 'gSKAMIDMKB2',        e.SKAMIDMKB2 ),     # 3
    ( 'gSKASURASKAP',       e.SKASURASKAP ),    # 4
    ( 'gSKA2',              e.SKA2 ),           # 5
    ( 'LSST',               e.LSST ),           # 6
    ( 'BOSS',               e.BOSS ),           # 7
    ( 'WFIRST',             e.WFIRST ),         # 8
    ( 'HETDEX',             e.HETDEX )          # 9
]
names, expts = zip(*expt_list)
names = list(names); expts = list(expts)

# Take command-line argument for which survey to calculate, or set manually
if len(sys.argv) > 1:
    k = int(sys.argv[1])
else:
    raise IndexError("Need to specify ID for experiment.")
if myid == 0:
    print "="*50
    print "Survey:", names[k]
    print "="*50
names[k] += EXPT_LABEL
expt = expts[k]
survey_name = names[k]
root = "output/" + survey_name

e.load_expt(expt)
zmin = expt['zmin']
zmax = expt['zmax']

#switches = []
switches = ['mg', 'sdbias']


################################################################################

# Define kbins (used for output)
kbins = np.logspace(np.log10(0.001), np.log10(50.), 91)
#kbins = np.logspace(np.log10(0.0001), np.log10(1.), 2) # FIXME

# Precompute cosmological functions, P(k), massive neutrinos, and T(k) for f_NL
cosmo_fns = rf.background_evolution_splines(cosmo)
cosmo = rf.load_power_spectrum(cosmo, "cache_pk_gal.dat", comm=comm)
#massive_nu_fn = rf.deriv_logpk_mnu(cosmo['mnu'], cosmo, "cache_mnu010", comm=comm)
#transfer_fn = rf.deriv_transfer(cosmo, "cache_transfer.dat", comm=comm)
#Neff_fn = rf.deriv_neutrinos(cosmo, "cache_Neff", Neff=cosmo['N_eff'], comm=comm)

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
eos_derivs = rf.eos_fisher_matrix_derivs(cosmo, cosmo_fns)


################################################################################
# Loop through redshift bins, assigning them to each process
################################################################################

Ngal = np.zeros(size)
for i in range(zmin.size):
    if i % size != myid:
      continue
    
    print ">>> %2d working on redshift bin %2d -- z = %3.3f" % (myid, i, zc[i])
    
    # Calculate basic Fisher matrix
    # (A, bHI, Tb, sigma_NL, sigma8, n_s, f, aperp, apar, [Mnu], [fNL], [pk]*Nkbins)
    F_pk, kc, binning_info, paramnames =  rf.galaxy.fisher_galaxy_survey(
                                           zmin[i], zmax[i], expt['nz'][i], 
                                           expt['b'][i], cosmo, expt, cosmo_fns, 
                                           switches=switches, return_pk=True, 
                                           kbins=kbins )
    # Expand Fisher matrix with EOS parameters
    ##F_eos = rf.fisher_with_excluded_params(F, [10, 11, 12]) # Exclude P(k)
    F_eos, paramnames = rf.expand_fisher_matrix(zc[i], eos_derivs, F_pk, 
                                                   exclude=[], names=paramnames)
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
if myid == 0: print "Total no. of galaxies: %3.3e" % np.sum(Ngal_tot)
