"""
Experimental settings for galaxy redshift surveys.
"""
import numpy as np

def sarea_to_fsky(sarea):
    """
    Convert a survey area, in deg^2, into a sky fraction, f_sky.
    """
    FULLSKY = 4.*np.pi * (180./np.pi)**2.
    return sarea / FULLSKY

def load_expt(expt):
    """
    Process experiment dict to load fields from file.
    """
    # No action taken if 'fname' not specified (warn if loadable fields exist)
    if 'fname' not in expt.keys():
        flagged_fields = False
        for key in expt.keys():
            if key[0] == '_': flagged_fields = True
        if flagged_fields:
            print "\tload_expt(): No filename specified; couldn't load some fields."
    else:
        # Load fields that need to be loaded
        dat = np.genfromtxt(expt['fname']).T
        for key in expt.keys():
            if key[0] == '_':
                expt[key[1:]] = dat[expt[key]]
    
    # Rescale n(z) if requested
    if 'rescale_nz' in expt.keys(): expt['nz'] *= expt['rescale_nz']
    
    # Process bias
    if 'b' not in expt.keys():
        zc = 0.5 * (expt['zmin'] + expt['zmax'])
        expt['b'] = expt['b(z)'](zc)
    return expt
    

# Define which measurements to include in forecasts
USE = {
  'f_rsd':             True,     # RSD constraint on f(z)
  'f_growthfactor':    False,    # D(z) constraint on f(z)
  'alpha_all':         True,     # Use all constraints on alpha_{perp,par}
  'alpha_volume':      False,
  'alpha_rsd_angle':   False,
  'alpha_rsd_shift':   False,
  'alpha_bao_shift':   True,
  'alpha_pk_shift':    False
}

# Generic survey parameters
SURVEY = {
    'kmin':     1e-4,   # Seo & Eisenstein say: shouldn't make much difference...
    'k_nl0':    0.14,   # 0.1 # Non-linear scale at z=0 (effective kmax)
    'use':      USE
}


#########################
# Euclid
#########################

EuclidOpt = {
    'fsky':        sarea_to_fsky(15e3),
    'fname':       'nz_euclid.dat',
    '_zmin':       0,
    '_zmax':       1,
    '_nz':         2,
    'rescale_nz':  0.67**3., # Ensures correct N_gal for Planck cosmology
    'b(z)':        lambda z: np.sqrt(1. + z),
}
EuclidOpt.update(SURVEY)

EuclidRef = {
    'fsky':        sarea_to_fsky(15e3),
    'fname':       'nz_euclid.dat',
    '_zmin':       0,
    '_zmax':       1,
    '_nz':         3,
    'rescale_nz':  0.67**3.,
    'b(z)':        lambda z: np.sqrt(1. + z),
}
EuclidRef.update(SURVEY)

EuclidPess = {
    'fsky':        sarea_to_fsky(15e3),
    'fname':       'nz_euclid.dat',
    '_zmin':       0,
    '_zmax':       1,
    '_nz':         4,
    'rescale_nz':  0.67**3.,
    'b(z)':        lambda z: np.sqrt(1. + z),
}
EuclidPess.update(SURVEY)


#########################
# SKA
#########################

SKAMIDMKB2 = {
    'fsky':        sarea_to_fsky(5e3),
    'fname':       'nz_MID_MK_B2.dat',
    '_zmin':       1,
    '_zmax':       2,
    '_nz':         3,
    '_b':          4
}
SKAMIDMKB2.update(SURVEY)

SKASURASKAP = {
    'fsky':        sarea_to_fsky(5e3),
    'fname':       'nz_SUR_ASKAP.dat',
    '_zmin':       1,
    '_zmax':       2,
    '_nz':         3,
    '_b':          4
}
SKASURASKAP.update(SURVEY)

SKA2 = {
    'fsky':        sarea_to_fsky(30e3),
    'fname':       'nz_SKA2.dat',
    '_zmin':       1,
    '_zmax':       2,
    '_nz':         3,
    '_b':          4
}
SKA2.update(SURVEY)


#########################
# LSST
#########################

LSST = {
    'fsky':        sarea_to_fsky(18e3),
    'sigma_z0':    0.05,                     # Photometric z error
    'fname':       'nz_lsst.dat',            # FIXME
    '_zmin':       0,
    '_zmax':       1,
    '_nz':         2,
    'b(z)':        lambda z: np.sqrt(1. + z) # FIXME
}
LSST.update(SURVEY)


#########################
# BOSS
#########################

BOSS = {
    'fsky':        sarea_to_fsky(10e3),
    'fname':       'nz_boss.dat',
    '_zmin':       0,
    '_zmax':       1,
    '_nz':         2,
    'b(z)':        lambda z: 2.0 + z*0. # b(z) ~ 2.0 [arXiv:1010.4915]
}
BOSS.update(SURVEY)


#########################
# WFIRST
#########################

WFIRST = {
    'fsky':        sarea_to_fsky(2e3),
    'fname':       'nz_wfirst.dat',
    '_zmin':       0,
    '_zmax':       1,
    '_nz':         2,
    'b(z)':        lambda z: 1.5 + 0.4*(z - 1.5) # p16, arXiv:1308.4164
}
WFIRST.update(SURVEY)


#########################
# HETDEX
#########################

HETDEX = {
    'fsky':        sarea_to_fsky(420.),
    'fname':       'nz_hetdex.dat',
    '_zmin':       0,
    '_zmax':       1,
    '_nz':         2,
    'b(z)':        lambda z: 2.2 + z*0. # b1=2.2, arXiv:1306.4157
}
HETDEX.update(SURVEY)


