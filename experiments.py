import numpy as np
import scipy.interpolate
from units import *

# Define fiducial cosmology and parameters
# Planck-only best-fit parameters, from Table 2 of Planck 2013 XVI.
cosmo = {
    'omega_M_0':        0.316,
    'omega_lambda_0':   0.684,
    'omega_b_0':        0.049,
    'omega_HI_0':       6.50e-4, # 9.4e-4
    'N_eff':            3.046,
    'h':                0.67,
    'ns':               0.962,
    'sigma_8':          0.834,
    'gamma':            0.55,
    'w0':               -1.,
    'wa':               0.,
    'fNL':              0.,
    'mnu':              0.,
    'aperp':            1.,
    'apar':             1.,
    'bHI0':             0.702,
    'A':                1.,
    'sigma_nl':         7.,
    'beta_1':           0.,         # Scale-dependent bias (k^1 term coeff. [Mpc])
    'beta_2':           0.          # Scale-dependent bias (k^2 term coeff. [Mpc^2])
}

# Define which measurements to include in forecasts
USE = {
  'f_rsd':             True,     # RSD constraint on f(z)
  'f_growthfactor':    False,    # D(z) constraint on f(z)
  'alpha_all':         True,     # Use all constraints on alpha_{perp,par}
  'alpha_volume':      False,
  'alpha_rsd_angle':   False, #t
  'alpha_rsd_shift':   False, #t
  'alpha_bao_shift':   True, # was True
  'alpha_pk_shift':    False # True
}

SURVEY = {
    'ttot':             10e3*HRS_MHZ,      # Total integration time [MHz^-1]
    'nu_line':          1420.406,          # Rest-frame freq. of emission line [MHz]
    'epsilon_fg':       1e-6,              # FG subtraction residual amplitude
    'use':              USE                # Which constraints to use/ignore
}

# Add foreground components to cosmology dict.
# (Extragal. ptsrc, extragal. free-free, gal. synch., gal. free-free)
foregrounds = {
    'A':     [57.0, 0.014, 700., 0.088],        # FG noise amplitude [mK^2]
    'nx':    [1.1, 1.0, 2.4, 3.0],              # Angular scale powerlaw index
    'mx':    [-2.07, -2.10, -2.80, -2.15],      # Frequency powerlaw index
    'l_p':   1000.,                             # Reference angular scale
    'nu_p':  130.                               # Reference frequency [MHz]
}
cosmo['foregrounds'] = foregrounds


################################################################################
# Illustrative experiments used in paper
################################################################################

exptS = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            1,                 # No. of dishes
    'Nbeam':            50,                # No. of beams (for multi-pixel detectors)
    'Ddish':            30.,               # Single dish diameter [m]
    'Tinst':            50.*(1e3),         # System temp. [mK]
    'survey_dnutot':    300.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1100.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            5e3*(D2RAD)**2.    # Total survey area [radians^2]
    }
exptS.update(SURVEY)

exptM = {
    'mode':             'interferom',      # Interferometer or single dish
    'Ndish':            160, #128          # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            4.,                # Single dish diameter [m]
    'Tinst':            35.*(1e3),         # System temp. [mK]
    'survey_dnutot':    400.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1000., #800.,      # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            2e3*(D2RAD)**2.,   # Total survey area [radians^2]
    'Dmax':             60.,               # Max. interferom. baseline [m]
    'Dmin':             4.                 # Min. interferom. baseline [m]
    }
exptM.update(SURVEY)

exptL = {
    'mode':             'combined',        # Interferometer or single dish
    'Ndish':            250,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    700.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1100.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'Dmax':             600.,              # Max. interferom. baseline [m]
    'Dmin':             15.                # Min. interferom. baseline [m]
    }
exptL.update(SURVEY)

# Matched to Euclid redshift/Sarea
exptCV = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            1e10,              # No. of dishes (HUGE!)
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            25.*(1e3),         # System temp. [mK]
    'survey_dnutot':    400.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     860.,              # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            15e3*(D2RAD)**2. #15e3   # Total survey area [radians^2]
    }
exptCV.update(SURVEY)


#################################
# OLD VERSIONS
"""
exptS = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            1,                 # No. of dishes
    'Nbeam':            50,                # No. of beams (for multi-pixel detectors)
    'Ddish':            30.,               # Single dish diameter [m]
    'Tinst':            50.*(1e3),         # System temp. [mK]
    'survey_dnutot':    300.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1100.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    }
exptS.update(SURVEY)

exptM = {
    'mode':             'interferom',      # Interferometer or single dish
    'Ndish':            250,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            10.,               # Single dish diameter [m]
    'Tinst':            50.*(1e3),         # System temp. [mK]
    'survey_dnutot':    400.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     800.,              # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Dmax':             100.,              # Max. interferom. baseline [m]
    'Dmin':             20.                # Min. interferom. baseline [m]
    }
exptM.update(SURVEY)

exptL = {
    'mode':             'combined',        # Interferometer or single dish
    'Ndish':            250,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    700.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1100.,             # Max. freq. of survey
    'dnu':              0.1,                # Bandwidth of single channel [MHz]
    'n(x)':             nx['SKAMREF'],     # Interferometer antenna density
    'Dmax':             100.,              # Max. interferom. baseline [m]
    'Dmin':             20.                # Min. interferom. baseline [m]
    }
exptL.update(SURVEY)
"""

################################################################################
# Configurations from Mario's notes
################################################################################

GBT = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            1,                 # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            100.,              # Single dish diameter [m]
    'Tinst':            29.*(1e3),         # System temp. [mK]
    'survey_dnutot':    240.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     920.,              # Max. freq. of survey
    'dnu':              0.1,             # Bandwidth of single channel [MHz]
    'Sarea':            1e2*(D2RAD)**2.,   # Total survey area [radians^2]
    }
GBT.update(SURVEY)

Parkes = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            1,                 # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            64.,               # Single dish diameter [m]
    'Tinst':            25.*(1e3),         # System temp. [mK]
    'survey_dnutot':    220.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1420.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            1e2*(D2RAD)**2.,   # Total survey area [radians^2]
    }
Parkes.update(SURVEY)

GMRT = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            30,                # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            45.,               # Single dish diameter [m]
    'Tinst':            70.*(1e3),         # System temp. [mK]
    'survey_dnutot':    420.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1420.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            1e2*(D2RAD)**2.,   # Total survey area [radians^2]
    }
GMRT.update(SURVEY)

# FIXME: What is the actual bandwidth of WSRT?
WSRT = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            14,                # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            25.,               # Single dish diameter [m]
    'Tinst':            120.*(1e3),        # System temp. [mK]
    'survey_dnutot':    500.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1200.,             # Max. freq. of survey
    'dnu':              0.1,                # Bandwidth of single channel [MHz]
    'Sarea':            1e3*(D2RAD)**2.,   # Total survey area [radians^2]
    }
WSRT.update(SURVEY)

APERTIF = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            14,                # No. of dishes
    'Nbeam':            37,                # No. of beams (for multi-pixel detectors)
    'Ddish':            25.,               # Single dish diameter [m]
    'Tinst':            52.*(1e3),         # System temp. [mK]
    'survey_dnutot':    300.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1300.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            5e3*(D2RAD)**2.,   # Total survey area [radians^2]
    }
APERTIF.update(SURVEY)

VLBA = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            10,                # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            25.,               # Single dish diameter [m]
    'Tinst':            27.*(1e3),         # System temp. [mK]
    'survey_dnutot':    220.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1420.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            5e3*(D2RAD)**2.,   # Total survey area [radians^2]
    }
VLBA.update(SURVEY)

JVLA = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            27,                # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            25.,               # Single dish diameter [m]
    'Tinst':            70.*(1e3),         # System temp. [mK]
    'survey_dnutot':    420.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1420.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            1e3*(D2RAD)**2.,   # Total survey area [radians^2]
    'n(x)': "array_config/nx_VLAD_dec90.dat" # Interferometer antenna density
    }
JVLA.update(SURVEY)

BINGO = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            1,                 # No. of dishes
    'Nbeam':            50,                # No. of beams (for multi-pixel detectors)
    'Ddish':            25.,               # Single dish diameter [m]
    'Tinst':            50.*(1e3),         # System temp. [mK]
    'survey_dnutot':    300.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1260.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            2e3*(D2RAD)**2.,   # Total survey area [radians^2]
    }
BINGO.update(SURVEY)

BAOBAB32 = {
    'mode':             'interferom',      # Interferometer or single dish
    'Ndish':            32,                # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            1.6,               # Single dish diameter [m]
    'Tinst':            40.*(1e3),         # System temp. [mK]
    'survey_dnutot':    300.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     900.,              # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            2e3*(D2RAD)**2.,   # Total survey area [radians^2]
    'Dmax':             13.8,              # Max. interferom. baseline [m]
    'Dmin':             1.6                # Min. interferom. baseline [m]
    }
BAOBAB32.update(SURVEY)

BAOBAB128 = {
    'mode':             'interferom',      # Interferometer or single dish
    'Ndish':            128,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            1.6,               # Single dish diameter [m]
    'Tinst':            40.*(1e3),         # System temp. [mK]
    'survey_dnutot':    300.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     900.,              # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            2e3*(D2RAD)**2.,   # Total survey area [radians^2]
    'Dmax':             26.,               # Max. interferom. baseline [m]
    'Dmin':             1.6                # Min. interferom. baseline [m]
    }
BAOBAB128.update(SURVEY)

KZN = {
    'mode':             'interferom',      # Interferometer or single dish
    'Ndish':            1225,              # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            5.0,               # Single dish diameter [m]
    'Tinst':            50.*(1e3),         # System temp. [mK]
    'survey_dnutot':    300.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     900.,              # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            2e3*(D2RAD)**2.,   # Total survey area [radians^2]
    'Dmax':             200.,              # Max. interferom. baseline [m]
    'Dmin':             5.0                # Min. interferom. baseline [m]
    }
KZN.update(SURVEY)

CHIME = {
    'mode':             'cylinder',        # Interferometer or single dish
    'Ndish':            1280,              # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            20.,               # Single dish diameter [m]
    'Tinst':            50.*(1e3),         # System temp. [mK]
    'survey_dnutot':    400.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     800.,              # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'Dmax':             128.,              # Max. interferom. baseline [m]
    'Dmin':             20.,               # Min. interferom. baseline [m]
    'n(x)': "array_config/nx_CHIME_800.dat" # Interferometer antenna density
    }
CHIME.update(SURVEY)

CHIME_nocut = {
    'mode':             'cylinder',        # Interferometer or single dish
    'Ndish':            1280,              # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            20.,               # Single dish diameter [m]
    'Tinst':            50.*(1e3),         # System temp. [mK]
    'survey_dnutot':    400.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     800.,              # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'Dmax':             128.,              # Max. interferom. baseline [m]
    'Dmin':             20.,               # Min. interferom. baseline [m]
    'n(x)': "array_config/nx_CHIME_800_nocut.dat" # Interferometer antenna density
    }
CHIME_nocut.update(SURVEY)

AERA3 = {
    'mode':             'interferom',      # Interferometer or single dish
    'Ndish':            100,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            5.,                # Single dish diameter [m]
    'Tinst':            40.*(1e3),         # System temp. [mK]
    'survey_dnutot':    500.,             # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1200.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            2e3*(D2RAD)**2.,   # Total survey area [radians^2]
    'Dmax':             80.,               # Max. interferom. baseline [m]
    'Dmin':             5.                 # Min. interferom. baseline [m]
    }
AERA3.update(SURVEY)

KAT7 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            7,                 # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            13.5,              # Single dish diameter [m]
    'Tinst':            30.*(1e3),         # System temp. [mK]
    'survey_dnutot':    220.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1420.,             # Max. freq. of survey
    'dnu':              0.1,                # Bandwidth of single channel [MHz]
    'Sarea':            2e3*(D2RAD)**2.,   # Total survey area [radians^2]
    'n(x)': "array_config/nx_KAT7_dec30.dat" # Interferometer antenna density
    }
KAT7.update(SURVEY)

MeerKATb1 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            64,                # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            13.5,              # Single dish diameter [m]
    'Tinst':            29.*(1e3),         # System temp. [mK]
    'survey_dnutot':    435.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1015.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            5e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'n(x)': "array_config/nx_MKREF2_dec30.dat" # Interferometer antenna density
    }
MeerKATb1.update(SURVEY)

MeerKATb2 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            64,                # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            13.5,              # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    520.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1420.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            5e3*(D2RAD)**2.,   # Total survey area [radians^2]
    'n(x)': "array_config/nx_MKREF2_dec30.dat" # Interferometer antenna density
    }
MeerKATb2.update(SURVEY)

ASKAP = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            36,                # No. of dishes
    'Nbeam':            36,                # No. of beams (for multi-pixel detectors)
    'Ddish':            12.,               # Single dish diameter [m]
    'Tinst':            50.*(1e3),         # System temp. [mK]
    'survey_dnutot':    300.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1000.,             # Max. freq. of survey
    'dnu':              0.1,                # Bandwidth of single channel [MHz]
    'Sarea':            10e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'n(x)': "array_config/nx_ASKAP_dec30.dat" # Interferometer antenna density
    }
ASKAP.update(SURVEY)

SKA1MIDbase1 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            190,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            28.*(1e3),         # System temp. [mK]
    'survey_dnutot':    700.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1050.,             # Max. freq. of survey
    'dnu':              0.1,                # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'n(x)': "array_config/nx_SKAM190_dec30.dat" # Interferometer antenna density
    }
SKA1MIDbase1.update(SURVEY)

SKA1MIDbase2 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            190,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    470.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1420.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'n(x)': "array_config/nx_SKAM190_dec30.dat" # Interferometer antenna density
    }
SKA1MIDbase2.update(SURVEY)

SKA1SURbase1 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            60,                # No. of dishes
    'Nbeam':            36,                # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            50.*(1e3),         # System temp. [mK]
    'survey_dnutot':    500.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     900.,              # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.   # Total survey area [radians^2]
    }
SKA1SURbase1.update(SURVEY)

SKA1SURbase2 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            60,                # No. of dishes
    'Nbeam':            36,                # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            30.*(1e3),         # System temp. [mK]
    'survey_dnutot':    500.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1150.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.   # Total survey area [radians^2]
    }
SKA1SURbase2.update(SURVEY)

SKA1MIDfull1 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            254,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish': (190.*15. + 64.*13.5)/(254.), # Single dish diameter [m]
    'Tinst':            28.*(1e3),         # System temp. [mK]
    'survey_dnutot':    435.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1015.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'n(x)': "array_config/nx_SKAMREF2_dec30.dat" # Interferometer antenna density
    }
SKA1MIDfull1.update(SURVEY)

SKA1MIDfull2 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            254,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish': (190.*15. + 64.*13.5)/(254.), # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    470.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1420.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'n(x)': "array_config/nx_SKAMREF2_dec30.dat" # Interferometer antenna density
    }
SKA1MIDfull2.update(SURVEY)

SKA1SURfull1 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            96,                # No. of dishes
    'Nbeam':            36,                # No. of beams (for multi-pixel detectors)
    'Ddish':    (60.*15. + 36.*12.)/(96.), # Single dish diameter [m]
    'Tinst':            50.*(1e3),         # System temp. [mK]
    'survey_dnutot':    500.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     900.,              # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.   # Total survey area [radians^2]
    }
SKA1SURfull1.update(SURVEY)

SKA1SURfull2 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            96,                # No. of dishes
    'Nbeam':            36,                # No. of beams (for multi-pixel detectors)
    'Ddish':    (60.*15. + 36.*12.)/(96.), # Single dish diameter [m]
    'Tinst':            30.*(1e3),         # System temp. [mK]
    'survey_dnutot':    500.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1150.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.   # Total survey area [radians^2]
    }
SKA1SURfull2.update(SURVEY)

"""
# Surveys that are defined as overlap between two instruments
SKAMID_PLUS = {
    'overlap':          [SKA1MID, MeerKAT],
    'Sarea':            25e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'n(x)':             "array_config/nx_SKAMREF2COMP_dec30.dat"
    }

SKAMID_PLUS_band1 = {
    'overlap':          [SKA1MID, MeerKAT_band1],
    'Sarea':            25e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'n(x)':             "array_config/nx_SKAMREF2COMP_dec30.dat"
    }

SKASUR_PLUS = {
    'overlap':          [SKA1SUR, ASKAP],
    'Sarea':            25e3*(D2RAD)**2.   # Total survey area [radians^2]
    }

SKASUR_PLUS_band1 = {
    'overlap':          [SKA1SUR_band1, ASKAP],
    'Sarea':            25e3*(D2RAD)**2.   # Total survey area [radians^2]
    }
"""

"""
exptO = {
    'mode':             'combined',        # Interferometer or single dish
    'Ndish':            250,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            2.5,               # Single dish diameter [m]
    'Tinst':            25.*(1e3),         # System temp. [mK]
    'survey_dnutot':    700.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1200.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            10e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'Dmax':             44.,               # Max. interferom. baseline [m]
    'Dmin':             2.5                # Min. interferom. baseline [m]
    }
exptO.update(SURVEY)

exptX = {
    'mode':             'combined',        # Interferometer or single dish
    'Ndish':            250000,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    700.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1100.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            30e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'Dmax':             600.,              # Max. interferom. baseline [m]
    'Dmin':             15.                # Min. interferom. baseline [m]
    }
exptX.update(SURVEY)

# FIXME
exptOpt = {
    'mode':             'interferom',      # Interferometer or single dish
    'Ndish':            1500,              # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            2.,                # Single dish diameter [m]
    'Tinst':            25.*(1e3),         # System temp. [mK]
    'survey_dnutot':    1070.,             # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1420.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            10e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'Dmax':             85, #125.,         # Max. interferom. baseline [m]
    'Dmin':             2.                 # Min. interferom. baseline [m]
    }
exptOpt.update(SURVEY)
"""
