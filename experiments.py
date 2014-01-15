import numpy as np
import scipy.interpolate
from units import *

# Define fiducial cosmology and parameters
cosmo = {
    'omega_M_0':        0.26,
    'omega_lambda_0':   0.74,
    'omega_b_0':        0.045,
    'omega_HI_0':       9.4e-4,
    'omega_n_0':        0.0,
    'omega_k_0':        0.0,
    'N_nu':             0,
    'h':                0.70,
    'n':                0.96,
    'sigma_8':          0.8,
    'gamma':            0.55,
    'w0':               -1.,
    'wa':               0.,
    'fNL':              0.,
    'mnu':              0.,
    'aperp':            1.,
    'apar':             1.,
    'bHI0':             0.702,
    'A':                1.,
    'sigma_nl':         7.
}

# Define which measurements to include in forecasts
USE = {
  'f_rsd':             True,     # RSD constraint on f(z)
  'f_growthfactor':    False,    # D(z) constraint on f(z)
  'alpha_all':         False,     # Use all constraints on alpha_{perp,par}
  'alpha_volume':      False,
  'alpha_rsd_angle':   False, #t
  'alpha_rsd_shift':   False, #t
  'alpha_bao_shift':   True,
  'alpha_pk_shift':    False # True
}

SURVEY = {
    'ttot':             10e3*HRS_MHZ,      # Total integration time [MHz^-1]
    'Sarea':            30e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'nu_line':          1420.406,          # Rest-frame freq. of emission line [MHz]
    'epsilon_fg':       1e-8,              # FG subtraction residual amplitude
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
# Interferometer n(u) distributions
################################################################################

# Define available n(x) files (all for dec=90deg for now)
nx_root = "array_config/"
nx_files = [ 
  "nx_KAT7_dec90.dat", 
  "nx_SKAM190_dec90_smalldu.dat",      "nx_SKAM190_dec90_bigdu.dat", 
  "nx_SKAMREF2_dec90_smalldu.dat",     "nx_SKAMREF2_dec90_bigdu.dat", 
  "nx_MKREF2_dec90_smalldu.dat",       "nx_MKREF2_dec90_bigdu.dat", 
  "nx_SKAMREF2COMP_dec90_smalldu.dat", "nx_SKAMREF2COMP_dec90_bigdu.dat",
  "nx_SKAMREF2COMP_dec90_minu.dat",    "nx_SKAMREF2_dec90_minu.dat"
]
nx_names = ["KAT7", "SKA1", "SKA1_bigdu", "SKAMID", "SKAMID_bigdu", "MK", "MK_bigdu", "SKAMID_compact", "SKAMID_compact_bigdu", "SKAMID_compact_minu", "SKAMID_minu"]

# Load n(x) interpolation functions (see rescale_baseline_density.py)
# and pack into a dictionary
nx = {}
for i in range(len(nx_files)):
    x, _nx = np.genfromtxt(nx_root + nx_files[i]).T
    interp_nx = scipy.interpolate.interp1d( x, _nx, kind='linear', 
                                            bounds_error=False, fill_value=0. )
    nx[nx_names[i]] = interp_nx


################################################################################
# Misc. experiments (OBSOLETE)
################################################################################

# Define experimental setups
ska1_singledish_800 = {
    'interferometer':   False,                # Interferometer or single dish
    'Ndish':            250,                  # No. of dishes
    'Nbeam':            1,                    # No. of beams (for multi-pixel detectors)
    'Tinst':            20*(1e3),             # System temp. [mK]
    'ttot':             5e3*HRS_MHZ,          # Total integration time [MHz^-1]
    'Sarea':            3e4*(D2RAD)**2.,      # Total survey area [radians^2]
    'dnu':              0.005,                # Bandwidth of single channel [MHz]
    'survey_dnutot':    800.,                 # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1380.,                # Max. freq. of survey
    'Ddish':            15.,                  # Single dish diameter [m]
    'nu_line':          1420.406,             # Rest-frame freq. of emission line [MHz]
    'epsilon_fg':       1e-6                  # FG subtraction residual amplitude
    }

ska1_singledish_800_small = {
    'interferometer':   False,                # Interferometer or single dish
    'Ndish':            250,                  # No. of dishes
    'Nbeam':            1,                    # No. of beams (for multi-pixel detectors)
    'Tinst':            20*(1e3),             # System temp. [mK]
    'ttot':             5e3*HRS_MHZ,          # Total integration time [MHz^-1]
    'Sarea':            5e3*(D2RAD)**2.,      # Total survey area [radians^2]
    'dnu':              0.005,                # Bandwidth of single channel [MHz]
    'survey_dnutot':    800.,                 # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1380.,                # Max. freq. of survey
    'Ddish':            15.,                  # Single dish diameter [m]
    'nu_line':          1420.406,             # Rest-frame freq. of emission line [MHz]
    'epsilon_fg':       1e-4                  # FG subtraction residual amplitude
    }

ska1_singledish_1000 = {
    'interferometer':   False,                # Interferometer or single dish
    'Ndish':            250,                  # No. of dishes
    'Nbeam':            1,                    # No. of beams (for multi-pixel detectors)
    'Tinst':            20*(1e3),             # System temp. [mK]
    'ttot':             5e3*HRS_MHZ,          # Total integration time [MHz^-1]
    'Sarea':            3e4*(D2RAD)**2.,      # Total survey area [radians^2]
    'dnu':              0.005,                # Bandwidth of single channel [MHz]
    'survey_dnutot':    1000.,                # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1380.,                # Max. freq. of survey
    'Ddish':            15.,                  # Single dish diameter [m]
    'nu_line':          1420.406,             # Rest-frame freq. of emission line [MHz]
    'epsilon_fg':       1e-4                  # FG subtraction residual amplitude
    }

meerkat_singledish_800 = {
    'interferometer':   False,                # Interferometer or single dish
    'Tinst':            20*(1e3),             # System temp. [mK]
    'Ndish':            64,                   # No. of dishes
    'Nbeam':            1,                    # No. of beams (for multi-pixel detectors)
    'ttot':             5e3*HRS_MHZ,          # Total integration time [MHz^-1]
    'Sarea':            3e4*(D2RAD)**2.,      # Total survey area [radians^2]
    'dnu':              0.005,                # Bandwidth of single channel [MHz]
    'survey_dnutot':    800.,                 # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1380.,                # Max. freq. of survey
    'Ddish':            13.5,                 # Single dish diameter [m]
    'nu_line':          1420.406,             # Rest-frame freq. of emission line [MHz]
    'epsilon_fg':       1e-4                  # FG subtraction residual amplitude
    }


meerkat_singledish_500 = {
    'interferometer':   False,                # Interferometer or single dish
    'Ndish':            64,                   # No. of dishes
    'Nbeam':            1,                    # No. of beams (for multi-pixel detectors)
    'Tinst':            30*(1e3),             # System temp. [mK]
    'ttot':             10e3*HRS_MHZ,         # Total integration time [MHz^-1]
    'Sarea':            3e4*(D2RAD)**2.,      # Total survey area [radians^2]
    'dnu':              0.005,                # Bandwidth of single channel [MHz]
    'survey_dnutot':    500.,                 # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1080.,                # Max. freq. of survey
    'Ddish':            13.5,                 # Single dish diameter [m]
    'nu_line':          1420.406,             # Rest-frame freq. of emission line [MHz]
    'epsilon_fg':       1e-4                  # FG subtraction residual amplitude
    }


################################################################################
# Experiments used in paper
################################################################################

# SKA 1 spec.
exptA = {
    'interferometer':   False,                # Interferometer or single dish
    'Ndish':            250,                  # No. of dishes
    'Nbeam':            1,                    # No. of beams (for multi-pixel detectors)
    'Tinst':            20*(1e3),             # System temp. [mK]
    'ttot':             10e3*HRS_MHZ,          # Total integration time [MHz^-1]
    'Sarea':            30e3*(D2RAD)**2.,     # Total survey area [radians^2]
    'dnu':              0.005,                # Bandwidth of single channel [MHz]
    'survey_dnutot':    800.,                 # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1380.,                # Max. freq. of survey
    'Ddish':            15.,                  # Single dish diameter [m]
    'nu_line':          1420.406,             # Rest-frame freq. of emission line [MHz]
    'epsilon_fg':       1e-8                  # FG subtraction residual amplitude
    }

# Same as A, but with 5000 deg^2 area
exptB = {
    'interferometer':   False,                # Interferometer or single dish
    'Ndish':            250,                  # No. of dishes
    'Nbeam':            1,                    # No. of beams (for multi-pixel detectors)
    'Tinst':            20*(1e3),             # System temp. [mK]
    'ttot':             5e3*HRS_MHZ,          # Total integration time [MHz^-1]
    'Sarea':            5e3*(D2RAD)**2.,      # Total survey area [radians^2]
    'dnu':              0.005,                # Bandwidth of single channel [MHz]
    'survey_dnutot':    800.,                 # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1380.,                # Max. freq. of survey
    'Ddish':            15.,                  # Single dish diameter [m]
    'nu_line':          1420.406,             # Rest-frame freq. of emission line [MHz]
    'epsilon_fg':       1e-8                  # FG subtraction residual amplitude
    }

# Same as A, but with larger bandwidth
exptC = {
    'interferometer':   False,                # Interferometer or single dish
    'Ndish':            250,                  # No. of dishes
    'Nbeam':            1,                    # No. of beams (for multi-pixel detectors)
    'Tinst':            20*(1e3),             # System temp. [mK]
    'ttot':             5e3*HRS_MHZ,          # Total integration time [MHz^-1]
    'Sarea':            30e3*(D2RAD)**2.,     # Total survey area [radians^2]
    'dnu':              0.005,                # Bandwidth of single channel [MHz]
    'survey_dnutot':    1000.,                # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1380.,                # Max. freq. of survey
    'Ddish':            15.,                  # Single dish diameter [m]
    'nu_line':          1420.406,             # Rest-frame freq. of emission line [MHz]
    'epsilon_fg':       1e-8                  # FG subtraction residual amplitude
    }

# MeerKAT spec.
exptD = {
    'interferometer':   False,                # Interferometer or single dish
    'Ndish':            64,                   # No. of dishes
    'Nbeam':            1,                    # No. of beams (for multi-pixel detectors)
    'Tinst':            20*(1e3),             # System temp. [mK]
    'ttot':             5e3*HRS_MHZ,          # Total integration time [MHz^-1]
    'Sarea':            30e3*(D2RAD)**2.,     # Total survey area [radians^2]
    'dnu':              0.005,                # Bandwidth of single channel [MHz]
    'survey_dnutot':    800.,                 # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1380.,                # Max. freq. of survey
    'Ddish':            13.5,                 # Single dish diameter [m]
    'nu_line':          1420.406,             # Rest-frame freq. of emission line [MHz]
    'epsilon_fg':       1e-8                  # FG subtraction residual amplitude
    }

# Something else. ASKAP?
exptE = {
    'interferometer':   False,                # Interferometer or single dish
    'Ndish':            64,                   # No. of dishes
    'Nbeam':            1,                    # No. of beams (for multi-pixel detectors)
    'Tinst':            30*(1e3),             # System temp. [mK]
    'ttot':             10e3*HRS_MHZ,         # Total integration time [MHz^-1]
    'Sarea':            30e3*(D2RAD)**2.,     # Total survey area [radians^2]
    'dnu':              0.005,                # Bandwidth of single channel [MHz]
    'survey_dnutot':    800.,                 # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1080.,                # Max. freq. of survey
    'Ddish':            13.5,                 # Single dish diameter [m]
    'nu_line':          1420.406,             # Rest-frame freq. of emission line [MHz]
    'epsilon_fg':       1e-8                  # FG subtraction residual amplitude
    }



################################################################################
# Configurations from Mario's notes
################################################################################

GBT = {
    'interferometer':   False,             # Interferometer or single dish
    'Ndish':            1,                 # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            100.,              # Single dish diameter [m]
    'Tinst':            29.*(1e3),         # System temp. [mK]
    'survey_dnutot':    240.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     920.,              # Max. freq. of survey
    'dnu':              0.005,             # Bandwidth of single channel [MHz]
    }
GBT.update(SURVEY)

BINGO = {
    'interferometer':   False,             # Interferometer or single dish
    'Ndish':            1,                 # No. of dishes
    'Nbeam':            50,                # No. of beams (for multi-pixel detectors)
    'Ddish':            30.,               # Single dish diameter [m]
    'Tinst':            50.*(1e3),         # System temp. [mK]
    'survey_dnutot':    300.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1260.,             # Max. freq. of survey
    'dnu':              0.005,             # Bandwidth of single channel [MHz]
    }
BINGO.update(SURVEY)

# FIXME: What is the actual bandwidth of WSRT?
WSRT = {
    'interferometer':   False,             # Interferometer or single dish
    'Ndish':            14,                # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            25.,               # Single dish diameter [m]
    'Tinst':            120.*(1e3),        # System temp. [mK]
    'survey_dnutot':    500.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1200.,             # Max. freq. of survey
    'dnu':              0.005,             # Bandwidth of single channel [MHz]
    }
WSRT.update(SURVEY)

# FIXME: Does this mean 37 beams per dish!?
APERTIF = {
    'interferometer':   False,             # Interferometer or single dish
    'Ndish':            14,                # No. of dishes
    'Nbeam':            37,                # No. of beams (for multi-pixel detectors)
    'Ddish':            25.,               # Single dish diameter [m]
    'Tinst':            52.*(1e3),         # System temp. [mK]
    'survey_dnutot':    300.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1300.,             # Max. freq. of survey
    'dnu':              0.005,             # Bandwidth of single channel [MHz]
    }
APERTIF.update(SURVEY)

# FIXME: Max. freq. was actually quoted as 1700 MHz!
JVLA = {
    'interferometer':   False,             # Interferometer or single dish
    'Ndish':            27,                # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            25.,               # Single dish diameter [m]
    'Tinst':            70.*(1e3),         # System temp. [mK]
    'survey_dnutot':    420.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1420.,             # Max. freq. of survey
    'dnu':              0.005,             # Bandwidth of single channel [MHz]
    }
JVLA.update(SURVEY)

ASKAP = {
    'interferometer':   False,             # Interferometer or single dish
    'Ndish':            36,                # No. of dishes
    'Nbeam':            30,                # No. of beams (for multi-pixel detectors)
    'Ddish':            12.,               # Single dish diameter [m]
    'Tinst':            50.*(1e3),         # System temp. [mK]
    'survey_dnutot':    300.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1000.,             # Max. freq. of survey
    'dnu':              0.005,             # Bandwidth of single channel [MHz]
    }
ASKAP.update(SURVEY)

KAT7 = {
    'interferometer':   False,             # Interferometer or single dish
    'Ndish':            7,                 # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            13.5,              # Single dish diameter [m]
    'Tinst':            30.*(1e3),         # System temp. [mK]
    'survey_dnutot':    220.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1420.,             # Max. freq. of survey
    'dnu':              0.005,             # Bandwidth of single channel [MHz]
    'n(x)':             nx['KAT7']         # Interferometer antenna density
    }
KAT7.update(SURVEY)

# NB: For MeerKAT Band 1 only.
MeerKAT = {
    'interferometer':   False,             # Interferometer or single dish
    'Ndish':            64,                # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            13.5,              # Single dish diameter [m]
    'Tinst':            30.*(1e3),         # System temp. [mK]
    'survey_dnutot':    650., #435.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1015.,             # Max. freq. of survey
    'dnu':              0.005,             # Bandwidth of single channel [MHz]
    'n(x)':             nx['MK']           # Interferometer antenna density
    }
MeerKAT.update(SURVEY)

SKA1 = {
    'interferometer':   False,             # Interferometer or single dish
    'Ndish':            190,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    800.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1150.,             # Max. freq. of survey
    'dnu':              0.005,             # Bandwidth of single channel [MHz]
    'n(x)':             nx['SKA1']         # Interferometer antenna density
    }
SKA1.update(SURVEY)

SKAMID = {
    'interferometer':   False,             # Interferometer or single dish
    'Ndish':            254,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':     (15.*190.+13.5*64.)/254., # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    435.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1015.,             # Max. freq. of survey
    'dnu':              0.005,             # Bandwidth of single channel [MHz]
    'n(x)':             nx['SKAMID']       # Interferometer antenna density
    }
SKAMID.update(SURVEY)

# Compact interferom. config
SKAMID_COMP = {
    'interferometer':   False,             # Interferometer or single dish
    'Ndish':            254,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':     (15.*190.+13.5*64.)/254., # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    435.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1015.,             # Max. freq. of survey
    'dnu':              0.005,             # Bandwidth of single channel [MHz]
    'n(x)':         nx['SKAMID_compact']   # Interferometer antenna density
    }
SKAMID_COMP.update(SURVEY)


# Big-z
SKAMID_COMP_BIGZ = {
    'interferometer':   False,             # Interferometer or single dish
    'Ndish':            254,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':     (15.*190.+13.5*64.)/254., # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    700.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1015.,             # Max. freq. of survey
    'dnu':              0.005,             # Bandwidth of single channel [MHz]
    'n(x)':    nx['SKAMID_compact_minu']   # Interferometer antenna density
    }
SKAMID_COMP_BIGZ.update(SURVEY)


# Big-z
SKAMID_BIGZ = {
    'interferometer':   False,             # Interferometer or single dish
    'Ndish':            254,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':     (15.*190.+13.5*64.)/254., # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    700.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1015.,             # Max. freq. of survey
    'dnu':              0.005,             # Bandwidth of single channel [MHz]
    'n(x)':    nx['SKAMID_minu']   # Interferometer antenna density
    }
SKAMID_BIGZ.update(SURVEY)

SKA_CORE = {
    'interferometer':   False,             # Interferometer or single dish
    'Ndish':            12,                # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    435.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1015.,             # Max. freq. of survey
    'dnu':              0.005,             # Bandwidth of single channel [MHz]
    'n(x)':             None        # Interferometer antenna density
    }
SKA_CORE.update(SURVEY)


"""
ska1_singledish_800 = {
    'interferometer':   False,                # Interferometer or single dish
    'Ndish':            250,                  # No. of dishes
    'Nbeam':            1,                    # No. of beams (for multi-pixel detectors)
    'Tinst':            20*(1e3),             # System temp. [mK]
    'ttot':             5e3*HRS_MHZ,          # Total integration time [MHz^-1]
    'Sarea':            3e4*(D2RAD)**2.,      # Total survey area [radians^2]
    'dnu':              0.005,                # Bandwidth of single channel [MHz]
    'survey_dnutot':    800.,                 # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1380.,                # Max. freq. of survey
    'Ddish':            15.,                  # Single dish diameter [m]
    'nu_line':          1420.406,             # Rest-frame freq. of emission line [MHz]
    'Dmax':             1.,                   # Max. interferom. baseline length [m]
    'N_ant':            100,                  # No. of interferometer antennas
    'fov':              2e4*(D2RAD)**2.,      # Interferom. field of view [radians]
    'epsilon_fg':       1e-4                  # FG subtraction residual amplitude
    }
"""
