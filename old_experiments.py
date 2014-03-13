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
