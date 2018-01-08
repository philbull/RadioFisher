#!/usr/bin/python
"""
Test binning strategy
"""
import numpy as np
import pylab as P
import radiofisher as rf

expt = rf.experiments.SKA1MID900

def zbins_fixed(expt, zbin_min=0., zbin_max=6., dz=0.1):
    """
    Construct a sensible binning for a given experiment for bins with fixed dz, 
    equally spaced from z=zbin_min. If the band does not exactly divide into 
    the binning, smaller bins will be added at the end of the band.
    """
    zs = np.arange(zbin_min, zbin_max, dz)
    zc = (zs + 0.5*dz)[:-1]
    
    # Get redshift ranges of actual experiment
    zmin = expt['nu_line'] / expt['survey_numax'] - 1.
    zmax = expt['nu_line'] / (expt['survey_numax'] - expt['survey_dnutot']) - 1.
    
    # Remove bins with no overlap
    idxs = np.where(np.logical_and(zs >= zmin, zs <= zmax))
    zs = zs[idxs]
    
    # Add end bins to cover as much of the full band as possible
    if (zs[0] - zmin) > 0.1*dz:
        zs = np.concatenate(([zmin,], zs))
    if (zmax - zs[-1]) > 0.1*dz:
        zs = np.concatenate((zs, [zmax,]))
    
    # Return bin edges and centroids
    zc = np.array([0.5*(zs[i+1] + zs[i]) for i in range(zs.size - 1)])
    return zs, zc


def zbins_split_width(expt, dz=(0.1, 0.3), zsplit=2.):
    """
    Construct a binning scheme with bin widths that change after a certain 
    redshift. The first redshift range is filled with equal-sized bins that may 
    go over the split redshift. The remaining range is filled with bins of the 
    other width (apart from the last bin, that may be truncated).
    
    Parameters
    ----------
    expt : dict
        Dict of experimental settings.
        
    dz : tuple (length 2)
        Widths of redshift bins before and after the split redshift.
    
    zsplit : float
        Redshift at which to change from the first bin width to the second.
    """
    
    # Get redshift ranges of actual experiment
    zmin = expt['nu_line'] / expt['survey_numax'] - 1.
    zmax = expt['nu_line'] / (expt['survey_numax'] - expt['survey_dnutot']) - 1.
    
    # Sanity checks
    assert zmax > zsplit, "Split redshift must be less than max. redshift of experiment."
    
    # Fill first range with equal-sized bins with width dz[0]
    nbins = np.ceil((zsplit - zmin) / dz[0])
    z1 = np.linspace(zmin, zmin + nbins*dz[0], nbins+1)
    
    # Fill remaining range with equal-sized bins with width dz[1]
    nbins = np.floor((zmax - z1[-1]) / dz[1])
    z2 = np.linspace(z1[-1] + dz[1], z1[-1] + nbins*dz[1], nbins)
    
    # Add final bin to fill range only if >20% of dz[1]
    if (zmax - z2[-1]) > 0.2 * dz[1]:
        z2 = np.concatenate((z2, [zmax,]))
    
    # Concatenate full range and return
    zs = np.concatenate((z1, z2))
    zc = np.array([0.5*(zs[i+1] + zs[i]) for i in range(zs.size - 1)])
    return zs, zc
    
    

zs, zc = rf.zbins_split_width(expt, dz=(0.1, 0.4), zsplit=2.)
print zs
