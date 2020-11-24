#!/usr/bin/python
"""
Calculate binned, circularly-averaged baseline distribution for a regular 
rectangular array of antennas.
"""
import numpy as np
import pylab as P
import scipy.integrate
import scipy.spatial

# Array specification and reference frequency
root = "CV256x256"
Ddish = 6.
Dmin = 6.
Nx = 128 # FIXME: 256x256 causes memory error
Ny = 128
nu = 400. # MHz
l = 3e8 / (nu * 1e6) # Lambda [m]
outfile = "array_config/nx_%s.dat" % root

def antenna_positions():
    """
    Generate antenna positions for a regular rectangular array, then return 
    baseline lengths.
     - Nx, Ny : No. of antennas in x and y directions
     - Dmin : Separation between neighbouring antennas
    """
    # Generate antenna positions on a regular grid
    x = np.arange(Nx) * Dmin
    y = np.arange(Ny) * Dmin
    xx, yy = np.meshgrid(x, y)

    # Calculate baseline separations
    xy = np.column_stack( (xx.flatten(), yy.flatten()) )
    d = scipy.spatial.distance.pdist(xy)
    return d

def fov(nu, D):
    """
    Field of view, in rad^2, as a fn. of frequency.
    """
    l = 3e8 / (nu*1e6)
    return 180. * 1.22 * (l/D) * (np.pi/180.)**2.

def ubin_width(nu, D):
    """
    Bin width, corresponding to du at a given frequency (u = d / lambda).
    """
    return 1. / np.sqrt(fov(nu, D))

def binned_baseline_dist(d, check_norm=True):
    """
    Calculate binned n(u), the baseline distribution in bins of constant 
    Delta u.
    """
    d = d[np.where(d > Ddish)] # Cut sub-FOV baselines
    d /= l # Rescale into u = d / lambda

    # Calculate bin edges
    du = ubin_width(nu, Ddish)
    imax = int(np.max(d) / du) + 1
    edges = np.linspace(0., imax * du, imax+1)

    # Calculate histogram (no. baselines in each ring of width du) 
    # and u bin centroids
    bins, edges = np.histogram(d, edges)
    u = np.array([0.5*(edges[i+1] + edges[i]) for i in range(edges.size-1)])

    # Convert to a density, n(u)
    nn = bins / (2. * np.pi * u * du)

    # Integrate n(u) to find normalisation (should give 1 if no baseline cuts)
    if check_norm:
        norm = scipy.integrate.simps(2.*np.pi*nn*u, u)
        Ndish = Nx * Ny
        print "n(u) renorm. factor: %6.6f (not applied)" \
                % (0.5 * Ndish * (Ndish - 1) / norm)
        #n *= 0.5 * Ndish * (Ndish - 1) / norm
    
    return u, nn


if __name__ == '__main__':
    
    # Get list of baselines
    d = antenna_positions()
    
    # Calculate binned baseline density
    u, n_u = binned_baseline_dist(d)
    #n_u *= 4.
    
    # Convert n(u) to freq.-independent expression, n(x) = n(u) * nu^2 
    # (nu in MHz), then save to file
    n_x = n_u * nu**2.
    x = u / nu
    np.savetxt(outfile, np.column_stack((x, n_x)))
    print "Saved to %s." % outfile

    # Plot histogram
    P.subplot(111)
    P.step(u, n_u, where='mid', lw=1.8, color='b')

    P.axvline(Dmin/l, color='r', lw=1.8)
    
    P.xlabel("$u$", fontsize=18.)
    P.ylabel("$n(u)$", fontsize=18.)
    
    P.xlim((0., 1200.))
    
    P.tight_layout()
    P.show()
    
