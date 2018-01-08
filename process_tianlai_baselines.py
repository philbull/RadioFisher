#!/usr/bin/python
"""
Output the Tianlai baseline density distribution.
"""
import numpy as np
import pylab as P
import scipy.integrate

nu = 1000. # MHz
l = 3e8 / (nu * 1e6) # Lambda [m]

root = "TIANLAI"
Ddish = 15.
Dmin = 15.
Ndish = 256 * 8
array_config = "array_config/TIANLAI_baselines.npy"
outfile = "array_config/nx_TIANLAI_%d.dat" % nu

root = "TIANLAIpathfinder"
Ddish = 15.
Dmin = 15.
Ndish = 32 * 3
array_config = "array_config/TIANLAIpath_baselines.npy"
outfile = "array_config/nx_TIANLAIpath_%d.dat" % nu

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
    return (1./10.) / np.sqrt(fov(nu, D)) # 1/10!

dat = np.load(array_config).T

# Remove D < Ddish baselines
dat = dat[np.where(dat > Ddish)] # Cut sub-FOV baselines
dat /= l # Rescale into u = d / lambda

# Calculate bin edges
du = ubin_width(nu, Ddish)
imax = int(np.max(dat) / du) + 1
edges = np.linspace(0., imax * du, imax+1)

# Calculate histogram (no. baselines in each ring of width du)
bins, edges = np.histogram(dat, edges)
u = np.array([0.5*(edges[i+1] + edges[i]) for i in range(edges.size-1)]) # Centroids
#idxs = np.where(u < Dmin/l)

#for i in range(bins.size):
#    print "%2d [%3.1f -- %3.1f]: %d" % (i, edges[i], edges[i+1], bins[i])

# Convert to a density, n(u)
nn = bins / (2. * np.pi * u * du)

# Integrate n(u) to find normalisation (should give unity if no baseline cuts applied)
norm = scipy.integrate.simps(2.*np.pi*nn*u, u)
print "n(u) renorm. factor:", 0.5 * Ndish * (Ndish - 1) / norm, "(not applied)"
#n *= 0.5 * Ndish * (Ndish - 1) / norm

# Convert to freq.-independent expression, n(x) = n(u) * nu^2,
# where nu is in MHz.
n_x = nn * nu**2.
x = u / nu
np.savetxt(outfile, np.column_stack((x, n_x)))
print "Saved to %s." % outfile


P.plot(u, nn)
P.axvline(Dmin/l, color='r')
P.show()
