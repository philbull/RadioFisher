#!/usr/bin/python
"""
Plot the CHIME baseline distribution from the raw baseline file sent by Prina.
"""
import numpy as np
import pylab as P

root = "CHIME256"
#dd = 0.31 #0.31 # 31 cm
Ddish = 20.
Dmin = 20.
nu = 800. # MHz

l = 3e8 / (nu * 1e6) # Lambda [m]

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
    return (1./3.) / np.sqrt(fov(nu, D)) # 1/30!

dat = np.genfromtxt("array_config/CHIME256_baselines.txt").T
dat /= l

# Calculate bin edges
du = ubin_width(nu, Ddish)
imax = int(np.max(dat) / du) + 1
edges = np.linspace(0., imax * du, imax+1)

# Calculate histogram
bins, edges = np.histogram(dat, edges)

# Put sub-Dmin baselines into first >Dmin bin (according to where the centroid is)
centroids = np.array([0.5*(edges[i+1] + edges[i]) for i in range(edges.size-1)])
idxs = np.where(centroids < Dmin/l)
bins[idxs[0][-1]] += np.sum(bins[idxs])
bins[idxs] = 0.

for i in range(bins.size):
    print "%2d [%3.1f -- %3.1f]: %d" % (i, edges[i], edges[i+1], bins[i])

# Output in usual format (min. bin from Dmin)
umin = edges[idxs[0][-1]+1]

print "du:", du
print "umin:", umin

fname = "%s_%3.2fe9_dec00_60sec.MS_bin%7.4f_du%7.5f.txt" % (root, nu/1e3, umin, du)
np.savetxt( fname, np.column_stack((edges[idxs[0][-1]+1:-1], bins[idxs[0][-1]+1:])) )
print "Saved to: %s" % fname

exit()
# Plot results
P.hist(dat, edges, edgecolor='none', log=True, alpha=0.3)
P.show()
