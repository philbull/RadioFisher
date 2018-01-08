#!/usr/bin/python
"""
Calculate weighted CHIME baseline distribution.
"""
import numpy as np
import pylab as P
import scipy.spatial.distance

# Cylinder spec.
N_ant = 256
N_cyl = 5
w = 20.
l = 80.
nu = 400. # freq., MHz
ll = 3e8 / (nu * 1e6)

dx = w
dy = l / N_ant

# Define baseline positions
x = []; y = []
for i in range(N_cyl):
    for j in range(N_ant):
        x.append(dx*i)
        y.append(dy*j)
x = np.array(x)
y = np.array(y)

# Calculate baseline lengths
d = scipy.spatial.distance.pdist(np.array([x, y]).T)

# Re-weight baselines that don't see the whole FOV
dfov = ll / (0.5 * np.pi)
print "D_fov(N-S): %3.3f" % dfov
print "dy(N-S):    %3.3f" % dy
d1 = d.copy()
d1[np.where(d1 < dfov)]


# Plot baseline distribution
P.hist(d, bins=115, range=(0., 115.), ec='none', alpha=0.5)

# Plot receiver distribution
#P.plot(x, y, 'r.')
P.show()


