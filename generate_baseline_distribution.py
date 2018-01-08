#!/usr/bin/python
"""
Generate a baseline distribution and output binned baseline distances.
"""
import numpy as np
import pylab as P
import scipy.spatial

#outfile = "array_config/TIANLAI_baselines"
outfile = "array_config/TIANLAIpath_baselines"

# Cylinder telescope (TIANLAI)
Ncyl = 8      # No. cylinders
w_cyl = 15.   # Cylinder width, in m
l_cyl = 120.  # Cylinder length, in m
Nfeeds = 256  # per cylinder

# Cylinder telescope (TIANLAI Pathfinder)
Ncyl = 3      # No. cylinders
w_cyl = 15.   # Cylinder width, in m
l_cyl = 16.   # Cylinder length, in m
Nfeeds = 32   # per cylinder

dx = l_cyl / float(Nfeeds)

# Layout receivers
x = []; y = []
for i in range(Ncyl):
    for j in range(Nfeeds):
        xx = i * w_cyl
        yy = j * dx
        #P.plot(xx, yy, 'bx')
        x.append(xx); y.append(yy)

# Calculate baseline separations and save to file
d = scipy.spatial.distance.pdist( np.column_stack((x, y)) )
np.save(outfile, d)

# Output stats
print "Cylinders:    %d" % Ncyl
print "Feeds/cyl.:   %d" % Nfeeds
print "Tot. feeds:   %d" % (Ncyl * Nfeeds)
print "Cycl. width:  %3.2f m" % w_cyl
print "Cycl. length: %3.2f m" % l_cyl
print "Feed sep.:    %3.3f m" % dx
print "Max baseline: %3.3f m" % np.max(d)
print "-"*50
print "Output file:  %s.npy" % outfile

#P.hist(d, bins=200)
#P.show()

