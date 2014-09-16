#!/usr/bin/python
"""
Generate a simulated IM field from a random distribution of galaxies.
"""
import numpy as np
import pylab as P
import scipy.ndimage

np.random.seed(10)
Ngal = 400

# Galaxy field
x = np.random.rand(Ngal)
y = np.random.rand(Ngal)

# Histogram
hist, Xe, Ye = np.histogram2d(x, y, bins=30, range=[[0., 1.], [0., 1.]])
xc = [0.5*(Xe[i] + Xe[i+1])for i in range(Xe.size-1)]
yc = [0.5*(Ye[i] + Ye[i+1])for i in range(Ye.size-1)]

# Resample and convolve
hh = scipy.ndimage.interpolation.zoom(hist, 6)
hh = scipy.ndimage.filters.gaussian_filter(hh, 7.)

P.subplot(121)
P.plot(x, y, 'k.')
P.xlim((0., 1.))
P.ylim((0., 1.))
P.tick_params(axis='both', which='major', labelleft='off', labelbottom='off', size=0.)


P.subplot(122)
P.imshow(hh.T, extent=(0., 1., 1., 0.), origin='lower')
P.tick_params(axis='both', which='major', labelleft='off', labelbottom='off', size=0.)

P.gcf().set_size_inches(10.,5.)
P.tight_layout()
P.savefig('im-field.pdf', transparent=True)
P.savefig('im-field.png', transparent=True)
P.show()
