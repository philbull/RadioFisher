#!/usr/bin/python
"""
Plot SKA antenna pattern.
"""
import numpy as np
import pylab as P

# Load antenna data (sent by Prina, 2013-12-11)
x, y = np.genfromtxt("interferom/SKA1_ANTENNAS_REF2_xy.txt").T

P.subplot(221)
P.plot(x, y, 'r.')
P.Rectangle([0., 0.], 10000., 10000.) #(0., 1., 0., 1.)

P.show()
