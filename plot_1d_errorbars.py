#!/usr/bin/python
"""
Plot 1D constraints as a series of errorbars
"""

import numpy as np
import pylab as P

colours = ['#CC0000', '#ED5F21', '#FAE300', '#5B9C0A', '#1619A1', '#56129F', '#990A9C']
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

err = np.sqrt(np.random.randn(7) + 2.) # FIXME

x0 = 1. # Fiducial value

x = x0 * np.ones(err.size)
y = np.arange(err.size)

P.subplot(111)
for i in range(err.size):
    P.errorbar(x0, i, xerr=err[i], color=colours[i], lw=2., marker='.', markersize=10.)
    P.annotate(labels[i], xy=(x0, i), xytext=(3., 10.), textcoords='offset points', ha='right', va='bottom')

P.ylim((-1., err.size))
P.xlim((-2., 4.,))
P.show()
