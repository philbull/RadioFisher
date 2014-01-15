#!/usr/bin/python

import numpy as np
import pylab as P
import baofisher
import euclid

F = np.array([
[0.172276e6, 0.490320e5, 0.674392e6, -0.208974e7, 0.325219e7, -0.790504e7, -0.549427e5],
[0.490320e5, 0.139551e5, 0.191940e6, -0.594767e6, 0.925615e6, -0.224987e7, -0.156374e5],
[0.674392e6, 0.191940e6, 0.263997e7, -0.818048e7, 0.127310e8, -0.309450e8, -0.215078e6],
[-0.208974e7, -0.594767e6, -0.818048e7, 0.253489e8, -0.394501e8, 0.958892e8, 0.666335e6],
[0.325219e7, 0.925615e6, 0.127310e8, -0.394501e8, 0.633564e8, -0.147973e9, -0.501247e6],
[-0.790504e7, -0.224987e7, -0.309450e8, 0.958892e8, -0.147973e9, 0.405079e9, 0.219009e7],
[-0.549427e5, -0.156374e5, -0.215078e6, 0.666335e6, -0.501247e6, 0.219009e7, 0.242767e6] ]).T

Fpl = euclid.planck_prior
Fpl[np.diag_indices(Fpl.shape[0])] *= 1.0001

Fpl[0,:] = 0.
Fpl[:,0] = 0.
Fpl[0,0] = 1.

print "Cond. num:", np.linalg.cond(Fpl)
print "Cond. num:", np.linalg.cond(F)

cov = np.linalg.inv(Fpl)
print cov

baofisher.plot_corrmat(cov, [])
