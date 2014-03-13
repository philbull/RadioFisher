#!/usr/bin/python
"""
Test Fisher matrices
"""
import numpy as np
import pylab as P
import euclid
from experiments import cosmo

Fc = np.genfromtxt("fisher_planck_camb.dat")
Fe = euclid.planck_prior_full

F_camb = euclid.camb_to_baofisher(Fc, cosmo)
F_detf = euclid.detf_to_baofisher("DETF_PLANCK_FISHER.txt", cosmo)
F_eucl = euclid.euclid_to_baofisher(Fe, cosmo)

y1 = F_detf / F_camb
y2 = F_eucl / F_camb

lbl = ['n_s', 'w0', 'wa', 'ob', 'ok', 'oDE', 'h']

for i in range(7):
    for j in range(i, 7):
        print "%d %d: %+4.2e %+4.2e -- %3s . %3s" % (i, j, y1[i,j], y2[i,j], lbl[i], lbl[j])
