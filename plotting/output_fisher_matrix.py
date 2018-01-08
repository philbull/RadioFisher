#!/usr/bin/python
"""
Output Fisher matrix for experiments to a text file.
"""
import numpy as np
import pylab as P
from rfwrapper import rf

cosmo = rf.experiments.cosmo

fmroot = "fishermat"
names = ['EuclidRef_baoonly', ]

# Loop through experiments and output them
_k = range(len(names))
for k in _k:
    root = "output/" + names[k]

    # Load cosmo fns.
    zc, Hc, dAc, Dc, fc = np.atleast_2d( np.genfromtxt(root+"-cosmofns-zc.dat") ).T
    zs, Hs, dAs, Ds, fs = np.genfromtxt(root+"-cosmofns-smooth.dat").T
    kc = np.genfromtxt(root+"-fisher-kc.dat").T

    # Load Fisher matrices as fn. of z
    Nbins = zc.size
    F_list = [np.genfromtxt(root+"-fisher-full-%d.dat" % i) for i in range(Nbins)]
        
    # EOS FISHER MATRIX
    pnames = rf.load_param_names(root+"-fisher-full-0.dat")
    zfns = ['b_HI',]
    excl = ['Tb', 'f', 'H', 'DA', 'apar', 'aperp', 'pk*', 'N_eff', 'fs8', 'bs8', 'A']
    F, lbls = rf.combined_fisher_matrix( F_list, expand=zfns, names=pnames,
                                         exclude=excl )
    """
    # Add DETF Planck prior?
    print "*** Using DETF Planck prior ***"
    l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h', 'sigma8']
    F_detf = euclid.detf_to_rf("DETF_PLANCK_FISHER.txt", cosmo, omegab=False)
    Fpl, lbls = rf.add_fisher_matrices(F, F_detf, lbls, l2, expand=True)
    if len(fixed_params) > 0:
        Fpl, lbls = rf.combined_fisher_matrix( [Fpl,], expand=[], 
                     names=lbls, exclude=fixed_params )
    """
    
    # Output Fisher matrix
    fname = "%s_%s.dat" % (fmroot, names[k])
    np.savetxt(fname, F, header=" / ".join(lbls))
    print "Saved Fisher matrix to: %s" % fname
