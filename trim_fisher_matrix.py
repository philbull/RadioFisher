#!/usr/bin/env python
"""
Combine an output Fisher matrix into a simplified version for other codes to use.
"""
import numpy as np
import radiofisher as rf
import os, sys

cosmo = rf.experiments.cosmo
#expt_name = 'gDESI_CV_desicv'
#expt_name = 'iCosVis256x256_2yr_wedge'
#expt_name = 'iCosVis256x256_2yr_3pbwedge'
#expt_name = 'iHIRAX_2yr_horizwedge'
#expt_name = 'HETDEXdz03'
#expt_name = 'iHIRAX_highz_2yr' #_3pbwedge'
#expt_name = 'gCVLOWZ'
#expt_name = 'SpecTel'
expt_name = 'gCVALLZ'

ZBIN_RANGE = None
ZBIN_RANGE = (int(sys.argv[1]), 30)

# Load Fisher matrices and combine
root = "output/" + expt_name

# Load cosmo fns.
dat = np.atleast_2d( np.genfromtxt(root+"-cosmofns-zc.dat") ).T
zc, Hc, dAc, Dc, fc = dat
z, H, dA, D, f = np.genfromtxt(root+"-cosmofns-smooth.dat").T
kc = np.genfromtxt(root+"-fisher-kc.dat").T

# Load Fisher matrices as fn. of z
if ZBIN_RANGE is None:
    Nbins = zc.size
    F_list = [np.genfromtxt(root+"-fisher-full-%d.dat" % i) for i in range(Nbins)]
else:
    # Only select a subset of z bins
    izmin, izmax = ZBIN_RANGE
    Nbins = zc.size
    zidxs = np.arange(Nbins)[izmin:izmax]
    
    zc = zc[izmin:izmax]
    F_list = [np.genfromtxt(root+"-fisher-full-%d.dat" % i) for i in zidxs]
    zmin, zmax = zc[0], zc[-1]

# Actually, (aperp, apar) are (D_A, H)
pnames = rf.load_param_names(root+"-fisher-full-0.dat")

zfns = ['bs8', 'fs8', 'H', 'DA',]
excl = ['Tb', 'n_s', 'sigma8', 'omegak', 'omegaDE', 'w0', 'wa', 'h',
        'gamma', 'N_eff', 'pk*', 'f', 'b_HI', 
        'gamma0', 'gamma1', 'eta0', 'eta1', 'A_xi', 'logkmg',
        'sigma8tot', 'sigma_8', 'k*', 'A', 'aperp', 'apar']

# Combine into a single Fisher matrix
F, lbls = rf.combined_fisher_matrix( F_list,
                                     expand=zfns, names=pnames,
                                     exclude=excl )
print(lbls)

# Save Fisher matrix to file
if ZBIN_RANGE is None:
    np.savetxt("Fisher-full-%s.dat" % expt_name, F, header=" ".join(lbls))
    np.savetxt("Fisher-full-%s.zbins" % expt_name, zc)
else:
    print("zmin = %2.2f, zmax = %2.2f" % (zmin, zmax))
    np.savetxt("Fisher-full-%s-%2.2f-%2.2f.dat" % (expt_name, zmin, zmax), 
               F, header=" ".join(lbls))
    np.savetxt("Fisher-full-%s-%2.2f-%2.2f.zbins" % (expt_name, zmin, zmax), zc)

