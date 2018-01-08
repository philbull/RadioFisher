#!/usr/bin/python
"""
Plot FOM as a function of survey area for SKA galaxy redshift surveys.
"""
import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
import matplotlib.ticker
from radiofisher.units import *
import os
from radiofisher import euclid

cosmo = rf.experiments.cosmo
sarea_vals = [100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 
              6000, 7000, 8000, 9000, 10000, 12000, 15000, 17000, 20000, 22000, 
              25000, 27000, 30000]

basenames = ['SKA1ref', 'SKA1ref_800_1300']
fom_list = []
for basename in basenames:
    foms = []
    for sarea in sarea_vals:
        root = "output/%s_%d" % (basename, sarea)
        print "="*50
        print root
        print "="*50

        # Load cosmo fns.
        dat = np.atleast_2d( np.genfromtxt(root+"-cosmofns-zc.dat") ).T
        zc, Hc, dAc, Dc, fc = dat
        zs, Hs, dAs, Ds, fs = np.genfromtxt(root+"-cosmofns-smooth.dat").T
        kc = np.genfromtxt(root+"-fisher-kc.dat").T
        
        # Load Fisher matrices as fn. of z
        Nbins = zc.size
        F_list = [np.genfromtxt(root+"-fisher-full-%d.dat" % i) for i in range(Nbins)]
        
        # EOS FISHER MATRIX
        pnames = rf.load_param_names(root+"-fisher-full-0.dat")
        zfns = ['b_HI',]
        excl = ['Tb', 'f', 'aperp', 'apar', 'DA', 'H', 'N_eff', 'pk*', 'fs8', 
                'bs8', 'gamma']
        F, lbls = rf.combined_fisher_matrix( F_list, expand=zfns, names=pnames,
                                             exclude=excl )
        # Add Planck prior
        print "*** Using DETF Planck prior ***"
        l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h', 'sigma8']
        F_detf = euclid.detf_to_rf("DETF_PLANCK_FISHER.txt", cosmo, omegab=False)
        Fpl, lbls = rf.add_fisher_matrices(F, F_detf, lbls, l2, expand=True)
        #Fpl = F; lbls = lbls
        
        # Decide whether to fix various parameters
        fixed_params = []
        #fixed_params = ['omegak', 'n_s', 'sigma8', 'omega_b',]
        if len(fixed_params) > 0:
            Fpl, lbls = rf.combined_fisher_matrix( [Fpl,], expand=[], 
                         names=lbls, exclude=fixed_params )
        
        # Invert matrix and calculate FOM
        pw0 = lbls.index('w0'); pwa = lbls.index('wa')
        cov_pl = np.linalg.inv(Fpl)
        fom = rf.figure_of_merit(pw0, pwa, None, cov=cov_pl)
        foms.append(fom)
    fom_list.append(foms)


for i in range(len(sarea_vals)):
    print "%6d: %6.3f %6.3f" % (sarea_vals[i], fom_list[0][i], fom_list[1][i])


P.subplot(111)
P.plot(sarea_vals, fom_list[0], 'r-', marker='.', lw=1.5, label="SKA1-Ref")
P.plot(sarea_vals, fom_list[1], 'b-', marker='.', lw=1.5, label="SKA1-Ref 800-1300 MHz")

P.xlabel(r"$S_{\rm area}$", fontdict={'fontsize':'xx-large'}, labelpad=15.)
P.ylabel(r"${\rm FOM}$", fontdict={'fontsize':'xx-large'})
P.tick_params(axis='both', which='major', labelsize=18, size=8., width=1.5, pad=8.)
P.title("SKA1-REF (with Planck prior)")
P.legend(loc='lower right', frameon=False)
P.tight_layout()

P.savefig("ska1ref_sarea_planck.pdf", transparent=True)
P.show()
