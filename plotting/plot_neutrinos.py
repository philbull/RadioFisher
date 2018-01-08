#!/usr/bin/python
"""
Plot 2D constraints on (M_nu, n_s).
"""
import numpy as np
import pylab as P
from rfwrapper import rf
import radiofisher.euclid as euclid
import matplotlib.patches
import matplotlib.cm
import matplotlib.ticker

cosmo = rf.experiments.cosmo
names = ['gSKA2_mnumnu010', 'EuclidRef_mnumnu010', 'SKA1MID900mnu010',]
labels = ['Full SKA (gal. survey)', 'Euclid (gal. survey)', 'SKA1-MID IM (900)',]

colours = [ ['#1619A1', '#B1C9FD'],
            ['#CC0000', '#F09B9B'],
            ['#FFB928', '#FFEA28'],
            ['#5B9C0A', '#BAE484'],
            ['#5B9C0A', '#BAE484'],
            ['#FFB928', '#FFEA28'],
            ['#FFB928', '#FFEA28'],
            ['#FFB928', '#FFEA28'],
            ['#FFB928', '#FFEA28'], ]

# Fiducial value and plotting
fig = P.figure()
ax = fig.add_subplot(111)

_k = range(len(names))[::-1]
for k in _k:
    root = "output/" + names[k]
    
    print ">"*50
    print "We're doing", names[k]

    # Load cosmo fns.
    dat = np.atleast_2d( np.genfromtxt(root+"-cosmofns-zc.dat") ).T
    zc, Hc, dAc, Dc, fc = dat
    zs, Hs, dAs, Ds, fs = np.genfromtxt(root+"-cosmofns-smooth.dat").T
    kc = np.genfromtxt(root+"-fisher-kc.dat").T
    
    # Load Fisher matrices as fn. of z
    Nbins = zc.size
    F_list = [np.genfromtxt(root+"-fisher-full-%d.dat" % i) for i in range(Nbins)]
    #F_list = [np.genfromtxt(root+"-fisher-full-%d.dat" % i) for i in np.where(zc>0.5)[0]]
    #Nbins = np.where(zc > 0.5)[0].size
    #if Nbins == 0: continue
    
    # EOS FISHER MATRIX
    pnames = rf.load_param_names(root+"-fisher-full-0.dat")
    zfns = ['b_HI',]
    excl = ['Tb', 'f', 'aperp', 'apar', 'DA', 'H', 'pk*', 'fs8', 'bs8',   'N_eff']
    F, lbls = rf.combined_fisher_matrix( F_list, expand=zfns, names=pnames,
                                         exclude=excl )
    
    # Add Planck prior
    print "*** Using DETF Planck prior ***"
    l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h', 'sigma8']
    F_detf = euclid.detf_to_rf("DETF_PLANCK_FISHER.txt", cosmo, omegab=False)
    Fpl, lbls = rf.add_fisher_matrices(F, F_detf, lbls, l2, expand=True)
    
    # Remove extra params
    Fpl, lbls = rf.combined_fisher_matrix( [Fpl,], expand=[], 
                     names=lbls, exclude=['omegak', 'w0', 'wa', 'gamma',] )
    print lbls
    
    # Get indices of w0, wa
    pMnu = lbls.index('Mnu'); pNeff = lbls.index('n_s')
    
    print "-"*50
    print names[k]
    print "-"*50
    
    # Invert matrix
    cov_pl = np.linalg.inv(Fpl)
    
    # Calculate FOM
    print "1D sigma(Mnu) = %3.4f" % np.sqrt(cov_pl[pMnu,pMnu])
    print "1D sigma(n_s) = %3.4f" % np.sqrt(cov_pl[pNeff,pNeff])
    
    x = 0.1 #rf.experiments.cosmo['w0'] # Mnu
    y = rf.experiments.cosmo['ns'] #3.046 #rf.experiments.cosmo['wa'] # Neff
    
    # Plot contours for w0, wa; omega_k free
    transp = [1., 0.85]
    w, h, ang, alpha = rf.ellipse_for_fisher_params(pMnu, pNeff, None, Finv=cov_pl)
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                height=alpha[kk]*h, angle=ang, fc=colours[k][kk], 
                ec=colours[k][0], lw=1.5, alpha=transp[kk]) for kk in [1,0]]
    for e in ellipses: ax.add_patch(e)
    
    # Centroid
    ax.plot(x, y, 'kx')
    print "\nDONE\n"

# Legend
labels = [labels[k] for k in range(len(labels))]
lines = [ matplotlib.lines.Line2D([0.,], [0.,], lw=8.5, color=colours[k][0], alpha=0.65) for k in range(len(labels))]

P.gcf().legend((l for l in lines), (name for name in labels), prop={'size':'x-large'}, bbox_to_anchor=[0.93, 0.96], frameon=False)

ax.tick_params(axis='both', which='major', labelsize=20, size=8., width=1.5, pad=8.)
xminorLocator = matplotlib.ticker.MultipleLocator(0.1)
yminorLocator = matplotlib.ticker.MultipleLocator(0.5)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)

ax.set_xlabel(r"$M_\nu$", fontdict={'fontsize':'xx-large'}, labelpad=15.)
ax.set_ylabel(r"$n_s$", fontdict={'fontsize':'xx-large'})


ax.set_xlim((0., 0.23))
ax.set_ylim((0.9545, 0.973))

# Set size and save
P.tight_layout()
P.gcf().set_size_inches(8.,6.)
P.savefig("ska-neutrinos.pdf", transparent=True)
P.show()
