#!/usr/bin/python
"""
Plot 2D constraints on M_nu and sigma_8.
"""
import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
import matplotlib.ticker
import os
from radiofisher import euclid

#fig_name = "pub-w0omegaDE.pdf"
#fig_name = "fig13-w0omegaDE-okfixed.pdf"

MARGINALISE_CURVATURE = False # Marginalise over Omega_K
MARGINALISE_INITIAL_PK = True # Marginalise over n_s, sigma_8
MARGINALISE_OMEGAB = True # Marginalise over Omega_baryons

cosmo = rf.experiments.cosmo
names = ['EuclidRef_mnumnu006', 'SKA1MID350mnu006'] #, 'exptS']
labels = ['DETF IV', 'SKA1-MID 350'] #, 'Stage I']

colours = [ ['#CC0000', '#F09B9B'],
            ['#1619A1', '#B1C9FD'],
            ['#5B9C0A', '#BAE484'],
            ['#FFB928', '#FFEA28'] ]

# Fiducial value and plotting
fig = P.figure()
ax1 = fig.add_subplot(111)

_k = range(len(names))[::-1]
for k in _k:
    root = "output/" + names[k]

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
    excl = ['Tb', 'f', 'fs8', 'bs8', 'aperp', 'apar', 'DA', 'H', 'gamma', 'N_eff', 'pk*']
    F, lbls = rf.combined_fisher_matrix( F_list, expand=zfns, names=pnames,
                                         exclude=excl )
    # Add Planck prior
    #Fpl = euclid.add_detf_planck_prior(F, lbls, info=False)
    #Fpl = euclid.add_planck_prior(F, lbls, info=False)
    
    # (a) DETF Planck prior
    #print "*** Using DETF Planck prior ***"
    #l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h', 'sigma8']
    #F_detf = euclid.detf_to_rf("DETF_PLANCK_FISHER.txt", cosmo, omegab=False)
    #Fpl, lbls = rf.add_fisher_matrices(F, F_detf, lbls, l2, expand=True)
    
    # (b) Planck Fisher matrix (derived from Planck 2015 MCMC chains)
    lbls_planck = ['n_s', 'Omega_b', 'omegaDE', 'h', 'sigma8', 'Mnu', 'Neff']
    F_planck = np.genfromtxt("/home/phil/oslo/Dropbox/HI_neutrinos/fisher_code/planck/planck_derived_fisher.dat").T
    F_planck, lbls_planck = rf.combined_fisher_matrix( [F_planck,], expand=[], 
                                    names=lbls_planck, exclude=['Neff',] )
    
    # Add Planck prior
    Fpl, lbls = rf.add_fisher_matrices(F, F_planck, lbls, lbls_planck, expand=True)
    
    # Decide whether to fix various parameters
    fixed_params = []
    if not MARGINALISE_CURVATURE: fixed_params += ['omegak',]
    if not MARGINALISE_INITIAL_PK: fixed_params += ['n_s', 'sigma8']
    if not MARGINALISE_OMEGAB: fixed_params += ['omega_b',]
    
    # Fix (w0, wa)
    #fixed_params += ['w0', 'wa']
    
    if len(fixed_params) > 0:
        Fpl, lbls = rf.combined_fisher_matrix( [Fpl,], expand=[], 
                     names=lbls, exclude=fixed_params )
    
    # Really hopeful H0 prior
    #ph = lbls.index('h')
    #Fpl[ph, ph] += 1./(0.012)**2.
    
    # Get indices of w0, Omega_DE
    pmnu = lbls.index('Mnu')
    psig8 = lbls.index('sigma8')
    
    print "-"*50
    print names[k]
    print "-"*50
    
    # Invert matrix
    cov_pl = np.linalg.inv(Fpl)
    
    # Calculate FOM
    print "2*sigma(M_nu) = %3.4f" % (2. * np.sqrt(cov_pl[pmnu,pmnu]))
    
    x = 0.06 #rf.experiments.cosmo['Mnu']
    y = rf.experiments.cosmo['sigma_8']
    
    transp = [1., 0.85]
    
    # Plot contours for w0, omega_DE
    w, h, ang, alpha = rf.ellipse_for_fisher_params(pmnu, psig8, None, Finv=cov_pl)
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                height=alpha[kk]*h, angle=ang, fc=colours[k][kk], 
                ec=colours[k][0], lw=1.5, alpha=transp[kk]) for kk in [1,0]]
    for e in ellipses: ax1.add_patch(e)
    ax1.plot(x, y, 'kx', mew=1.2)

# Report on what options were used
print "-"*50
s1 = "Marginalised over Omega_K" if MARGINALISE_CURVATURE else "Fixed Omega_K"
s2 = "Marginalised over ns, sigma8" if MARGINALISE_INITIAL_PK else "Fixed ns, sigma8"
s3 = "Marginalised over Omega_b" if MARGINALISE_OMEGAB else "Fixed Omega_b"
print "NOTE:", s1
print "NOTE:", s2
print "NOTE:", s3




P.show()
exit()


# Move subplots
# pos = [[x0, y0], [x1, y1]]
l0 = 0.2
b0 = 0.10
ww = 0.75
hh = 0.89 / 2.
ax1.set_position([l0, b0, ww, hh])
ax2.set_position([l0, b0 + hh, ww, hh])


ax1.tick_params(axis='both', which='major', labelsize=20, size=8., width=1.5, pad=8.)
ax2.tick_params(axis='both', which='major', labelsize=20, size=8., width=1.5, pad=8.)
ax1.tick_params(axis='both', which='minor', labelsize=20, size=5., width=1.5)
ax2.tick_params(axis='both', which='minor', labelsize=20, size=5., width=1.5)

ax1.set_xlabel(r"$w_0$", fontdict={'fontsize':'xx-large'}, labelpad=15.)
ax2.tick_params(axis='x', which='major', labelbottom='off')

ax1.set_ylabel(r"$\Omega_\mathrm{DE}$", fontdict={'fontsize':'xx-large'}, labelpad=15.)
ax2.set_ylabel(r"$w_a$", fontdict={'fontsize':'xx-large'}, labelpad=15.)

ax1.set_xlim((-1.45, -0.55))
ax2.set_xlim((-1.45, -0.55))
ax2.set_ylim((-0.95, 0.95))
ax1.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
ax1.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
ax1.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
ax2.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
ax2.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.4))
ax2.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.2))


# Legend
labels = [labels[k] + " + Planck" for k in range(len(labels))]
lines = [ matplotlib.lines.Line2D([0.,], [0.,], lw=8.5, color=colours[k][0], alpha=0.65) for k in range(len(labels))]

ax2.legend((l for l in lines), (name for name in labels), loc='upper right', prop={'size':'large'}, frameon=False)

"""
xminorLocator = matplotlib.ticker.MultipleLocator(0.1)
yminorLocator = matplotlib.ticker.MultipleLocator(0.5)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)

ax.set_xlabel(r"$w_0$", fontdict={'fontsize':'xx-large'}, labelpad=15.)
ax.set_ylabel(r"$\Omega_\mathrm{DE}$", fontdict={'fontsize':'xx-large'}, labelpad=15.)

ax.set_xlim((-1.75, -0.25))
ax.set_ylim((0.60, 0.77))
"""

# Set size and save
#P.tight_layout()
P.gcf().set_size_inches(8.,9.)
P.savefig(fig_name, transparent=True)
P.show()
