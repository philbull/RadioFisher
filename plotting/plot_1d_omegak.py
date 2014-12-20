#!/usr/bin/python
"""
Plot 1D constraints on a parameter, by default the curvature (Fig. 16).
"""
import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
import matplotlib.ticker
import os
from radiofisher import euclid

cosmo = rf.experiments.cosmo

fig_name = "fig16-ok.pdf"
#fig_name = "ska-omegak.png"

param1 = "omegak"
label1 = "$\Omega_K$"
fid1 = 0.

USE_DETF_PLANCK_PRIOR = True
MARGINALISE_CURVATURE = True    # Marginalise over Omega_K
MARGINALISE_INITIAL_PK = True   # Marginalise over (n_s, sigma_8)
MARGINALISE_OMEGAB = True       # Marginalise over Omega_baryons
MARGINALISE_W0WA = True         # Marginalise over (w0, wa)

names = ['EuclidRef_paper', 'exptL_paper', 'aexptM_paper'] #, 'exptS']
labels = ['DETF IV', 'Facility', 'Stage II'] #, 'Stage I']
colours = ['#BAE484', '#5B9C0A',
           '#B1C9FD', '#1619A1',
           '#F6ADAD', '#CC0000',
           '#FFB928', '#FFEA28']

# Fiducial value and plotting
fig = P.figure()
ax = fig.add_subplot(111)

Nexpt = len(names)
m = 0
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
    excl = ['Tb', 'f', 'aperp', 'apar', 'H', 'DA', 'gamma', 'N_eff', 'pk*', 'fs8', 'bs8']
    F, lbls = rf.combined_fisher_matrix( F_list, expand=zfns, names=pnames,
                                         exclude=excl )
    
    # DETF Planck prior
    print "*** Using DETF Planck prior ***"
    l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h']
    F_detf = euclid.detf_to_rf("DETF_PLANCK_FISHER.txt", cosmo)
    Fpl, lbls = rf.add_fisher_matrices(F, F_detf, lbls, l2, expand=True)
    
    # Decide whether to fix various parameters
    fixed_params = []
    if not MARGINALISE_CURVATURE: fixed_params += ['omegak',]
    if not MARGINALISE_INITIAL_PK: fixed_params += ['n_s', 'sigma8']
    if not MARGINALISE_OMEGAB: fixed_params += ['omega_b',]
    if not MARGINALISE_W0WA: fixed_params += ['w0', 'wa']
    
    if len(fixed_params) > 0:
        print "REMOVING:", fixed_params
        Fpl, lbls = rf.combined_fisher_matrix( [Fpl,], expand=[], 
                     names=lbls, exclude=fixed_params )
    
    # Invert matrix (w0, wa marginalised)
    cov_pl = np.linalg.inv(Fpl)
    
    # Invert matrix with (w0,wa) fixed
    Fpl2, lbls2 = rf.combined_fisher_matrix( [Fpl,], expand=[], 
                     names=lbls, exclude=['w0', 'wa'] )
    cov_pl2 = np.linalg.inv(Fpl2)
    
    # Plot errorbars and annotate
    # (w0, wa) free
    p1 = lbls.index(param1)
    x = fid1
    err = np.sqrt(np.diag(cov_pl))[p1]
    print "%10s -- %s: %4.2e" % (lbls[p1], names[k], err), "\n"
    
    ax.errorbar( x, m, xerr=err, color=colours[m], lw=2., 
                 marker='.', markersize=10., markeredgewidth=2. )
    ax.annotate( labels[k], xy=(x, m+1.0), xytext=(0., -10.), 
                 fontsize='large', textcoords='offset points', ha='center', va='center' )
    
    # (w0, wa) fixed
    p1 = lbls.index(param1)
    x = fid1
    err = np.sqrt(np.diag(cov_pl2))[p1]
    print "\n%10s -- %s: %4.2e" % (lbls[p1], names[k], err), "(w0/wa fixed)\n"
    
    ax.errorbar( x, m+0.5, xerr=err, color=colours[m+1], lw=2., markeredgewidth=2.,
                 marker='.', markersize=10. )
    #ax.annotate( labels[k], xy=(x, m), xytext=(0., 10.), 
    #             fontsize='large', textcoords='offset points', ha='center', va='bottom' )
    m += 2


# Report on what options were used
print "-"*50
s1 = "Marginalised over Omega_K" if MARGINALISE_CURVATURE else "Fixed Omega_K"
s2 = "Marginalised over ns, sigma8" if MARGINALISE_INITIAL_PK else "Fixed ns, sigma8"
s3 = "Marginalised over Omega_b" if MARGINALISE_OMEGAB else "Fixed Omega_b"
s4 = "Marginalised over w0, wa" if MARGINALISE_W0WA else "Fixed w0, wa"
print "NOTE:", s1
print "NOTE:", s2
print "NOTE:", s3
print "NOTE:", s4

# Planck-only 1D
omegak_planck_up = -5e-4 + 0.5*6.6e-3 # From Planck 2013 XVI, Table 10, Planck+WMAP+highL+BAO, 95% CL
omegak_planck_low = -5e-4 - 0.5*6.6e-3

ax.axvspan(omegak_planck_up, 1., ec='none', fc='#f2f2f2')
ax.axvspan(-1., omegak_planck_low, ec='none', fc='#f2f2f2')
ax.axvline(omegak_planck_up, ls='dotted', color='k', lw=2.)
ax.axvline(omegak_planck_low, ls='dotted', color='k', lw=2.)

#ax.axvspan(-4e-4, 4e-4, ec='none', fc='#f2f2f2')
#ax.axvline(-4e-4, ls='dotted', color='k', lw=1.5)
#ax.axvline(+4e-4, ls='dotted', color='k', lw=1.5)

fontsize = 20
for tick in ax.xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in ax.yaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)

ax.set_xlabel(label1, fontdict={'fontsize':'xx-large'}, labelpad=15.)
#ax.set_ylabel(label2, fontdict={'fontsize':'xx-large'}, labelpad=15.)

ax.set_xlim((-5e-3, 5e-3))
ax.set_ylim((-0.75, 2.*Nexpt - 0.5))

ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.002))
ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.001))

ax.tick_params(axis='both', which='major', labelleft='off', left='off', right='off', length=8., width=1.5, pad=8.)
ax.tick_params(axis='both', which='minor', labelleft='off', left='off', right='off', length=5., width=1.5, pad=8.)

# Set size and save
P.tight_layout()
P.gcf().set_size_inches(8.,6.)
P.savefig(fig_name, transparent=True)
P.show()
