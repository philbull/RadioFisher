#!/usr/bin/python
"""
Plot 1D constraints on Omega_K as a series of errorbars
"""
import numpy as np
import pylab as P
import baofisher
import matplotlib.patches
import matplotlib.cm
import matplotlib.ticker
from units import *
from mpi4py import MPI
import experiments
import os
import euclid

cosmo = experiments.cosmo

MARGINALISE_W0WA = False
if MARGINALISE_W0WA:
    print "*** Marginalising over (w_0, w_a) ***"
else:
    print "*** Fixing (w_0, w_a) ***"

names = ["exptS", "iexptM", "cexptL"] #, "cexptL"]
Nexpt = 2 #3

names = ["cSKA1MID", "SKA1SUR",] # "superSKA1MID"] # "SKA1MID"] # "superSKA1MID"
labels = ["SKA1-MID (Combined)", "SKA1-SUR (Dish)", "SKA1-MID (Combined) + Planck", "SKA1-SUR (Dish) + Planck", "Super + Planck"] #, "SKA1-MID (Dish)"] "SuperMID"

colours = ['#CC0000', '#ED5F21', '#5B9C0A', '#1619A1', '#FAE300', '#56129F', '#990A9C', 'y', 'c']
colours += colours + colours
labels += labels + labels + labels
print labels

"""
names = ['SKA1MID190', 'SKA1MID250', 'SKA1MID350',
         'SKA1MID190oHI9', 'SKA1MID250oHI9', 'SKA1MID350oHI9',
         'SKA1MID350oHI9-numax1150', 'SKA1MID350oHI9-numax1150-dnu800',
         'SKA1MID350oHI9-numax1150-dnu800-nokfg']
labels = ['SKA1MID190', 'SKA1MID250', 'SKA1MID350',
         'SKA1MID190oHI9', 'SKA1MID250oHI9', 'SKA1MID350oHI9',
         'SKA1MID350oHI9-numax1150', 'SKA1MID350oHI9-numax1150-dnu800',
         'SKA1MID350oHI9-numax1150-dnu800-nokfg']
"""

#labels = ['Snapshot', 'Mature', 'Behemoth', 'Snapshot + Planck', 'Mature + Planck', 'Behemoth + Planck']

cosmo_fns, cosmo = baofisher.precompute_for_fisher(experiments.cosmo, "camb/baofisher_matterpower.dat")
H, r, D, f = cosmo_fns

# Fiducial value and plotting
fig = P.figure()
ax = fig.add_subplot(111)
m = 0

for k in range(len(names)):
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
    # Actually, (aperp, apar) are (D_A, H)
    pnames = ['A', 'b_HI', 'Tb', 'sigma_NL', 'sigma8', 'n_s', 'f', 'aperp', 'apar', 
             'omegak', 'omegaDE', 'w0', 'wa', 'h', 'gamma', 'Mnu']
    pnames += ["pk%d" % i for i in range(kc.size)]
    
    zfns = [1,]
    excl = [2,4,5,  6,7,8, 14,15]
    if not MARGINALISE_W0WA: excl += [11, 12] # Marginalise over (w0, wa)
    
    excl += [i for i in range(len(pnames)) if "pk" in pnames[i]]
    
    F, lbls = baofisher.combined_fisher_matrix( F_list,
                                                expand=zfns, names=pnames,
                                                exclude=excl )
    # Add Planck prior
    Fpl = euclid.add_planck_prior(F, lbls, info=False)
    
    print "-"*50
    print names[k]
    print "-"*50
    
    # Invert matrices
    pok = baofisher.indexes_for_sampled_fns(3, zc.size, zfns)
    cov = np.linalg.inv(F)
    cov_pl = np.linalg.inv(Fpl)
    err = np.sqrt(cov[pok,pok])
    err_pl = np.sqrt(cov_pl[pok,pok])
    
    omegak = 0.
    
    # Without Planck
    if True: #m != 0 and not MARGINALISE_W0WA: # Skip exptS without Planck
        P.errorbar( omegak, m, xerr=err, color=colours[m], lw=2., marker='.', 
                    markersize=10. )
        P.annotate( labels[m], xy=(omegak, m), xytext=(0., 10.), fontsize='large',
                    textcoords='offset points', ha='center', va='bottom' )
    
    # With Planck
    P.errorbar( omegak, m+Nexpt, xerr=err_pl, color=colours[m+Nexpt], lw=2., marker='.', 
                markersize=10. )
    P.annotate( labels[m+Nexpt], xy=(omegak, m+Nexpt), xytext=(0., 10.), fontsize='large',
                textcoords='offset points', ha='center', va='bottom' )
    m += 1
    print err, err_pl


# Planck-only 1D
omegak_planck_up = -5e-4 + 0.5*6.6e-3 # From Planck 2013 XVI, Table 10, Planck+WMAP+highL+BAO, 95% CL
omegak_planck_low = -5e-4 - 0.5*6.6e-3

P.axvspan(omegak_planck_up, 1., ec='none', fc='#f2f2f2')
P.axvspan(-1., omegak_planck_low, ec='none', fc='#f2f2f2')
P.axvline(omegak_planck_up, ls='dotted', color='k', lw=2.)
P.axvline(omegak_planck_low, ls='dotted', color='k', lw=2.)

# Shaded region, tick number format, and minor tick location
minorLocator = matplotlib.ticker.MultipleLocator(0.001)
majorFormatter = matplotlib.ticker.FormatStrFormatter('%3.3f')
ax.xaxis.set_minor_locator(minorLocator)
ax.xaxis.set_major_formatter(majorFormatter)

fontsize = 20
for tick in ax.xaxis.get_major_ticks():
  tick.label1.set_fontsize(fontsize)
for tick in ax.yaxis.get_major_ticks():
  tick.label1.set_visible(False)
  #tick.label1.set_fontsize(fontsize)

ax.set_xlabel(r"$\Omega_K$", fontdict={'fontsize':'20'})
#ax.set_ylabel(r"$w_a$", fontdict={'fontsize':'20'})

if MARGINALISE_W0WA:
    ax.set_ylim((1., 6.)) # 2, 6
else:
    ax.set_ylim((-1., 4.)) # 0, 6


ax.set_xlim((-8.1e-3, 8.1e-3))
P.tight_layout()

# Set size and save
#P.gcf().set_size_inches(16.5,10.5)
if MARGINALISE_W0WA:
    P.savefig('pub-omegak-w0wamarg.png', dpi=100)
else:
    P.savefig('pub-omegak-w0wafixed.png', dpi=100)

P.show()
