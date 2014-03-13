#!/usr/bin/python
"""
Plot functions of redshift.
"""
import numpy as np
import pylab as P
import baofisher
import matplotlib.patches
import matplotlib.cm
from units import *
from mpi4py import MPI
import experiments
import os
import euclid

# Choose which function of redshift to plot
#fn = 'DA'
#fn = 'H'
fn = 'f'

cosmo = experiments.cosmo

names = ['EuclidRef', 'cexptL', 'iexptM', 'exptS']
colours = ['#CC0000', '#1619A1', '#5B9C0A', '#990A9C'] # DETF/F/M/S
labels = ['DETF IV', 'Facility', 'Mature', 'Snapshot']
linestyle = [[2, 4, 6, 4], [1,0], [8, 4], [3, 4]]


cosmo_fns = baofisher.background_evolution_splines(cosmo)
#cosmo = baofisher.load_power_spectrum(cosmo, "cache_pk.dat", force_load=True)

# Fiducial value and plotting
fig = P.figure()
ax1 = fig.add_subplot(111)

for k in range(len(names)):
    root = "output/" + names[k]

    # Load cosmo fns.
    dat = np.atleast_2d( np.genfromtxt(root+"-cosmofns-zc.dat") ).T
    zc, Hc, dAc, Dc, fc = dat
    z, H, dA, D, f = np.genfromtxt(root+"-cosmofns-smooth.dat").T
    kc = np.genfromtxt(root+"-fisher-kc.dat").T

    # Load Fisher matrices as fn. of z
    Nbins = zc.size
    F_list = [np.genfromtxt(root+"-fisher-full-%d.dat" % i) for i in range(Nbins)]
    
    # EOS FISHER MATRIX
    # Actually, (aperp, apar) are (D_A, H)
    pnames = ['A', 'b_HI', 'Tb', 'sigma_NL', 'sigma8', 'n_s', 'f', 'aperp', 'apar', 
             'omegak', 'omegaDE', 'w0', 'wa', 'h', 'gamma'] #, 'fNL']
    pnames += ["pk%d" % i for i in range(kc.size)]
    zfns = [0,1,6,7,8]
    excl = [2,4,5,  9,10,11,12,13,14] # Exclude all cosmo params
    excl += [i for i in range(len(pnames)) if "pk" in pnames[i]]
    
    F, lbls = baofisher.combined_fisher_matrix( F_list,
                                                expand=zfns, names=pnames,
                                                exclude=excl )
    cov = np.linalg.inv(F)
    
    # Get functions of z
    zfns = [0,1,3,4,5] # A, b_HI, f, DA, H
    pA  = baofisher.indexes_for_sampled_fns(0, zc.size, zfns)
    #pb  = baofisher.indexes_for_sampled_fns(1, zc.size, zfns)
    pDA = baofisher.indexes_for_sampled_fns(4, zc.size, zfns)
    pH  = baofisher.indexes_for_sampled_fns(5, zc.size, zfns)
    pf  = baofisher.indexes_for_sampled_fns(3, zc.size, zfns)
    
    if k == 0:
        print "DA:", [lbls[j] for j in pDA], "\n"
        print "H:", [lbls[j] for j in pH], "\n"
        print "f:", [lbls[j] for j in pf], "\n"
    
    # Plot errorbars
    errs = np.sqrt(np.diag(cov))
    if fn == 'DA':
        yc = dAc
        err = 1e3 * errs[pDA]
    if fn == 'H':
        yc = Hc
        err = 1e2 * errs[pH]
    if fn == 'f':
        yc = fc
        err = errs[pf]
    
    line = ax1.errorbar( zc, yc, yerr=err, color=colours[k], lw=1.8, marker='.',
                         label=labels[k],)
    # Set custom linestyle    
    line[0].set_dashes(linestyle[k])
    
    if k == 0:
        if fn == 'DA': ax1.plot(z, dA, color='k', lw=2., alpha=0.4)
        if fn == 'H':  ax1.plot(z, H, color='k', lw=2., alpha=0.4)
        if fn == 'f':  ax1.plot(z, f, color='k', lw=2., alpha=0.4)

if fn == 'DA':
    ax1.set_xlim((0.2, 1.65))
    ax1.set_ylim((600., 2100.))
if fn == 'H':
    ax1.set_xlim((0.2, 1.65))
    ax1.set_ylim((55., 182.))
if fn == 'f':
    ax1.set_xlim((0.1, 2.0))
    ax1.set_ylim((0.55, 1.0))
axes = [ax1,]

# Resize labels/ticks
fontsize = 18
for i in range(len(axes)):
    ax = axes[i]
    for tick in ax.xaxis.get_major_ticks():
      tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
      tick.label1.set_fontsize(fontsize)

# y labels
if fn == 'DA':
    ax1.set_ylabel(r"$D_A(z)$", fontdict={'fontsize':'20'})
if fn == 'H':
    ax1.set_ylabel(r"$H(z)$", fontdict={'fontsize':'20'})
if fn == 'f':
    ax1.set_ylabel(r"$f(z)$", fontdict={'fontsize':'20'})

ax1.set_xlabel(r"$z$", fontdict={'fontsize':'20'})
ax1.legend(loc='upper left', prop={'size':'large'})

#P.tight_layout()
# Set size
P.gcf().set_size_inches(8.5, 7.)

if fn == 'DA': P.savefig('pub-da-z.pdf', transparent=True)
if fn == 'H': P.savefig('pub-h-z.pdf', transparent=True)
if fn == 'f': P.savefig('pub-f-z.pdf', transparent=True)

P.show()
