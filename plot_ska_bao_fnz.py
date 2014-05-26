#!/usr/bin/python
"""
Process EOS Fisher matrices and plot P(k).
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

cosmo = experiments.cosmo

CUR_K = 2

names = ["SKAHI73", "EuclidRef", "SKAHI100",] # "SKAHI73", 'EuclidRef']
labels = ['SKA2 HI gal.', 'Euclid', 'SKA1 HI gal.']
colours = ['#1619A1', '#CC0000', '#FFB928', '#5B9C0A', '#990A9C', '#FFB928', '#CC0000']
linestyle = [[1,0], [1, 0], [1, 0],]
marker = ['o', 'D', 's',]

# Get f_bao(k) function
cosmo_fns = baofisher.background_evolution_splines(cosmo)
cosmo = baofisher.load_power_spectrum(cosmo, "cache_pk.dat", force_load=True)
fbao = cosmo['fbao']

# Fiducial value and plotting
#axes = [P.subplot(111),]

for k in [CUR_K,]:
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
    lbls = baofisher.load_param_names(root+"-fisher-full-0.dat")
    #zfns = ['b_HI',]
    #excl = ['Tb', 'f', 'aperp', 'apar', 'DA', 'H', 'fs8', 'bs8', 'gamma', 'N_eff']
    #F, lbls = baofisher.combined_fisher_matrix( F_list,
    #                                            expand=zfns, names=pnames,
    #                                            exclude=excl )
    
    pk = cosmo['pk_nobao'](kc) * (1. + fbao(kc))
    for j in range(len(F_list)):
        F = F_list[j]
        
        # Just do the simplest thing for P(k) and get 1/sqrt(F)
        cov = [np.sqrt(1. / np.diag(F)[lbls.index(lbl)]) for lbl in lbls if "pk" in lbl]
        cov = np.array(cov)
        
        scale = 1.2
        x = kc
        y = scale*fbao(x) + 0.18*zc[j]*10.
        err = scale*cov
        
        print j, zc[j], (zc[j] < 1.9)
        
        if zc[j] < 1.9:
            P.errorbar( x, y, yerr=err, color=colours[k], ls='none', 
                          lw=1., capthick=1., label=names[k], ms='.' )
            
            if j % 2 == 0:
                P.annotate( "%3.2f<z<%3.2f"%(zc[j]-0.05, zc[j]+0.05), 
                          xy=(4e-1, 0.18*zc[j]*10.), xytext=(0., 0.), 
                          fontsize='small', textcoords='offset points', 
                          ha='center', va='center' )
                          
        P.xscale('log')
        P.xlim((1.8e-2, 6e-1))
        P.ylim((-0.5, 3.8))
        #axes[k].set_title(labels[k], fontsize='x-large')
        
    #P.text(0.5, 1.03, labels[k], horizontalalignment='center', fontsize=19, 
    #       transform=P.gca().transAxes)
    #P.title(labels[k], fontsize=19)
    print "-"*50

    
    # Plot errorbars
    #yup, ydn = baofisher.fix_log_plot(pk, cov)

# Move subplots
# pos = [[x0, y0], [x1, y1]]
"""
l0 = 0.1
b0 = 0.15
ww = 0.85 / 2
hh = 0.75
for i in range(len(names))[::-1]:
    P.set_position([l0+ww*i, b0, ww, hh])
    
P.figtext(0.55, 0.05, "$k \,[\mathrm{Mpc}^{-1}]$", horizontalalignment='center', fontsize=19)
"""

P.xlabel("$k \,[\mathrm{Mpc}^{-1}]$", fontsize=19)

P.tick_params(axis='y', which='major', labelleft=False, size=0., width=1.5, pad=8.)
P.tick_params(axis='x', which='major', size=8., width=1.5, pad=8., 
        labelsize='x-large')
P.tick_params(axis='x', which='minor', size=5., width=1.5, pad=8., labelsize='large')


# Bin labels
"""
P.text(0.8, 0.95,'$0.0 < z < 0.1$', horizontalalignment='center', 
       transform=axes[0].transAxes, fontsize=18)

P.text(0.7, 0.35,'$0.65 < z < 0.75$', horizontalalignment='center', 
       transform=axes[1].transAxes, fontsize=18)
P.text(0.7, 0.9,'$1.75 < z < 1.85$', horizontalalignment='center', 
       transform=axes[1].transAxes, fontsize=18)
"""
P.tight_layout()

P.savefig('ska-bao-%s.pdf' % names[k], transparent=True)

P.show()
exit()

"""
    # Fix for PDF
    yup[np.where(yup > 1e1)] = 1e1
    ydn[np.where(ydn > 1e1)] = 1e1
    axes[k].errorbar( kc, fbao(kc), yerr=[ydn, yup], color=colours[k], ls='none', 
                      lw=1.8, capthick=1.8, label=names[k], ms='.' )

    # Plot f_bao(k)
    kk = np.logspace(-3., 1., 2000)
    axes[k].plot(kk, fbao(kk), 'k-', lw=1.8, alpha=0.6)
    
    # Set limits
    axes[k].set_xscale('log')
    axes[k].set_xlim((4e-3, 1e0))
    axes[k].set_ylim((-0.13, 0.13))
    #axes[k].set_ylim((-0.08, 0.08))
    
    axes[k].text( 0.39, 0.09, labels[k], fontsize=14, 
                  bbox={'facecolor':'white', 'alpha':1., 'edgecolor':'white',
                        'pad':15.} )


# Move subplots
# pos = [[x0, y0], [x1, y1]]
l0 = 0.15
b0 = 0.1
ww = 0.75
hh = 0.8 / 4.
for i in range(len(names))[::-1]:
    axes[i].set_position([l0, b0 + hh*i, ww, hh])
    
# Resize labels/ticks
for i in range(len(axes)):
    ax = axes[i]
    
    ax.tick_params(axis='both', which='major', labelsize=20, size=8., width=1.5, pad=8.)
    ax.tick_params(axis='both', which='minor', labelsize=20, size=5., width=1.5)
    
    if i != 0: ax.tick_params(axis='x', which='major', labelbottom='off')

axes[0].set_xlabel(r"$k \,[\mathrm{Mpc}^{-1}]$", fontdict={'fontsize':'xx-large'}, labelpad=10.)
#ax.set_ylabel(r"$P(k)$", fontdict={'fontsize':'20'})

# Set size
P.gcf().set_size_inches(8.5,12.)
#P.gcf().set_size_inches(8.5,10.)
P.savefig('pub-fbao.pdf', transparent=True)
P.show()
"""
