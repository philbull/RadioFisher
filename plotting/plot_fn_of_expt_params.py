#!/usr/bin/python
"""
Plot figures of merit as function of experimental settings.
"""
import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
from units import *
from mpi4py import MPI

import os
import euclid

USE_DETF_PLANCK_PRIOR = True

cosmo = rf.experiments.cosmo

# Define names of parameters being varied
snames = ['ttot', 'Sarea', 'epsilon_fg', 'omega_HI_0']
slabels = ['$t_\mathrm{tot} [10^3 \mathrm{hrs}]$', '$S_\mathrm{area} [10^3 \mathrm{deg}^2]$', 
          '$\epsilon_\mathrm{FG}$', '$\Omega_\mathrm{HI} / 10^{-4}$']
logscale = [False, False, True, False]
fac = [1e3 * HRS_MHZ, 1e3 * (D2RAD)**2., 1., 1e-4] # Divide by this factor to get sensible units

names = ['cexptL',]
colours = ['#1619A1',]
linestyle = [[1,0],]

names = ['exptS', 'iexptM', 'cexptL']
colours = ['#990A9C', '#5B9C0A', '#1619A1', '#CC0000']
linestyle = [[3, 4], [8, 4], [1,0], [2, 4, 6, 4]]


# Fiducial value and plotting
fig = P.figure()
axes = [fig.add_subplot(221), fig.add_subplot(222), fig.add_subplot(223), 
        fig.add_subplot(224)]


# Loop through experimental params., for each experiment
for j in range(4):
    for k in range(len(names)):
        # Load array of values for varied parameter
        fname_vals = "../output/%s_%s_values.txt" % (names[k], snames[j])
        param_values_expt = np.genfromtxt(fname_vals).T
        
        # Load cosmo fns.
        mainroot = "../output/%s_%s" % (names[k], snames[j])
        dat = np.atleast_2d( np.genfromtxt(mainroot+"-cosmofns-zc.dat") ).T
        zc, Hc, dAc, Dc, fc = dat
        
        # Loop through values of varying parameter
        fom_values_expt = []
        for v in range(param_values_expt.size):
            root = "../output/%s_%s_%d" % (names[k], snames[j], v)

            # Load Fisher matrices as fn. of z
            Nbins = zc.size
            F_list = [np.genfromtxt(root+"-fisher-full-%d.dat" % i) for i in range(Nbins)]
            
            # EOS FISHER MATRIX
            # Actually, (aperp, apar) are (D_A, H)
            pnames = rf.load_param_names(root+"-fisher-full-0.dat")
            zfns = [1,]
            excl = [2,   6,7,8,  14,]
            excl  += [i for i in range(len(pnames)) if "pk" in pnames[i]]
            
            F, lbls = rf.combined_fisher_matrix( F_list,
                                                        expand=zfns, names=pnames,
                                                        exclude=excl )
            # Add Planck prior
            if USE_DETF_PLANCK_PRIOR:
                # DETF Planck prior
                #print "*** Using DETF Planck prior ***"
                l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h', 'sigma8']
                F_detf = euclid.detf_to_rf("DETF_PLANCK_FISHER.txt", 
                                                   cosmo, omegab=False)
                Fpl, lbls = rf.add_fisher_matrices(F, F_detf, lbls, l2, expand=True)
            else:
                # Euclid Planck prior
                #print "*** Using Euclid (Mukherjee) Planck prior ***"
                l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h']
                Fe = euclid.planck_prior_full
                F_eucl = euclid.euclid_to_rf(Fe, cosmo)
                Fpl, lbls = rf.add_fisher_matrices(F, F_eucl, lbls, l2, expand=True)
            
            # Get indices of w0, wa
            pw0 = lbls.index('w0'); pwa = lbls.index('wa'); pA = lbls.index('A')
            
            # Calculate FOM
            cov_pl = np.linalg.inv(Fpl)
            fom = rf.figure_of_merit(pw0, pwa, None, cov=cov_pl)
            fom_values_expt.append(fom)
            #fom_values_expt.append(1./np.sqrt(cov_pl[pA,pA])) # FIXME
            print "%s: FOM = %3.2f, sig(A) = %3.3f" % (names[k], fom, 
                                                       np.sqrt(cov_pl[pA,pA]))
            print ">>> Paramname:", snames[j], " -- val:", param_values_expt[v]
        
        # Sort values
        idxs = param_values_expt.argsort()
        param_values_expt = param_values_expt[idxs]
        fom_values_expt = np.array(fom_values_expt)[idxs]
        
        # Plot line for this parameter
        line = axes[j].plot(param_values_expt/fac[j], fom_values_expt/np.max(fom_values_expt), label=names[k], color=colours[k], lw=1.8, marker='o')
        line[0].set_dashes(linestyle[k])
    
    # Settings for each subplot
    #axes[j].set_xlabel(slabels[j])
    #axes[j].set_ylabel("FOM")
    if logscale[j]: axes[j].set_xscale('log')
    
    if j == 0 or j == 1:
        axes[j].tick_params(axis='both', which='both', labelleft='off', left='on', right='on', bottom='off', top='on', labeltop='on', labelbottom='off', length=6., width=1.5)
        #axes[j].xaxis.set_label_position('top')
    #axes[j].set_yscale('log')
    #axes[j].legend(loc='upper right', prop={'size':'xx-small'})

# Move subplots
# pos = [[x0, y0], [x1, y1]]
l0 = 0.1
b0 = 0.1
ww = 0.4
hh = 0.8 / 2.
j = 0
for i in range(len(axes)):

    axes[i].set_position([l0 + ww*j, b0 + hh*(i%2), ww, hh])
    
    #axes[i].tick_params(axis='both', which='major', labelsize=14)
    #axes[i].tick_params(axis='both', which='minor', labelsize=14)
    
    # Set axis limits
    #axes[i].set_xlim((0.25, 2.2))
    #axes[i].set_ylim((0., ymax[i]))
    
    # Add label to panel
    P.figtext(l0 + ww*j + 0.02, b0 + hh*(0.86+(i%2)), slabels[i], 
              fontdict={'size':'x-large'}) #, bbox=dict(ec='k', fc='none', lw=1.2))
    
    """
    # Fix ticks
    if j==1:
        axes[i].yaxis.tick_right()
        axes[i].yaxis.set_label_position("right")
    if i%2 == 1:
        for tick in axes[i].xaxis.get_major_ticks():
            tick.label1.set_visible(False)
    """
    if i % 2 == 1: j += 1
    """
    # Hide alternating ticks
    kk = 0
    for tick in axes[i].yaxis.get_major_ticks():
        if kk%2==1:
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
        kk += 1
    """

P.show()
exit()

# Manually add shared x label
#P.figtext(0.5, 0.02, "$z$", fontdict={'size':'xx-large'})

# Set size
P.gcf().set_size_inches(10., 7.)
P.savefig('pub-expt-settings.pdf', transparent=True)
P.show()
