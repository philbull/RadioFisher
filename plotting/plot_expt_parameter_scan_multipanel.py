#!/usr/bin/python
"""
Plot several figures of merit as function of experimental settings.
(Fig. 21) (Fig. 22) (Fig. 23) (Fig. 24) (Fig. 25) (Fig. 26).
"""
import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
from radiofisher.units import *
import os, sys
from radiofisher import euclid

cosmo = rf.experiments.cosmo

# Define names of parameters being varied
snames = ['ttot', 'Sarea', 'epsilon_fg', 'omega_HI_0', 'kfg_fac', 'sigma_nl']
slabels = ['$t_\mathrm{tot} [10^3 \mathrm{hrs}]$',
           '$S_\mathrm{area} [10^3 \mathrm{deg}^2]$', 
           '$\epsilon_\mathrm{FG}$',
           '$\Omega_{\mathrm{HI},0} / 10^{-4}$',
           '$k_\mathrm{FG} / k_{\mathrm{FG}, 0}$',
           '$\sigma_\mathrm{NL} \, [\mathrm{Mpc}]$']
logscale = [False, False, True, False, False, False]
fname = ['fig25-ttot.pdf', 'fig26-sarea.pdf', 'fig23-efg.pdf', 
         'fig21-omegaHI.pdf', 'fig24-kfg.pdf', 'fig22-signl.pdf']
fac = [1e3 * HRS_MHZ, 1e3 * (D2RAD)**2., 1., 
       1e-4, 1., 1.] # Divide by this factor to get sensible units

markers = ['o', 'D', 's']

# Choose which parameter to plot
if len(sys.argv) > 1:
    j = int(sys.argv[1])
else:
    raise IndexError("Need to specify ID of parameter to plot.")


# Experiments
names = ['aexptM_paper', 'exptL_paper']
#colours = ['#990A9C', '#5B9C0A', '#1619A1', '#CC0000']
colours = ['#5B9C0A', '#1619A1', '#CC0000']

#colours = [['#440445', '#990a9c', '#ec0ef0'],
#           ['#284504', '#5b9c0a', '#8af00e'],
#           ['#0a0b4a', '#1619A1', '#2226f5']]

#linestyle = [[3, 4], [8, 4], [1,0], [2, 4, 6, 4]]
linestyle = [[8, 4], [],]
lw = [1.2, 1.8, 2.2]

# Fiducial value and plotting
fig = P.figure()
ax = fig.add_subplot(111)

for k in range(len(names)):
    # Load array of values for varied parameter
    fname_vals = "output/%s_%s_values.txt" % (names[k], snames[j])
    param_values_expt = np.genfromtxt(fname_vals).T
    
    # Load cosmo fns.
    mainroot = "output/%s_%s" % (names[k], snames[j])
    dat = np.atleast_2d( np.genfromtxt(mainroot+"-cosmofns-zc.dat") ).T
    zc, Hc, dAc, Dc, fc = dat
    
    # Loop through values of varying parameter
    fom_values_expt = []
    sk_values_expt = []
    sg_values_expt = []
    for v in range(param_values_expt.size):
        root = "output/%s_%s_%d" % (names[k], snames[j], v)

        # Load Fisher matrices as fn. of z
        Nbins = zc.size
        F_list = [np.genfromtxt(root+"-fisher-full-%d.dat" % i) for i in range(Nbins)]
        
        # EOS FISHER MATRIX
        # Actually, (aperp, apar) are (D_A, H)
        pnames = rf.load_param_names(root+"-fisher-full-0.dat")
        zfns = ['b_HI',]
        excl = ['Tb', 'f', 'H', 'DA', 'apar', 'aperp', 'pk*', 'fs8', 'bs8', 'N_eff',]
        F, lbls = rf.combined_fisher_matrix( F_list, expand=zfns, names=pnames,
                                             exclude=excl )
        # DETF Planck prior
        #print "*** Using DETF Planck prior ***"
        l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h', 'sigma8']
        F_detf = euclid.detf_to_rf("DETF_PLANCK_FISHER.txt", cosmo, omegab=False)
        Fpl, lbls = rf.add_fisher_matrices(F, F_detf, lbls, l2, expand=True)
        
        # Get indices of w0, wa
        pw0 = lbls.index('w0'); pwa = lbls.index('wa'); pA = lbls.index('A')
        pok = lbls.index('omegak'); pgam = lbls.index('gamma')
        
        # Calculate FOM
        cov_pl = np.linalg.inv(Fpl)
        
        fom = rf.figure_of_merit(pw0, pwa, None, cov=cov_pl)
        fom_values_expt.append(fom)
        sk_values_expt.append(1./np.sqrt(cov_pl[pok,pok]))
        sg_values_expt.append(1./np.sqrt(cov_pl[pgam,pgam]))
        
        print "%s: FOM = %3.2f, sig(A) = %3.3f" % (names[k], fom, 
                                                   np.sqrt(cov_pl[pA,pA]))
        print ">>> Paramname:", snames[j], " -- val:", param_values_expt[v]
    
    # Sort values
    idxs = param_values_expt.argsort()
    param_values_expt = param_values_expt[idxs]
    fom_values_expt = np.array(fom_values_expt)[idxs]
    sk_values_expt = np.array(sk_values_expt)[idxs]
    sg_values_expt = np.array(sg_values_expt)[idxs]
    
    # Plot line for this parameter
    y = [fom_values_expt / np.max(fom_values_expt), 
         sk_values_expt / np.max(sk_values_expt),
         sg_values_expt / np.max(sg_values_expt)]
    for n in range(3):
        line = ax.plot(param_values_expt/fac[j], y[n], color=colours[k], lw=lw[n], marker=markers[n])
        line[0].set_dashes(linestyle[k])

# Legend
bbox = [[0.92,0.45],
        [0.92,0.45],
        [0.42,0.45],
        [0.92,0.45],
        [0.39,0.45],
        [0.39,0.45]]
labels = ['DE FOM', '$(\sigma_K)^{-1}$', '$(\sigma_\gamma)^{-1}$']
lines = [ matplotlib.lines.Line2D([0.,], [0.,], lw=lw[k], marker=markers[k], color='k', alpha=1.) for k in range(3)]
P.gcf().legend((l for l in lines), (name for name in labels), prop={'size':'large'}, bbox_to_anchor=bbox[j])

ax.set_ylabel("$\mathrm{FOM} / \mathrm{FOM}|_\mathrm{max}$", fontdict={'fontsize':'xx-large'}, labelpad=15.)
ax.set_xlabel(slabels[j], fontdict={'fontsize':'xx-large'}, labelpad=15.)
ax.tick_params(axis='both', which='major', labelsize=20, size=8., width=1.5, pad=8.)
ax.tick_params(axis='both', which='minor', size=4., width=1.2)

if j == 0: ax.set_xlim((0., 20.2))
if j == 1: ax.set_xlim((0., 30.4))
if j == 2: ax.set_xlim((8e-8, 1.3e-4))
if j == 3: ax.set_xlim((1.8, 11.2))
if j == 4: ax.set_xlim((9e-2, 1.3e1))
if j == 5: ax.set_xlim((1.4, 14.5))

if j == 0: ax.set_ylim((0., 1.02))
if j == 1: ax.set_ylim((0., 1.05))
if j == 2: ax.set_ylim((7e-3, 1.5))
if j == 3: ax.set_ylim((3e-3, 1.2))
if j == 4: ax.set_ylim((0.775, 1.01))
if j == 5: ax.set_ylim((0.2, 1.05))

if j == 2: ax.set_xscale('log')
if j == 4: ax.set_xscale('log')

if j == 2: ax.set_yscale('log')
if j == 3: ax.set_yscale('log')

if j == 3: ax.axvline(cosmo['omega_HI_0']/1e-4, color='k', ls='dotted', alpha=0.5, lw=1.8)
if j == 2: ax.axvline(1e-6, color='k', ls='dotted', alpha=0.5, lw=1.8)
if j == 4: ax.axvline(1., color='k', ls='dotted', alpha=0.5, lw=1.8)
if j == 5: ax.axvline(cosmo['sigma_nl'], color='k', ls='dotted', alpha=0.5, lw=1.8)

# Set size
P.tight_layout()
P.gcf().set_size_inches(8., 6.)
P.savefig(fname[j], transparent=True)
P.show()
