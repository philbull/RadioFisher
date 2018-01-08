#!/usr/bin/python
"""
Plot functions of redshift.
"""
import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm

cosmo = rf.experiments.cosmo

#-------------------------------------------------------------------------------
# SKA galaxy survey chapter H, D_A constraints
#-------------------------------------------------------------------------------
"""
filename = "hirax-hda.pdf"
names = [ 'iHIRAX_hrx_kmax0.14', 'iHIRAX_hrx_kmax0.20', 'iHIRAX_hrx_kmax0.30', 
          'iHIRAX_hrx_kmax1.00', 'iHIRAX_hrx_kmax10.00', 
          'iCVTEST1_hrx_kmax0.14', 'iCVTEST1_hrx_kmax0.20', 'iCVTEST1_hrx_kmax0.30', 
          'iCVTEST1_hrx_kmax1.00', 'iCVTEST1_hrx_kmax10.00', 
          ]
labels = [ 'HIRAX 0.14', 'HIRAX 0.2', 'HIRAX 0.3', 'HIRAX 1.0', 'HIRAX 10.0',
           'CV1 0.14', 'CV1 0.2', 'CV1 0.3', 'CV1 1.0', 'CV1 10.0' ]
"""

filename = "hirax-test.pdf"
#names = [ 'iHIRAX_hrx_kmax0.14', 'iCVTEST1_hrx_kmax0.14', 
#          'ihirax_amadeus_21000_nubin20.0_ttot8760hours',
#          'iHIRAX_hrx_kmax0.14_amadeus',
#          ]
#labels = [ 'HIRAX 0.14', 'CV1 0.14', 'iHIRAX Amadeus', 'HIRAX 0.14 Amadeusttot' ]

names = ['iHIRAX_hrx_opt_500', 'iHIRAX_hrx_opt_1000', 'iHIRAX_hrx_opt_2000',
         'iHIRAX_hrx_opt_4000', 'iHIRAX_hrx_opt_6000', 'iHIRAX_hrx_opt_8000',
         'iHIRAX_hrx_opt_10000', 'iHIRAX_hrx_opt_12000', 'iHIRAX_hrx_opt_15000', 
         'iHIRAX_hrx_opt_18000', ]
labels = ['iHIRAX 500', 'iHIRAX 1000', 'iHIRAX 2000', 'iHIRAX 4000', 
          'iHIRAX 6000', 'iHIRAX 8000', 'iHIRAX 10000', 'iHIRAX 12000',
          'iHIRAX 15000', 'iHIRAX 18000' ]

colours = ['#1619A1', '#FFB928', '#CC0000', 
           '#3D3D3D', '#858585', #'#c1c1c1',
           '#1619A1', '#FFB928', '#CC0000', 
           '#3D3D3D', '#858585', '#c1c1c1']
linestyle = [[2,2], [], [], [3,3], [], [], [], [], [], [], [], []]
marker = ['s', 's', 's', 's', 's', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
ms = np.ones(10)*4.5
#ymax = [0.012, 0.012]
ymax = [0.034, 0.034]


################################################################################
# Fiducial value and plotting
fig = P.figure()
axes = [fig.add_subplot(121), fig.add_subplot(122)]

for k in range(len(names)):
  try:
    root = "output/" + names[k]

    # Load cosmo fns.
    dat = np.atleast_2d( np.genfromtxt(root+"-cosmofns-zc.dat") ).T
    zc, Hc, dAc, Dc, fc = dat
    z, H, dA, D, f = np.genfromtxt(root+"-cosmofns-smooth.dat").T
    kc = np.genfromtxt(root+"-fisher-kc.dat").T

    # Load Fisher matrices as fn. of z
    Nbins = zc.size
    F_list = [np.genfromtxt(root+"-fisher-full-%d.dat" % i) for i in range(Nbins)]
    pnames = rf.load_param_names(root+"-fisher-full-0.dat")
    
    #zfns = ['A', 'b_HI', 'f', 'DV', 'F']
    zfns = ['A', 'bs8', 'fs8', 'DA', 'H']
    excl = ['Tb', 'sigma8', 'n_s', 'omegak', 'omegaDE', 'w0', 'wa', 'h', 
            'gamma', 'N_eff', 'pk*', 'f', 'b_HI', 'sigma_8', 'sigma8tot']
    F, lbls = rf.combined_fisher_matrix( F_list, expand=zfns, names=pnames,
                                         exclude=excl )
    cov = np.linalg.inv(F)
    errs = np.sqrt(np.diag(cov))
    print "="*50
    print names[k]
    print lbls
    print "="*50
    
    # Identify functions of z
    pDA = rf.indices_for_param_names(lbls, 'DA*')
    pH = rf.indices_for_param_names(lbls, 'H*')
    
    errDA = 1e3 * errs[pDA] / dAc
    errH = 1e2 * errs[pH] / Hc
    indexes = [pDA, pH]
    err = [errDA, errH]
    
    # Plot errors as fn. of redshift
    for jj in range(len(axes)):
        line = axes[jj].plot( zc, err[jj], color=colours[k], lw=1.8, 
                              marker=marker[k], ms=ms[k], label=labels[k] )
        line[0].set_dashes(linestyle[k])
  except:
    print ">>> FAILED:", names[k]
    raise
    pass

# Subplot labels
ax_lbls = ["$\sigma_{D_A}/ D_A$", "$\sigma_H/H$"]


# Move subplots
# pos = [[x0, y0], [x1, y1]]
l0 = 0.15
b0 = 0.1
ww = 0.8
hh = 0.88 / 2.
j = 0
for i in range(len(axes)):
    axes[i].set_position([l0, b0 + hh*i, ww, hh])
    
    axes[i].tick_params(axis='both', which='major', labelsize=14, width=1.5, size=8.)
    axes[i].tick_params(axis='both', which='minor', labelsize=14, width=1.5, size=8.)
    
    if i == 1: axes[i].tick_params(axis='both', which='major', labelbottom='off')
    axes[i].tick_params(axis='both', which='major', labelsize=20.)
    
    # Set axis limits
    axes[i].set_xlim((0., 2.8))
    axes[i].set_ylim((0., ymax[i]))
    
    if i == 0: axes[i].set_xlabel('$z$', labelpad=10., fontdict={'fontsize':'xx-large'})
    axes[i].set_ylabel(ax_lbls[i], labelpad=15., fontdict={'fontsize':'xx-large'})
    
    # Set tick locations
    axes[i].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.01))
    axes[i].yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.005))

leg = P.legend(prop={'size':'medium'}, loc='upper right', frameon=True, ncol=2)
leg.get_frame().set_alpha(0.8)
leg.get_frame().set_edgecolor('w')

# Set size
P.gcf().set_size_inches(8.4, 7.8)
#P.tight_layout()

#P.savefig(filename, transparent=True)
P.show()
