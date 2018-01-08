#!/usr/bin/python
"""
Plot functions of redshift for RSDs.
"""
import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
e = rf.experiments
eg = rf.experiments_galaxy
cosmo = rf.experiments.cosmo

fname = 'mg-fbsigma8-combined.pdf'

names = ['SKA1MID900_mg', 'SKA1MID350_mg', 'fSKA1SUR650_mg', 'fSKA1SUR350_mg', 
         'gSKAMIDMKB2_mg', 'gSKASURASKAP_mg', 'gSKA2_mg', 'EuclidRef_mg_Dz_kmg0.01']
labels = ['SKA1-MID 900 (IM)', 'SKA1-MID 350 (IM)', 'SKA1-SUR 650 (IM)', 
          'SKA1-SUR 350 (IM)', 'SKA1-MID (gal.)', 'SKA1-SUR (gal.)', 
          'SKA2 (gal.)', 'Euclid (gal.)']
expts = [e.SKA1MID900, e.SKA1MID350, e.SKA1SUR650, e.SKA1SUR350, 
         eg.SKAMIDMKB2, eg.SKASURASKAP, eg.SKA2, eg.EuclidRef]

colours = ['#8082FF', '#1619A1', '#FFB928', '#ff6600', '#95CD6D', '#007A10', '#CC0000', 
           '#000000', '#858585', '#c1c1c1']
linestyle = [[], [], [], [], [], [], [], [], []]
marker = ['s', 's', 's', 's', 'o', 'o', 'o', 'o', 'o']
ms = [6., 6., 6., 6., 6., 6., 5., 5., 5.]

# Fiducial value and plotting
fig = P.figure()
ax = [fig.add_subplot(311), fig.add_subplot(312), fig.add_subplot(313)]

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
    pnames = rf.load_param_names(root+"-fisher-full-0.dat")
    
    #zfns = ['A', 'b_HI', 'f', 'H', 'DA', 'aperp', 'apar']
    zfns = ['A', 'bs8', 'fs8', 'H', 'DA', 'aperp', 'apar']
    excl = ['Tb', 'n_s', 'sigma8', 'omegak', 'omegaDE', 'w0', 'wa', 'h',
            'gamma', 'N_eff', 'pk*', 'f', 'b_HI', 
            'gamma0', 'gamma1', 'eta0', 'eta1', 'A_xi', 'k_mg']
    
    # Marginalising over b_1
    F, lbls = rf.combined_fisher_matrix( F_list,
                                         expand=zfns, names=pnames,
                                         exclude=excl )
    cov = np.linalg.inv(F)
    errs = np.sqrt(np.diag(cov))
    
    # Fixing b_1
    excl.append('b_1')
    F2, lbls2 = rf.combined_fisher_matrix( F_list,
                                           expand=zfns, names=pnames,
                                           exclude=excl )
    cov2 = np.linalg.inv(F2)
    errs2 = np.sqrt(np.diag(cov2))
    
    # Identify functions of z
    pfs8 = rf.indices_for_param_names(lbls, 'fs8*')
    pfs8_2 = rf.indices_for_param_names(lbls2, 'fs8*')
    pbs8 = rf.indices_for_param_names(lbls, 'bs8*')
    pbs8_2 = rf.indices_for_param_names(lbls2, 'bs8*')
    
    """
    print ""
    print "#", names[k]
    print "# z, fsigma8, sigma(fsigma8)"
    for j in range(zc.size):
        print "%4.4f %5.5e %5.5e" % (zc[j], (cosmo['sigma_8']*fc*Dc)[j], errs[pfs8][j])
    """
    # FIXME: Disable to get redshift markers
    #marker[k] = None
    
    print lbls2
    print errs2
    
    # Load bias for experiment
    try:
        print names[k]
        expt = rf.experiments_galaxy.load_expt(expts[k])
        b = expt['b']
        print b
    except:
        print "IM survey!"
        b = rf.bias_HI(zc, e.cosmo)
        pass
    bs8 = cosmo['sigma_8']*b*Dc
    
    # (1) Plot bs8 errors as fn. of redshift (b_1 fixed)
    err2 = errs2[pbs8_2]
    print err2/bs8
    line = ax[2].plot( zc, err2/bs8, color=colours[k], label=labels[k], 
                       lw=2.4, ls='solid', marker='None', markersize=ms[k], 
                       markeredgecolor=colours[k] )
    ax[2].set_ylabel('$\sigma(b \sigma_8) / (b \sigma_8)$', labelpad=15., 
                 fontdict={'fontsize':'xx-large'})
    
    # (2) Plot fs8 errors as fn. of redshift (b_1 fixed)
    #err2 = errs2[pfs8_2] / (cosmo['sigma_8']*fc*Dc)
    #line = ax[1].plot( zc, err2, color=colours[k], label=labels[k], lw=1.8, 
    #                   ls='solid', marker=marker[k], markersize=ms[k],
    #                   markeredgecolor=colours[k] )
    #ax[1].set_ylabel('$\sigma(f \sigma_8) / (f \sigma_8)$', labelpad=15., 
    #             fontdict={'fontsize':'xx-large'})
    
    # (2) Plot fs8-bs8 correlation as fn. of redshift (b_1 fixed)
    err2 = errs2[pfs8_2] / (cosmo['sigma_8']*fc*Dc)
    r = cov2[pfs8_2, pbs8_2] / np.sqrt(cov2[pfs8_2, pfs8_2] * cov2[pbs8_2, pbs8_2])
    line = ax[0].plot( zc, r, color=colours[k], label=labels[k], lw=2.4, 
                       ls='solid', marker='None', markersize=ms[k],
                       markeredgecolor=colours[k] )
    ax[0].set_ylabel(r'$\rho(f\sigma_8, b\sigma_8)$', labelpad=15., 
                 fontdict={'fontsize':'xx-large'})
    
    # (3a) Plot fs8 errors as fn. of redshift (b_1 marginalised)
    err = errs[pfs8] / (cosmo['sigma_8']*fc*Dc)
    line = ax[1].plot( zc, err, color=colours[k], lw=2.4, ls='dashed', 
                       marker='None', markersize=ms[k], 
                       markeredgecolor=colours[k] )
    # (3b) Plot fs8 errors as fn. of redshift (b_1 fixed)
    err2 = errs2[pfs8_2] / (cosmo['sigma_8']*fc*Dc)
    line = ax[1].plot( zc, err2, color=colours[k], label=labels[k], lw=2.4, 
                       ls='solid', marker='None', markersize=ms[k],
                       markeredgecolor=colours[k] )
    ax[1].set_ylabel('$\sigma(f \sigma_8) / (f \sigma_8)$', labelpad=15., 
                 fontdict={'fontsize':'xx-large'})

# Load actual f.sigma_8 data and plot it
#dat = np.genfromtxt("fsigma8_data.dat").T
#ax[1].plot(dat[1], dat[3], 'kD')
#ax[0].plot(dat[1], dat[3], 'kD')

# Set common axis properties 
for _ax in ax:
    _ax.tick_params(axis='both', which='major', labelsize=20, labelbottom=False, 
                    width=1.5, size=8., pad=10)
    _ax.tick_params(axis='both', which='minor', width=1.5, size=5., pad=10)
    
    # Set tick locations
    _ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.01))
    _ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.01))

    # Set axis limits
    _ax.set_xlim((-0.02, 2.6))

# Labels only on bottom panel
ax[0].tick_params(axis='both', which='major', labelbottom=True, labelsize=20, width=1.5, size=8., pad=10)
ax[0].tick_params(axis='both', which='minor', labelsize=20, width=1.5, size=8.)

ax[0].set_ylim((-0.8, 0.52))
ax[1].set_ylim((0.0, 0.059))
ax[2].set_ylim((0.0, 0.059))

ax[0].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
ax[0].yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
ax[0].axhline(0., color='k', ls='dashed', lw=1.5)

ax[0].set_xlabel('$z$', labelpad=10., fontdict={'fontsize':'xx-large'})

# Set tick locations
#P.gca().yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.01))
#P.gca().yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.01))
    
leg = ax[2].legend(prop={'size':'large'}, loc='upper right', frameon=True, ncol=2)
leg.get_frame().set_edgecolor('w')
leg.get_frame().set_alpha(0.8)

# Set size
#P.gcf().set_size_inches(9.5, 6.8)
P.gcf().set_size_inches(9., 15.5)

# Move subplots
# pos = [[x0, y0], [x1, y1]]
l0 = 0.15
b0 = 0.07
ww = 0.8
hh = 0.9 / 3.
for i in range(len(ax))[::-1]:
    ax[i].set_position([l0, b0 + hh*i, ww, hh])

P.savefig(fname, transparent=True)
P.show()
